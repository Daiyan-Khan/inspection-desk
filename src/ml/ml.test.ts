import { describe, expect, it, vi } from 'vitest';
import { calibrationThreshold, scoreFeatures, selectBankPatches } from './scoring';
import { letterboxTransform, modelInput, patchFeatures, validPatchMask } from './preprocessing';
import { classicalFeatures, fitStandardization, pixelFeatures, standardize } from './baseline';
import { checkedFetch, decodeBank, DIMENSIONS, MODEL_SHA256, sha256, validateManifest } from './artifacts';
import { MEAN, MODEL, PADDING, PATCHES, PIXEL_DIMENSIONS, SIZE, STD } from './config';
import type { ArtifactManifest } from '../types';

describe('canonical model input', () => {
  it('normalises RGB into NCHW using the exact channel statistics', () => {
    const rgb = new Uint8Array(SIZE * SIZE * 3);
    rgb.set([255, 128, 0]);
    const input = modelInput(rgb);
    expect(input[0]).toBeCloseTo((1 - MEAN[0]) / STD[0], 6);
    expect(input[SIZE * SIZE]).toBeCloseTo((128 / 255 - MEAN[1]) / STD[1], 6);
    expect(input[2 * SIZE * SIZE]).toBeCloseTo(-MEAN[2] / STD[2], 6);
    expect(() => modelInput(new Uint8Array(12))).toThrow();
  });
  it('excludes every patch crossing the actual letterboxed image boundary', () => {
    const transform = letterboxTransform(1000, 500);
    expect(transform).toMatchObject({ resizedWidth: 224, resizedHeight: 112, padLeft: 0, padTop: 56 });
    const valid = validPatchMask(transform);
    expect(valid.filter(Boolean)).toHaveLength(128);
    expect(valid.slice(0, 64).every(x => !x)).toBe(true);
    const narrow = letterboxTransform(1000, 533);
    validPatchMask(narrow).forEach((value, i) => {
      if (value) expect(Math.floor(i / 16) * 14).toBeGreaterThanOrEqual(narrow.padTop);
    });
  });
  it('discards CLS and normalises each spatial feature independently', () => {
    const output = new Float32Array(257 * 384);
    output.fill(99, 0, 384);
    for (let p = 1; p < 257; p++) { output[p * 384] = 3; output[p * 384 + 1] = 4; }
    const features = patchFeatures(output, [1, 257, 384]);
    expect(features.length).toBe(256 * 384);
    expect(features[0]).toBeCloseTo(0.6);
    expect(features[1]).toBeCloseTo(0.8);
    expect(features[2]).toBe(0);
    expect(() => patchFeatures(output, [1, 256, 384])).toThrow();
    output[400] = NaN;
    expect(() => patchFeatures(output, [1, 257, 384])).toThrow();
  });
});

describe('reference banks and calibration', () => {
  it('uses the same deterministic locations independent of input image order', () => {
    const images = Array.from({ length: 8 }, (_, index) => ({ id: `fit-${index}`, valid: Array(256).fill(true) as boolean[] }));
    const selection = selectBankPatches(images, 16);
    expect([...selection]).toEqual([...selectBankPatches([...images].reverse(), 16)]);
    expect([...selection.values()].flat()).toHaveLength(16);
    for (const selected of selection.values()) expect(new Set(selected).size).toBe(selected.length);
    expect(selection.size).toBe(8);
  });
  it('scores only eligible patches and averages the highest five distances', () => {
    const features = new Float32Array(256);
    features.set([100, 1, 2, 3, 4, 5, 6]);
    const valid = Array<boolean>(256).fill(false); valid.fill(true, 1, 7);
    const result = scoreFeatures(features, new Float32Array([0]), 1, valid);
    expect(result.heatmap[0]).toBe(0);
    expect(result.score).toBe(4);
  });
  it('uses a finite-sample normal calibration quantile with strict exceedance', () => {
    const scores = Array.from({ length: 180 }, (_, i) => i + 1);
    const threshold = calibrationThreshold(scores);
    expect(threshold).toBe(172);
    expect(scores.filter(score => score > threshold)).toHaveLength(8);
    expect(() => calibrationThreshold([1, 2])).toThrow();
  });
  it('rejects corrupted bank length and non-finite values', () => {
    expect(() => decodeBank(new ArrayBuffer(8), 3, 1)).toThrow();
    expect(() => decodeBank(new Float32Array([1, NaN, 3]).buffer, 3, 1)).toThrow();
  });
  it('rejects artifacts made with the superseded native backend even when tensor shapes match', () => {
    const manifest = { schemaVersion: 1, version: 'candle-q8-wasm-v1-0123456789ab', splitVersion: 'frozen-split',
      model: { id: MODEL.id, revision: MODEL.revision, dtype: 'q8', sha256: MODEL_SHA256, localPath: 'models/model.onnx' },
      preprocessing: { size: 224, patchSize: 14, padding: [...PADDING], mean: [...MEAN], std: [...STD] }, createdAt: 'test',
      methods: Object.fromEntries(Object.entries(DIMENSIONS).map(([method, dimensions]) => [method, {
        bankUrl: `artifacts/${method}.bin`, bankSha256: '0'.repeat(64), dimensions, count: 1024, threshold: 0.5, heatmapScale: 1,
        ...(method === 'classical' ? { standardizationUrl: 'artifacts/stats.json', standardizationSha256: '1'.repeat(64) } : {}),
      }])) } as ArtifactManifest;
    expect(() => validateManifest(manifest)).not.toThrow();
    expect(() => validateManifest({ ...manifest, version: 'candle-q8-v1-0123456789ab' })).toThrow();
  });
  it('checks downloaded bytes against the expected artifact hash and rejects corruption', async () => {
    const bytes = new Uint8Array([11, 22, 33, 44]).buffer;
    const digest = await sha256(bytes);
    vi.stubGlobal('fetch', vi.fn(async () => new Response(bytes)));
    try {
      expect(new Uint8Array(await checkedFetch('https://example.invalid/bank.bin', digest))).toEqual(new Uint8Array(bytes));
      await expect(checkedFetch('https://example.invalid/bank.bin', '0'.repeat(64))).rejects.toThrow('integrity');
    } finally { vi.unstubAllGlobals(); }
  });
});

describe('conventional baseline controls', () => {
  const transform = letterboxTransform(224, 224);
  it('retains raw patch RGB in the pixel ablation', () => {
    const rgb = new Uint8Array(SIZE * SIZE * 3).fill(255);
    rgb.set([0, 128, 255]);
    const features = pixelFeatures(rgb);
    expect(features.length).toBe(PATCHES * PIXEL_DIMENSIONS);
    expect(features[0]).toBe(0); expect(features[1]).toBeCloseTo(128 / 255); expect(features[2]).toBe(1);
    expect(features[PIXEL_DIMENSIONS]).toBe(1);
  });
  it('has zero texture features on uniform colour at all three scales', () => {
    const features = classicalFeatures(new Uint8Array(SIZE * SIZE * 3).fill(127), transform);
    for (let p = 0; p < PATCHES; p++) for (let scale = 0; scale < 3; scale++) {
      expect(features[p * 24 + scale * 8 + 3]).toBe(0);
      for (let d = 4; d < 8; d++) expect(features[p * 24 + scale * 8 + d]).toBeLessThan(1e-6);
    }
  });
  it('fits feature scaling on its supplied normal bank only and handles constant dimensions', () => {
    const bank = new Float32Array([0, 2, 2, 2]);
    const stats = fitStandardization(bank, 2);
    expect(stats).toEqual({ mean: [1, 2], std: [1, 0.001] });
    expect([...standardize(bank, 2, stats)]).toEqual([-1, 0, 1, 0]);
  });
});
