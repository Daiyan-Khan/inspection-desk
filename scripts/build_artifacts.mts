import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, copyFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import sharp from 'sharp';
import type { ArtifactManifest, Method, Transform } from '../src/types.ts';
import { DIMENSIONS, MODEL_SHA256, validateManifest } from '../src/ml/artifacts.ts';
import { classicalFeatures, fitStandardization, pixelFeatures, standardize } from '../src/ml/baseline.ts';
import { BANK_CAP, MEAN, MODEL, PADDING, PATCHES, SIZE, STD } from '../src/ml/config.ts';
import { letterboxTransform, validPatchMask } from '../src/ml/preprocessing.ts';
import { calibrationThreshold, quantile, scoreFeatures, selectBankPatches } from '../src/ml/scoring.ts';
import { createModel, RUNTIME_VERSION } from './ml_runtime.mts';

type SplitEntry = { id: string; split: 'fit' | 'calibration' | 'test'; label: 'normal' | 'anomaly'; imagePath: string; sha256: string };
type Canonical = { id: string; path: string; transform: Transform; sha256: string; validPatches: boolean[] };
const root = process.cwd();
const dataRoot = path.join(root, 'data/visa/candle');
const publicRoot = path.join(root, 'public');
const artifactRoot = path.join(publicRoot, 'artifacts');
const methods: Method[] = ['dinov2', 'classical', 'pixel'];
const hash = (data: string | Uint8Array) => createHash('sha256').update(data).digest('hex');
const asBytes = (data: Float32Array) => Buffer.from(data.buffer, data.byteOffset, data.byteLength);
const json = (value: unknown) => JSON.stringify(value, null, 2) + '\n';
async function exists(file: string) { try { return await readFile(file); } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null; throw error; } }

await mkdir(artifactRoot, { recursive: true });
await mkdir(path.join(dataRoot, 'canonical'), { recursive: true });
await mkdir(path.join(dataRoot, 'feature-cache'), { recursive: true });
const modelPath = path.join(publicRoot, 'models', MODEL.file);
let modelBytes = await exists(modelPath);
if (!modelBytes) {
  console.log(`Downloading model pinned to ${MODEL.revision}…`);
  const response = await fetch(MODEL.url);
  if (!response.ok) throw new Error(`Model download failed: ${response.status}`);
  modelBytes = Buffer.from(await response.arrayBuffer());
  await mkdir(path.dirname(modelPath), { recursive: true });
  await writeFile(modelPath, modelBytes);
}
if (hash(modelBytes) !== MODEL_SHA256) throw new Error('Pinned model SHA-256 does not match.');
const splitBytes = await readFile(path.join(dataRoot, 'split.json'));
const split = JSON.parse(splitBytes.toString()) as { entries: SplitEntry[]; splitVersion?: string };
const splitVersion = split.splitVersion ?? hash(splitBytes);
const fit = split.entries.filter(x => x.split === 'fit').sort((a, b) => a.id.localeCompare(b.id));
const calibration = split.entries.filter(x => x.split === 'calibration').sort((a, b) => a.id.localeCompare(b.id));
const test = split.entries.filter(x => x.split === 'test').sort((a, b) => a.id.localeCompare(b.id));
if (fit.length !== 719 || calibration.length !== 180 || test.length !== 200 ||
    [...fit, ...calibration].some(x => x.label !== 'normal') || new Set(split.entries.map(x => x.id)).size !== 1099) {
  throw new Error('Expected frozen 719 normal fit / 180 normal calibration / 200 test split after near-duplicate removal.');
}
const ortPackage = JSON.parse(await readFile(path.join(root, 'node_modules/onnxruntime-web/package.json'), 'utf8'));
if (ortPackage.version !== '1.22.0') throw new Error('This artifact recipe requires ONNX Runtime Web1.22.0.');
const kernelFiles = ['src/ml/preprocessing.ts', 'src/ml/baseline.ts', 'src/ml/scoring.ts', 'src/ml/config.ts', 'scripts/ml_runtime.mts', 'scripts/build_artifacts.mts'];
const kernelHash = hash(Buffer.concat(await Promise.all(kernelFiles.map(file => readFile(path.join(root, file))))));
const version = `candle-q8-wasm-v1-${hash(splitVersion + MODEL_SHA256 + kernelHash).slice(0, 12)}`;
console.log(`Building ${version}: 719 fit, 180 calibration, 200 test.`);

const canonical: Canonical[] = [];
for (let i = 0; i < split.entries.length; i++) {
  const entry = split.entries[i];
  const original = await readFile(path.join(root, entry.imagePath));
  if (hash(original) !== entry.sha256) throw new Error(`Dataset integrity failed: ${entry.id}`);
  const metadata = await sharp(original).metadata();
  if (!metadata.width || !metadata.height) throw new Error(`Missing image dimensions: ${entry.id}`);
  const transform = letterboxTransform(metadata.width, metadata.height);
  const relative = `data/visa/candle/canonical/${entry.id}.png`;
  const output = await sharp(original).removeAlpha().toColourspace('srgb')
    .resize(transform.resizedWidth, transform.resizedHeight, { fit: 'fill', kernel: 'lanczos3' })
    .extend({ left: transform.padLeft, top: transform.padTop,
      right: SIZE - transform.padLeft - transform.resizedWidth, bottom: SIZE - transform.padTop - transform.resizedHeight,
      background: { r: PADDING[0], g: PADDING[1], b: PADDING[2] } })
    .png({ compressionLevel: 9 }).toBuffer();
  await writeFile(path.join(root, relative), output);
  canonical.push({ id: entry.id, path: relative, transform, sha256: hash(output), validPatches: validPatchMask(transform) });
  if ((i + 1) % 100 === 0) console.log(`Canonical images ${i + 1}/${split.entries.length}`);
}
await writeFile(path.join(dataRoot, 'canonical.json'), json({ schemaVersion: 1, entries: canonical }));
console.log('Canonical manifest ready.');
const canonicalMap = new Map(canonical.map(x => [x.id, x]));
const selected = selectBankPatches(fit.map(x => ({ id: x.id, valid: canonicalMap.get(x.id)!.validPatches })), BANK_CAP);
await writeFile(path.join(dataRoot, 'bank-selection.json'), json({ version, seed: 42, capacity: BANK_CAP,
  selection: [...selected].map(([id, patches]) => ({ id, patches })) }));
const banks = Object.fromEntries(methods.map(method => [method, new Float32Array(BANK_CAP * DIMENSIONS[method])])) as Record<Method, Float32Array>;
const model = await createModel(modelPath);
console.log('Pinned ONNX model loaded; spatial output verified.');
const fitStarted = performance.now();
let bankOffset = 0;
async function rgbFor(entry: Canonical) { return new Uint8Array(await sharp(path.join(root, entry.path)).removeAlpha().raw().toBuffer()); }
for (let i = 0; i < fit.length; i++) {
  const entry = canonicalMap.get(fit[i].id)!;
  const rgb = await rgbFor(entry);
  const cacheKey = hash(entry.sha256 + MODEL_SHA256 + kernelHash);
  const cachePath = path.join(dataRoot, 'feature-cache', `${entry.id}-${cacheKey.slice(0, 16)}.f32`);
  const cached = await exists(cachePath);
  let ai: Float32Array;
  if (cached && cached.length === PATCHES * DIMENSIONS.dinov2 * 4) {
    ai = new Float32Array(cached.buffer.slice(cached.byteOffset, cached.byteOffset + cached.byteLength));
  } else {
    ai = await model.extract(rgb);
    await writeFile(cachePath, asBytes(ai));
  }
  const features = { dinov2: ai, classical: classicalFeatures(rgb, entry.transform), pixel: pixelFeatures(rgb) };
  for (const patch of selected.get(entry.id)!) {
    for (const method of methods) {
      const dims = DIMENSIONS[method];
      banks[method].set(features[method].subarray(patch * dims, (patch + 1) * dims), bankOffset * dims);
    }
    bankOffset++;
  }
  if ((i + 1) % 40 === 0) console.log(`Normal reference features ${i + 1}/${fit.length}`);
  if (i === 4) console.log(`First five fit images: ${((performance.now() - fitStarted) / 1000).toFixed(2)} seconds.`);
}
if (bankOffset !== BANK_CAP) throw new Error('Reference bank did not fill its fixed budget.');
const stats = fitStandardization(banks.classical, DIMENSIONS.classical);
banks.classical = standardize(banks.classical, DIMENSIONS.classical, stats);
const metadata = {} as ArtifactManifest['methods'];
for (const method of methods) {
  const bytes = asBytes(banks[method]);
  await writeFile(path.join(artifactRoot, `${method}-bank.bin`), bytes);
  metadata[method] = { bankUrl: `artifacts/${method}-bank.bin`, bankSha256: hash(bytes), dimensions: DIMENSIONS[method], count: BANK_CAP, threshold: 0, heatmapScale: 1 };
}
const statsBytes = json(stats);
await writeFile(path.join(artifactRoot, 'classical-standardization.json'), statsBytes);
metadata.classical.standardizationUrl = 'artifacts/classical-standardization.json';
metadata.classical.standardizationSha256 = hash(statsBytes);
const predictions: Record<string, unknown>[] = [];
const calScores = Object.fromEntries(methods.map(m => [m, []])) as Record<Method, number[]>;
const calPatchScores = Object.fromEntries(methods.map(m => [m, []])) as Record<Method, number[]>;
let calibrationFrozenAt = '';
for (const [i, source] of [...calibration, ...test].entries()) {
  const entry = canonicalMap.get(source.id)!;
  const rgb = await rgbFor(entry);
  const outputs: Record<string, unknown> = { id: entry.id, validPatches: entry.validPatches };
  for (const method of methods) {
    const start = performance.now();
    const features = method === 'dinov2' ? await model.extract(rgb) : method === 'classical' ?
      standardize(classicalFeatures(rgb, entry.transform), DIMENSIONS.classical, stats) : pixelFeatures(rgb);
    const scored = scoreFeatures(features, banks[method], DIMENSIONS[method], entry.validPatches);
    const elapsed = performance.now() - start;
    const prefix = method === 'dinov2' ? 'ai' : method === 'classical' ? 'baseline' : 'pixel';
    outputs[`${prefix}Score`] = scored.score; outputs[`${prefix}Heatmap`] = scored.heatmap; outputs[`${prefix}Ms`] = elapsed;
    if (source.split === 'calibration') {
      calScores[method].push(scored.score);
      calPatchScores[method].push(...scored.heatmap.filter((_, index) => entry.validPatches[index]));
    }
  }
  predictions.push(outputs);
  if (i === calibration.length - 1) {
    for (const method of methods) {
      metadata[method].threshold = calibrationThreshold(calScores[method], 0.05);
      metadata[method].heatmapScale = Math.max(1e-9, quantile(calPatchScores[method], 0.995));
    }
    calibrationFrozenAt = new Date().toISOString();
    await writeFile(path.join(dataRoot, 'calibration-lock.json'), json({ version, splitVersion, frozenAt: calibrationFrozenAt,
      thresholdRule: 'ceil((n+1)*0.95)-1; strictly greater than threshold', calibrationIds: calibration.map(x => x.id),
      calibrationScoreSha256: hash(json(calScores)), methods: metadata }));
    console.log('Normal calibration thresholds and heatmap scales frozen before test inference.');
  }
  if ((i + 1) % 20 === 0) console.log(`Calibration/test predictions ${i + 1}/380`);
}
if (!calibrationFrozenAt) throw new Error('Calibration must be frozen before publishing test predictions.');
const manifest: ArtifactManifest = { schemaVersion: 1, version, splitVersion,
  model: { id: MODEL.id, revision: MODEL.revision, dtype: 'q8', localPath: `models/${MODEL.file}`, sha256: MODEL_SHA256 },
  preprocessing: { size: 224, patchSize: 14, padding: [...PADDING], mean: [...MEAN], std: [...STD] },
  methods: metadata, createdAt: new Date().toISOString() };
validateManifest(manifest);
await writeFile(path.join(dataRoot, 'predictions.json'), json({ modelVersion: version, artifactVersion: version, splitVersion,
  environment: `ONNX Runtime Web 1.22.0 WASM; threads=1; ${os.platform()} ${os.arch()}; ${os.cpus()[0]?.model}; Node ${process.version}`,
  timingDefinition: 'Warm model feature extraction plus nearest-neighbour scoring; excludes image decoding and file/network loading. Measured in Node with the same single-thread WASM backend as the browser; timings still vary by browser/device.',
  images: predictions }));
await mkdir(path.join(publicRoot, 'ort'), { recursive: true });
for (const file of await readdir(path.join(root, 'node_modules/onnxruntime-web/dist'))) {
  if (['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'].includes(file)) {
    await copyFile(path.join(root, 'node_modules/onnxruntime-web/dist', file), path.join(publicRoot, 'ort', file));
  }
}
await writeFile(path.join(artifactRoot, 'methodology.json'), json({ version, kernelHash, runtime: RUNTIME_VERSION, modelSha256: MODEL_SHA256, calibrationFrozenAt,
  input: 'Sharp Lanczos3 aspect-ratio preserving 224px letterbox; fixed RGB(127,127,127) padding; canonical PNGs shared verbatim with browser. ImageNet channel normalisation.',
  patchEligibility: 'Only 14×14 patches entirely contained within the resized image; all other patch scores displayed as zero and excluded from scoring/evaluation.',
  dinov2: 'Frozen q8 DINOv2-small last_hidden_state; discard CLS; per-patch L2 normalisation; no model training. Both offline Node and browser use ONNX Runtime Web1.22.0 WASM, one thread.',
  classical: 'Lab mean/std and L-channel central-difference gradient magnitude mean/std, over 14/28/42px clipped windows (24 features); fit-bank per-feature z-score with std floor0.001.',
  pixel: 'Raw 14×14×3 RGB patch scaled to [0,1] (588 features); no feature learning.',
  bank: 'Same1024fitpatchlocations per method: at least one per fit image, remaining patches by seeded deterministic FNV-1a hash rank. No coreset optimisation.',
  scoring: 'Exact Euclidean nearest-reference distance per eligible patch; image score mean of five highest patch distances.',
  calibration: '180normalonly images; finite-sample split-conformal95% upper quantile ceil((n+1)*0.95)-1; strict score>threshold flag.',
  heatmapScale: 'Fixed per method at calibration-normal eligible patch99.5th percentile; test data never sets thresholds or display scale.',
  limits: ['Patch embeddings can attend to padding even though padded patches are excluded.', 'Fixed224resolution may discard small defects.', 'Reported WASM timings were collected in Node; real browser/device latencies vary.', 'Native CPU q8 execution was rejected after fit-only parity checks showed material feature differences; published results use WASM throughout.'] }));
await writeFile(path.join(artifactRoot, 'manifest.json'), json(manifest));
await model.release();
console.log(`Artifacts complete: ${version}`);
