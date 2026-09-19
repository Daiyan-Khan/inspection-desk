import type { Transform } from '../types';
import { CLASSICAL_DIMENSIONS, GRID, PATCHES, PATCH_SIZE, PIXEL_DIMENSIONS, SIZE } from './config';
import { assertRgb, assertTransform } from './preprocessing';

export interface Standardization { mean: number[]; std: number[] }

function linear(value: number): number {
  value /= 255;
  return value <= 0.04045 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
}
function labCurve(value: number): number {
  return value > 216 / 24389 ? Math.cbrt(value) : (24389 / 27 * value + 16) / 116;
}

/** Multiscale Lab moments and luminance gradient moments, no learned parameters. */
export function classicalFeatures(rgb: Uint8Array, t: Transform): Float32Array {
  assertRgb(rgb); assertTransform(t);
  const plane = SIZE * SIZE;
  const lab = new Float32Array(plane * 3);
  const luminance = new Float32Array(plane);
  const gradient = new Float32Array(plane);
  for (let i = 0; i < plane; i++) {
    const r = linear(rgb[3 * i]), g = linear(rgb[3 * i + 1]), b = linear(rgb[3 * i + 2]);
    const x = labCurve((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047);
    const y = labCurve(0.2126729 * r + 0.7151522 * g + 0.072175 * b);
    const z = labCurve((0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883);
    lab[i * 3] = (116 * y - 16) / 100;
    lab[i * 3 + 1] = 500 * (x - y) / 128;
    lab[i * 3 + 2] = 200 * (y - z) / 128;
    luminance[i] = lab[i * 3];
  }
  const xEnd = t.padLeft + t.resizedWidth - 1;
  const yEnd = t.padTop + t.resizedHeight - 1;
  for (let y = t.padTop; y <= yEnd; y++) {
    for (let x = t.padLeft; x <= xEnd; x++) {
      const dx = (luminance[y * SIZE + Math.min(x + 1, xEnd)] - luminance[y * SIZE + Math.max(x - 1, t.padLeft)]) / 2;
      const dy = (luminance[Math.min(y + 1, yEnd) * SIZE + x] - luminance[Math.max(y - 1, t.padTop) * SIZE + x]) / 2;
      gradient[y * SIZE + x] = Math.hypot(dx, dy);
    }
  }
  const output = new Float32Array(PATCHES * CLASSICAL_DIMENSIONS);
  for (let p = 0; p < PATCHES; p++) {
    const cx = (p % GRID) * PATCH_SIZE + PATCH_SIZE / 2;
    const cy = Math.floor(p / GRID) * PATCH_SIZE + PATCH_SIZE / 2;
    for (let scale = 0; scale < 3; scale++) {
      const half = [7, 14, 21][scale];
      const x0 = Math.max(t.padLeft, cx - half), x1 = Math.min(xEnd + 1, cx + half);
      const y0 = Math.max(t.padTop, cy - half), y1 = Math.min(yEnd + 1, cy + half);
      const sum = [0, 0, 0, 0], sq = [0, 0, 0, 0];
      let count = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
        const i = y * SIZE + x;
        for (let c = 0; c < 4; c++) {
          const value = c === 3 ? gradient[i] : lab[3 * i + c];
          sum[c] += value; sq[c] += value * value;
        }
        count++;
      }
      if (count === 0) continue;
      for (let c = 0; c < 4; c++) {
        output[p * CLASSICAL_DIMENSIONS + scale * 8 + c] = sum[c] / count;
        output[p * CLASSICAL_DIMENSIONS + scale * 8 + 4 + c] = Math.sqrt(Math.max(0, sq[c] / count - (sum[c] / count) ** 2));
      }
    }
  }
  return output;
}

/** Deliberately simple ablation: all raw RGB values of a patch, scaled to [0,1]. */
export function pixelFeatures(rgb: Uint8Array): Float32Array {
  assertRgb(rgb);
  const output = new Float32Array(PATCHES * PIXEL_DIMENSIONS);
  for (let p = 0; p < PATCHES; p++) {
    const x0 = (p % GRID) * PATCH_SIZE, y0 = Math.floor(p / GRID) * PATCH_SIZE;
    for (let y = y0; y < y0 + PATCH_SIZE; y++) for (let x = x0; x < x0 + PATCH_SIZE; x++) {
      const localPixel = (y - y0) * PATCH_SIZE + x - x0;
      for (let c = 0; c < 3; c++) output[p * PIXEL_DIMENSIONS + localPixel * 3 + c] = rgb[(y * SIZE + x) * 3 + c] / 255;
    }
  }
  return output;
}

export function fitStandardization(bank: Float32Array, dimensions: number): Standardization {
  const count = bank.length / dimensions;
  if (!Number.isInteger(count) || count < 2) throw new Error('Invalid standardization bank.');
  const mean = Array<number>(dimensions).fill(0), std = Array<number>(dimensions).fill(0);
  for (let i = 0; i < count; i++) for (let d = 0; d < dimensions; d++) mean[d] += bank[i * dimensions + d] / count;
  for (let i = 0; i < count; i++) for (let d = 0; d < dimensions; d++) std[d] += (bank[i * dimensions + d] - mean[d]) ** 2 / count;
  for (let d = 0; d < dimensions; d++) std[d] = Math.max(0.001, Math.sqrt(std[d]));
  return { mean, std };
}

export function standardize(features: Float32Array, dimensions: number, stats: Standardization): Float32Array {
  if (stats.mean.length !== dimensions || stats.std.length !== dimensions ||
      !stats.mean.every(Number.isFinite) || !stats.std.every(x => Number.isFinite(x) && x >= 0.001)) {
    throw new Error('Invalid feature standardization.');
  }
  const output = new Float32Array(features.length);
  for (let i = 0; i < features.length; i++) output[i] = (features[i] - stats.mean[i % dimensions]) / stats.std[i % dimensions];
  return output;
}
