import type { Transform } from '../types';
import { GRID, MEAN, PATCHES, PATCH_SIZE, SIZE, STD } from './config';

export function assertTransform(t: Transform): void {
  const values = [t.originalWidth, t.originalHeight, t.resizedWidth, t.resizedHeight, t.padLeft, t.padTop, t.inputSize];
  if (!values.every(Number.isInteger) || t.inputSize !== SIZE || t.originalWidth < 1 || t.originalHeight < 1 ||
      t.resizedWidth < PATCH_SIZE || t.resizedHeight < PATCH_SIZE || t.padLeft < 0 || t.padTop < 0 ||
      t.padLeft + t.resizedWidth > SIZE || t.padTop + t.resizedHeight > SIZE) {
    throw new Error('Invalid image transform; expected a 224px letterbox.');
  }
}

export function letterboxTransform(width: number, height: number): Transform {
  const scale = Math.min(SIZE / width, SIZE / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  const t = { originalWidth: width, originalHeight: height, resizedWidth, resizedHeight,
    padLeft: Math.floor((SIZE - resizedWidth) / 2), padTop: Math.floor((SIZE - resizedHeight) / 2), inputSize: SIZE };
  assertTransform(t);
  return t;
}

/** A patch is eligible only when all 14×14 pixels are inside the real image. */
export function validPatchMask(t: Transform): boolean[] {
  assertTransform(t);
  return Array.from({ length: PATCHES }, (_, i) => {
    const x = (i % GRID) * PATCH_SIZE;
    const y = Math.floor(i / GRID) * PATCH_SIZE;
    return x >= t.padLeft && y >= t.padTop && x + PATCH_SIZE <= t.padLeft + t.resizedWidth &&
      y + PATCH_SIZE <= t.padTop + t.resizedHeight;
  });
}

export function assertRgb(rgb: Uint8Array): void {
  if (rgb.length !== SIZE * SIZE * 3) throw new Error('Expected an exact 224×224 RGB image.');
}

/** Canonical RGB bytes -> model NCHW tensor; identical code runs offline and in the worker. */
export function modelInput(rgb: Uint8Array): Float32Array {
  assertRgb(rgb);
  const plane = SIZE * SIZE;
  const output = new Float32Array(rgb.length);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) output[c * plane + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
  }
  return output;
}

/** DINO's first token is CLS, not a spatial patch. Return L2-normalised patch features. */
export function patchFeatures(data: ArrayLike<number>, dims: readonly number[]): Float32Array {
  if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== PATCHES + 1 || dims[2] !== 384 || data.length !== 257 * 384) {
    throw new Error(`Unexpected DINO output shape: ${dims.join('×')}.`);
  }
  const output = new Float32Array(PATCHES * 384);
  for (let p = 0; p < PATCHES; p++) {
    let norm = 0;
    for (let d = 0; d < 384; d++) {
      const value = data[(p + 1) * 384 + d];
      if (!Number.isFinite(value)) throw new Error('Model returned non-finite features.');
      output[p * 384 + d] = value;
      norm += value * value;
    }
    if (norm < 1e-12) throw new Error('Model returned an empty patch feature.');
    norm = Math.sqrt(norm);
    for (let d = 0; d < 384; d++) output[p * 384 + d] /= norm;
  }
  return output;
}
