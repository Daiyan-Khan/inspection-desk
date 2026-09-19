import type { ArtifactManifest, Method } from '../types';
import { BANK_CAP, CLASSICAL_DIMENSIONS, MEAN, MODEL, PADDING, PIXEL_DIMENSIONS, STD } from './config';

export const DIMENSIONS: Record<Method, number> = { dinov2: MODEL.dimensions, classical: CLASSICAL_DIMENSIONS, pixel: PIXEL_DIMENSIONS };
export const MODEL_SHA256 = '3afdc8bc63b50558d6e5770f5b799bb82455c2311183a2de43803f343a29d917';

export function validateManifest(manifest: ArtifactManifest): void {
  if (manifest.schemaVersion !== 1 || !/^candle-q8-wasm-v1-[a-f0-9]{12}$/.test(manifest.version) || !manifest.splitVersion ||
      manifest.model.id !== MODEL.id || manifest.model.revision !== MODEL.revision || manifest.model.dtype !== 'q8' ||
      manifest.model.sha256 !== MODEL_SHA256 || !manifest.model.localPath || manifest.preprocessing.size !== 224 ||
      manifest.preprocessing.patchSize !== 14 || JSON.stringify(manifest.preprocessing.mean) !== JSON.stringify(MEAN) ||
      JSON.stringify(manifest.preprocessing.std) !== JSON.stringify(STD) || JSON.stringify(manifest.preprocessing.padding) !== JSON.stringify(PADDING)) {
    throw new Error('Artifact manifest is incompatible with this model/preprocessing version.');
  }
  for (const method of ['dinov2', 'classical', 'pixel'] as const) {
    const artifact = manifest.methods[method];
    if (!artifact || artifact.dimensions !== DIMENSIONS[method] || artifact.count !== BANK_CAP ||
        !Number.isFinite(artifact.threshold) || artifact.threshold < 0 || !Number.isFinite(artifact.heatmapScale) || artifact.heatmapScale <= 0 ||
        !/^[a-f0-9]{64}$/.test(artifact.bankSha256) || !artifact.bankUrl) throw new Error(`Invalid ${method} artifact metadata.`);
    if (method === 'classical' && (!artifact.standardizationUrl || !/^[a-f0-9]{64}$/.test(artifact.standardizationSha256 ?? ''))) {
      throw new Error('Missing classical feature standardization.');
    }
  }
}

export async function sha256(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

export async function checkedFetch(url: string, expected: string): Promise<ArrayBuffer> {
  // Content-addressed entries survive reloads. Storage denial/quota never prevents inspection.
  let cache: Cache | undefined;
  const key = new URL(url);
  key.searchParams.set('artifact-sha256', expected);
  try {
    if (typeof caches !== 'undefined') {
      cache = await caches.open('inspection-desk-artifacts-v1');
      const cached = await cache.match(key.href);
      if (cached) {
        const bytes = await cached.arrayBuffer();
        if (await sha256(bytes) === expected) return bytes;
        await cache.delete(key.href);
      }
    }
  } catch { cache = undefined; }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load artifact (${response.status}).`);
  const bytes = await response.arrayBuffer();
  if (await sha256(bytes) !== expected) throw new Error('Artifact integrity check failed. Please reload the demo.');
  try { await cache?.put(key.href, new Response(bytes)); } catch { /* Browser storage may be unavailable. */ }
  return bytes;
}

export function decodeBank(bytes: ArrayBuffer, dimensions: number, count: number): Float32Array {
  if (bytes.byteLength !== dimensions * count * 4) throw new Error('Reference-bank size does not match manifest.');
  const bank = new Float32Array(bytes);
  if (!bank.every(Number.isFinite)) throw new Error('Reference bank contains invalid values.');
  return bank;
}
