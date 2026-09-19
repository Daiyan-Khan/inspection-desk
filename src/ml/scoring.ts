import { PATCHES, TOP_PATCHES } from './config';

export interface ScoredImage { score: number; heatmap: number[] }

/** Exact nearest-neighbour L2 distance. Image score is mean of the five highest eligible patch distances. */
export function scoreFeatures(features: Float32Array, bank: Float32Array, dimensions: number, valid: boolean[]): ScoredImage {
  if (features.length !== PATCHES * dimensions || bank.length % dimensions !== 0 || bank.length === 0 ||
      valid.length !== PATCHES || valid.filter(Boolean).length < TOP_PATCHES) throw new Error('Invalid feature bank or patch mask.');
  const count = bank.length / dimensions;
  const heatmap = Array<number>(PATCHES).fill(0);
  const eligible: number[] = [];
  for (let p = 0; p < PATCHES; p++) {
    if (!valid[p]) continue;
    const offset = p * dimensions;
    let nearest = Infinity;
    for (let n = 0; n < count; n++) {
      const bankOffset = n * dimensions;
      let distance = 0;
      for (let d = 0; d < dimensions; d++) {
        const delta = features[offset + d] - bank[bankOffset + d];
        distance += delta * delta;
      }
      if (distance < nearest) nearest = distance;
    }
    const distance = Math.sqrt(nearest);
    if (!Number.isFinite(distance)) throw new Error('Non-finite anomaly score.');
    heatmap[p] = distance; eligible.push(distance);
  }
  eligible.sort((a, b) => b - a);
  return { score: eligible.slice(0, TOP_PATCHES).reduce((sum, value) => sum + value, 0) / TOP_PATCHES, heatmap };
}

/** Split-conformal upper quantile; flag iff score > threshold. Labels are never used. */
export function calibrationThreshold(scores: number[], alpha = 0.05): number {
  if (scores.length === 0 || !scores.every(Number.isFinite) || alpha <= 0 || alpha >= 1) throw new Error('Invalid calibration scores.');
  const sorted = [...scores].sort((a, b) => a - b);
  const index = Math.ceil((sorted.length + 1) * (1 - alpha)) - 1;
  if (index >= sorted.length) throw new Error('Insufficient normal calibration images for target false alarm rate.');
  return sorted[index];
}

export function quantile(values: number[], probability: number): number {
  if (values.length === 0 || !values.every(Number.isFinite) || probability < 0 || probability > 1) throw new Error('Invalid quantile inputs.');
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * probability))];
}

export function stableHash(value: string): number {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) { hash ^= value.charCodeAt(i); hash = Math.imul(hash, 16777619); }
  return hash >>> 0;
}

/** One patch per fit image, then fill the remaining capacity by deterministic hash rank. */
export function selectBankPatches(images: { id: string; valid: boolean[] }[], capacity = 1024): Map<string, number[]> {
  if (images.length > capacity) throw new Error('Reference capacity must cover every fit image.');
  const selected = new Map<string, number[]>();
  const remaining: { id: string; patch: number; rank: number }[] = [];
  for (const image of [...images].sort((a, b) => a.id.localeCompare(b.id))) {
    const candidates = image.valid.flatMap((valid, patch) => valid ? [{ id: image.id, patch, rank: stableHash(`42:${image.id}:${patch}`) }] : []);
    candidates.sort((a, b) => a.rank - b.rank || a.patch - b.patch);
    if (!candidates.length) throw new Error(`No eligible patches for ${image.id}.`);
    selected.set(image.id, [candidates[0].patch]); remaining.push(...candidates.slice(1));
  }
  remaining.sort((a, b) => a.rank - b.rank || a.id.localeCompare(b.id) || a.patch - b.patch);
  for (const item of remaining.slice(0, capacity - images.length)) selected.get(item.id)!.push(item.patch);
  return selected;
}
