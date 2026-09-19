import * as ort from 'onnxruntime-web/wasm';
import type { ArtifactManifest, Method, WorkerRequest, WorkerResponse } from './types';
import { checkedFetch, decodeBank, validateManifest } from './ml/artifacts';
import { classicalFeatures, pixelFeatures, standardize } from './ml/baseline';
import type { Standardization } from './ml/baseline';
import { GRID, SIZE } from './ml/config';
import { modelInput, patchFeatures, validPatchMask } from './ml/preprocessing';
import { scoreFeatures } from './ml/scoring';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
// These version-pinned runtime files are copied into public/ort by the artifact builder.
ort.env.wasm.wasmPaths = new URL('../ort/', self.location.href).href;
const banks = new Map<string, Float32Array>();
const statistics = new Map<string, Standardization>();
let sessionPromise: Promise<ort.InferenceSession> | undefined;
let loadedModelKey = '';

function reply(message: WorkerResponse): void { self.postMessage(message); }
function resource(path: string): string {
  // The worker lives in assets/ in production; document-relative URLs arrive from the main thread.
  return new URL(path, new URL('../', self.location.href)).href;
}

async function loadRgb(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error('The canonical sample image could not be loaded.');
  const image = await createImageBitmap(await response.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  try {
    if (image.width !== SIZE || image.height !== SIZE) throw new Error('Sample is not a canonical 224×224 image.');
    const canvas = new OffscreenCanvas(SIZE, SIZE);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('This browser does not support image decoding in a worker.');
    context.drawImage(image, 0, 0);
    const rgba = context.getImageData(0, 0, SIZE, SIZE).data;
    const rgb = new Uint8Array(SIZE * SIZE * 3);
    for (let i = 0; i < SIZE * SIZE; i++) for (let c = 0; c < 3; c++) rgb[3 * i + c] = rgba[4 * i + c];
    return rgb;
  } finally { image.close(); }
}

async function getBank(manifest: ArtifactManifest, method: Method): Promise<Float32Array> {
  const artifact = manifest.methods[method];
  const key = artifact.bankSha256;
  if (!banks.has(key)) banks.set(key, decodeBank(await checkedFetch(resource(artifact.bankUrl), key), artifact.dimensions, artifact.count));
  const statisticsKey = artifact.standardizationSha256;
  if (method === 'classical' && !statistics.has(statisticsKey!)) {
    const bytes = await checkedFetch(resource(artifact.standardizationUrl!), artifact.standardizationSha256!);
    statistics.set(statisticsKey!, JSON.parse(new TextDecoder().decode(bytes)) as Standardization);
  }
  return banks.get(key)!;
}

async function getSession(manifest: ArtifactManifest): Promise<ort.InferenceSession> {
  const model = manifest.model;
  if (!sessionPromise || loadedModelKey !== model.sha256) {
    loadedModelKey = model.sha256;
    sessionPromise = checkedFetch(resource(model.localPath!), model.sha256).then(bytes =>
      ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })).catch(error => {
        sessionPromise = undefined;
        throw error;
      });
  }
  return sessionPromise;
}

async function inspect(request: WorkerRequest): Promise<void> {
  const { requestId, manifest, sample, method } = request;
  try {
    validateManifest(manifest);
    if (!['dinov2', 'classical', 'pixel'].includes(method)) throw new Error('Unsupported inspection method.');
    const valid = validPatchMask(sample.transform);
    reply({ type: 'progress', requestId, progress: { phase: 'loading', message: method === 'dinov2' ? 'Loading and checking the vision model…' : 'Loading the normal reference bank…' } });
    const bank = await getBank(manifest, method);
    const rgb = await loadRgb(resource(sample.inputUrl));
    const session = method === 'dinov2' ? await getSession(manifest) : undefined;
    reply({ type: 'progress', requestId, progress: { phase: 'processing', message: 'Comparing image patches with normal references…' } });
    const start = performance.now();
    let features: Float32Array;
    if (method === 'dinov2') {
      const output = await session!.run({ pixel_values: new ort.Tensor('float32', modelInput(rgb), [1, 3, SIZE, SIZE]) });
      const tensor = output.last_hidden_state;
      if (!tensor) throw new Error('Model did not return spatial patch features.');
      features = patchFeatures(tensor.data as Float32Array, tensor.dims);
    } else if (method === 'classical') {
      features = standardize(classicalFeatures(rgb, sample.transform), manifest.methods.classical.dimensions, statistics.get(manifest.methods.classical.standardizationSha256!)!);
    } else features = pixelFeatures(rgb);
    const scored = scoreFeatures(features, bank, manifest.methods[method].dimensions, valid);
    const artifact = manifest.methods[method];
    reply({ type: 'result', requestId, result: { schemaVersion: 1, sampleId: sample.id, method, artifactVersion: manifest.version,
      score: scored.score, threshold: artifact.threshold, flagged: scored.score > artifact.threshold, heatmap: scored.heatmap,
      gridWidth: GRID, gridHeight: GRID, heatmapScale: artifact.heatmapScale, transform: sample.transform,
      durationMs: performance.now() - start, completedAt: new Date().toISOString() } });
  } catch (error) {
    reply({ type: 'error', requestId, error: error instanceof Error ? error.message : 'Inspection failed.' });
  }
}

// Serialise jobs; a second message cannot overlap an ONNX session run or exhaust memory.
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  if (event.data?.type === 'inspect') queue = queue.then(() => inspect(event.data));
};
