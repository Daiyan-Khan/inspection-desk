import { readFile, writeFile } from 'node:fs/promises';
import sharp from 'sharp';
import * as wasm from 'onnxruntime-web/wasm';
import * as cpu from 'onnxruntime-node';
import { modelInput, patchFeatures } from '../src/ml/preprocessing.ts';
import { scoreFeatures } from '../src/ml/scoring.ts';

// This numerical smoke check uses only fit images; it cannot influence test selection.
wasm.env.wasm.numThreads = 1;
const split = JSON.parse(await readFile('data/visa/candle/split.json', 'utf8'));
const canonical = JSON.parse(await readFile('data/visa/candle/canonical.json', 'utf8'));
const fitIds = split.entries.filter((entry: { split: string }) => entry.split === 'fit').slice(0, 3).map((entry: { id: string }) => entry.id);
const bytes = await readFile('public/models/dinov2-small-q8.onnx');
const native = await cpu.InferenceSession.create(bytes, { executionProviders: ['cpu'], intraOpNumThreads: 1, interOpNumThreads: 1, graphOptimizationLevel: 'all' });
const web = await wasm.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
const bankBytes = await readFile('public/artifacts/dinov2-bank.bin');
const bank = new Float32Array(bankBytes.buffer.slice(bankBytes.byteOffset, bankBytes.byteOffset + bankBytes.byteLength));
const results = [];
for (const id of fitIds) {
  const entry = canonical.entries.find((item: { id: string }) => item.id === id);
  const rgb = new Uint8Array(await sharp(entry.path).removeAlpha().raw().toBuffer());
  const cpuOutput = await native.run({ pixel_values: new cpu.Tensor('float32', modelInput(rgb), [1, 3, 224, 224]) });
  const cpuFeatures = patchFeatures(cpuOutput.last_hidden_state.data as Float32Array, cpuOutput.last_hidden_state.dims);
  const start = performance.now();
  const output = await web.run({ pixel_values: new wasm.Tensor('float32', modelInput(rgb), [1, 3, 224, 224]) });
  const webFeatures = patchFeatures(output.last_hidden_state.data as Float32Array, output.last_hidden_state.dims);
  const wasmFeatureMs = performance.now() - start;
  const cpuScored = scoreFeatures(cpuFeatures, bank, 384, entry.validPatches);
  const wasmScored = scoreFeatures(webFeatures, bank, 384, entry.validPatches);
  const differences = cpuFeatures.map((value, i) => Math.abs(value - webFeatures[i]));
  results.push({ id, wasmFeatureMs, maxFeatureDifference: Math.max(...differences),
    nativeScore: cpuScored.score, wasmScore: wasmScored.score, scoreDifference: Math.abs(cpuScored.score - wasmScored.score),
    maxHeatmapDifference: Math.max(...cpuScored.heatmap.map((value, i) => Math.abs(value - wasmScored.heatmap[i]))) });
}
await writeFile('data/visa/candle/runtime-parity.json', JSON.stringify({ runtime: 'Node native CPU vs Node WASM, both ORT1.22.0 single-thread; fit-images only; not a browser performance measurement', images: results }, null, 2) + '\n');
console.log(JSON.stringify(results, null, 2));
await native.release(); await web.release();
