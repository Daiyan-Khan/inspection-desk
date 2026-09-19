import { readFile } from 'node:fs/promises';
import * as ort from 'onnxruntime-web/wasm';
import { modelInput, patchFeatures } from '../src/ml/preprocessing.ts';
import { SIZE } from '../src/ml/config.ts';

export const RUNTIME_VERSION = 'onnxruntime-web@1.22.0/wasm/single-thread';

export async function createModel(path: string) {
  // Use the deployed browser backend offline as well: native CPU q8 kernels differ measurably.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const session = await ort.InferenceSession.create(await readFile(path), {
    executionProviders: ['wasm'], graphOptimizationLevel: 'all',
  });
  if (!session.inputNames.includes('pixel_values') || !session.outputNames.includes('last_hidden_state')) {
    throw new Error(`Unexpected ONNX interface: ${session.inputNames} / ${session.outputNames}`);
  }
  return {
    async extract(rgb: Uint8Array): Promise<Float32Array> {
      const output = await session.run({ pixel_values: new ort.Tensor('float32', modelInput(rgb), [1, 3, SIZE, SIZE]) });
      const tensor = output.last_hidden_state;
      return patchFeatures(tensor.data as Float32Array, tensor.dims);
    },
    release: () => session.release(),
  };
}
