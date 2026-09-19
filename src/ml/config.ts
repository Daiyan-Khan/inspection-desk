export const MODEL = {
  id: 'Xenova/dinov2-small',
  revision: 'c2bb04a51fab207c420665f1946016107bffc701',
  file: 'dinov2-small-q8.onnx',
  url: 'https://huggingface.co/Xenova/dinov2-small/resolve/c2bb04a51fab207c420665f1946016107bffc701/onnx/model_quantized.onnx',
  dimensions: 384,
} as const;
export const SIZE = 224;
export const PATCH_SIZE = 14;
export const GRID = SIZE / PATCH_SIZE;
export const PATCHES = GRID * GRID;
export const BANK_CAP = 1024;
export const TOP_PATCHES = 5;
export const MEAN = [0.485, 0.456, 0.406] as const;
export const STD = [0.229, 0.224, 0.225] as const;
export const PADDING = [127, 127, 127] as const;
export const CLASSICAL_DIMENSIONS = 24;
export const PIXEL_DIMENSIONS = PATCH_SIZE * PATCH_SIZE * 3;
