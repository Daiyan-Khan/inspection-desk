import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, rename, rm } from 'node:fs/promises';
import path from 'node:path';

// Restore the published model without rerunning the frozen experiment.
const manifest = JSON.parse(await readFile('public/artifacts/manifest.json', 'utf8'));
const destination = path.resolve('public/models/dinov2-small-q8.onnx');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const existing = await readFile(destination).catch(error => {
  if (error.code === 'ENOENT') return null;
  throw error;
});
if (existing && digest(existing) === manifest.model.sha256) {
  console.log('The pinned model is already present and verified.');
} else {
  const revision = '43cbb952fea51beee295589ff0d5e94bf8d971e4';
  const url = `https://huggingface.co/spaces/Daibolical/inspection-desk/resolve/${revision}/models/dinov2-small-q8.onnx`;
  console.log('Downloading the published model (24.5 MB)…');
  const response = await fetch(url, { signal: AbortSignal.timeout(180_000) });
  if (!response.ok) throw new Error(`Model download failed: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (digest(bytes) !== manifest.model.sha256) throw new Error('Model checksum mismatch; no file was replaced.');
  await mkdir(path.dirname(destination), { recursive: true });
  const temporary = `${destination}.${process.pid}.download`;
  try {
    await writeFile(temporary, bytes);
    await rename(temporary, destination);
  } finally {
    await rm(temporary, { force: true });
  }
  console.log('Model downloaded and SHA-256 verified. Run npm run dev.');
}
