import { cp, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
const root = process.cwd(), destination = path.join(root, '.publish', 'space');
for (const file of ['dist/index.html', 'dist/models/dinov2-small-q8.onnx', 'dist/artifacts/manifest.json', 'dist/data/sample-manifest.json', 'dist/data/evaluation.json']) {
  await stat(path.join(root, file)).catch(() => { throw new Error(`Missing ${file}. Finish artifact generation, evaluation and npm run build first.`); });
}
const manifest = JSON.parse(await readFile(path.join(root, 'dist/artifacts/manifest.json'), 'utf8'));
const evaluation = JSON.parse(await readFile(path.join(root, 'dist/data/evaluation.json'), 'utf8'));
if (evaluation.artifactVersion !== manifest.version || evaluation.splitVersion !== manifest.splitVersion) throw new Error('Evaluation and artifact versions differ.');
await mkdir(destination, { recursive: true });
await cp(path.join(root, 'dist'), destination, { recursive: true });
for (const file of ['LICENSE', 'THIRD_PARTY_NOTICES.md']) await cp(path.join(root, file), path.join(destination, file));
await writeFile(path.join(destination, 'README.md'), `---\ntitle: Inspection Desk\nemoji: 🔎\ncolorFrom: green\ncolorTo: gray\nsdk: static\napp_file: index.html\npinned: false\nlicense: mit\n---\n\n# Inspection Desk\n\nA browser-based inspection assistant with local review history. Live CPU/WASM inference, real VisA candle images, and an honestly measured comparison of frozen DINOv2 features with classical vision.\n\nExperiment: ${manifest.version}\n\nApplication code: MIT. VisA images: CC BY 4.0. DINOv2 weights: Apache 2.0. See THIRD_PARTY_NOTICES.md. No manufacturer deployment or business impact is claimed.\n`);
console.log(`Verified static upload prepared at ${destination}`);
