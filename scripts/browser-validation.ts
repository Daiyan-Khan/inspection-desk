import { InspectionWorkerClient } from '../src/worker-client';
import type { ArtifactManifest, Method, SampleManifest } from '../src/types';
const button = document.querySelector<HTMLButtonElement>('#start')!;
const status = document.querySelector('#status')!;
const output = document.querySelector('#results')!;
button.onclick = async () => {
  button.disabled = true;
  const worker = new InspectionWorkerClient();
  try {
    const [manifest, samples, fixtures] = await Promise.all([
      fetch('./artifacts/manifest.json').then(r => r.json()) as Promise<ArtifactManifest>,
      fetch('./data/sample-manifest.json').then(r => r.json()) as Promise<SampleManifest>,
      fetch('./data/parity-fixtures.json').then(r => r.json()) as Promise<{ artifactVersion: string; fixtures: { sampleId: string; method: Method; score: number; heatmap: number[] }[] }>,
    ]);
    if (fixtures.artifactVersion !== manifest.version) throw new Error('Fixture artifact version mismatch.');
    const rows = [];
    for (const fixture of fixtures.fixtures) {
      const sample = samples.samples.find(s => s.id === fixture.sampleId);
      if (!sample) throw new Error('Fixture sample missing.');
      const start = performance.now();
      const result = await worker.inspect(sample, fixture.method, manifest, progress => { status.textContent = `${fixture.method} · ${sample.name} · ${progress.message}`; });
      const endToEndMs = performance.now() - start;
      const scoreError = Math.abs(result.score - fixture.score);
      const maxPatchError = Math.max(...result.heatmap.map((value, i) => Math.abs(value - fixture.heatmap[i])));
      rows.push({ sampleId: sample.id, method: fixture.method, score: result.score, scoreError, maxPatchError, warmProcessingMs: result.durationMs, endToEndMs, passed: scoreError <= 0.002 && maxPatchError <= 0.005 });
      output.textContent = JSON.stringify({ status: 'running', rows }, null, 2);
    }
    const report = { schemaVersion: 1, status: rows.every(r => r.passed) ? 'passed' : 'failed', artifactVersion: manifest.version,
      measuredAt: new Date().toISOString(), userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency,
      tolerance: { absoluteImageScore: 0.002, absolutePatchDistance: 0.005 },
      memory: (performance as Performance & { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory ?? null,
      notes: ['Timing is measured on this browser/device, not a guarantee for other devices.', 'Offline and browser both use the pinned ORT WASM provider.', 'End-to-end includes resource loading on the first request; warmProcessing excludes loading/decoding.', 'JS heap excludes some WASM/native allocations and is not peak process memory.'], rows };
    output.textContent = JSON.stringify(report, null, 2);
    status.textContent = report.status === 'passed' ? 'All real browser parity checks passed.' : 'Parity differences found; inspect the report.';
  } catch (error) { status.textContent = `Failed: ${error instanceof Error ? error.message : error}`; }
  finally { worker.cancel(); button.disabled = false; }
};
