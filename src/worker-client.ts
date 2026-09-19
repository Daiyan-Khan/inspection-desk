import type { ArtifactManifest, InspectionResult, Method, ModelProgress, Sample, WorkerResponse } from './types';
export class InspectionWorkerClient {
  private worker: Worker | null = null;
  private cancelPending: (() => void) | null = null;
  inspect(sample: Sample, method: Method, manifest: ArtifactManifest, progress: (value: ModelProgress) => void): Promise<InspectionResult> {
    if (this.cancelPending) return Promise.reject(new Error('Only one image can be processed at a time.'));
    this.worker ??= new Worker(new URL('./inference.worker.ts', import.meta.url), { type: 'module' });
    const worker = this.worker;
    const requestId = crypto.randomUUID();
    return new Promise((resolve, reject) => {
      const finish = () => { clearTimeout(timeout); worker.removeEventListener('message', onMessage); worker.removeEventListener('error', onError); this.cancelPending = null; };
      const onError = () => { finish(); worker.terminate(); this.worker = null; reject(new Error('The inference worker stopped. Retry, or continue with manual review.')); };
      const onMessage = (event: MessageEvent<WorkerResponse>) => {
        const message = event.data;
        if (message.requestId !== requestId) return;
        if (message.type === 'progress') progress(message.progress);
        else if (message.type === 'result') { finish(); resolve(message.result); }
        else { finish(); reject(new Error(message.error)); }
      };
      const timeout = setTimeout(() => { finish(); worker.terminate(); this.worker = null; reject(new Error('Processing timed out after three minutes. Retry when your connection is available.')); }, 180_000);
      this.cancelPending = () => { finish(); worker.terminate(); this.worker = null; reject(new DOMException('Processing cancelled', 'AbortError')); };
      worker.addEventListener('message', onMessage); worker.addEventListener('error', onError);
      worker.postMessage({ type: 'inspect', requestId, sample, method, manifest });
    });
  }
  cancel() { if (this.cancelPending) this.cancelPending(); else { this.worker?.terminate(); this.worker = null; } }
}
