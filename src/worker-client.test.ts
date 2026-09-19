import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { InspectionWorkerClient } from './worker-client';
import type { ArtifactManifest, Sample } from './types';

class MockWorker {
  static instances: MockWorker[] = [];
  handlers = new Map<string, Set<(event: any) => void>>();
  posted: any;
  terminated = false;
  constructor() { MockWorker.instances.push(this); }
  addEventListener(name: string, callback: (event: any) => void) { if (!this.handlers.has(name)) this.handlers.set(name, new Set()); this.handlers.get(name)!.add(callback); }
  removeEventListener(name: string, callback: (event: any) => void) { this.handlers.get(name)?.delete(callback); }
  postMessage(value: any) { this.posted = value; }
  terminate() { this.terminated = true; }
  emit(type: string, payload: any) { for (const callback of this.handlers.get(type) ?? []) callback(payload); }
}
const sample = { id: 'fixture' } as Sample;
const manifest = { version: 'fixture' } as ArtifactManifest;
beforeEach(() => { MockWorker.instances = []; vi.stubGlobal('Worker', MockWorker); });
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
describe('sequential inference worker lifecycle', () => {
  it('terminates and rejects in-flight processing on cancellation', async () => {
    const client = new InspectionWorkerClient();
    const promise = client.inspect(sample, 'dinov2', manifest, () => {});
    const assertion = expect(promise).rejects.toMatchObject({ name: 'AbortError' });
    client.cancel(); await assertion;
    expect(MockWorker.instances[0].terminated).toBe(true);
    const second = client.inspect(sample, 'classical', manifest, () => {});
    const again = expect(second).rejects.toMatchObject({ name: 'AbortError' });
    expect(MockWorker.instances).toHaveLength(2); client.cancel(); await again;
  });
  it('does not allow a second job to overwrite the active request', async () => {
    const client = new InspectionWorkerClient();
    const pending = client.inspect(sample, 'dinov2', manifest, () => {});
    await expect(client.inspect(sample, 'classical', manifest, () => {})).rejects.toThrow('one image');
    const assertion = expect(pending).rejects.toThrow('cancelled'); client.cancel(); await assertion;
  });
  it('ignores stale response IDs and reuses a healthy worker', async () => {
    const client = new InspectionWorkerClient(), progress = vi.fn();
    const pending = client.inspect(sample, 'dinov2', manifest, progress), worker = MockWorker.instances[0];
    worker.emit('message', { data: { type: 'progress', requestId: 'stale', progress: { phase: 'ready' } } });
    expect(progress).not.toHaveBeenCalled();
    const result = { sampleId: 'fixture', artifactVersion: 'fixture' };
    worker.emit('message', { data: { type: 'result', requestId: worker.posted.requestId, result } });
    await expect(pending).resolves.toEqual(result);
    const second = client.inspect(sample, 'classical', manifest, progress);
    expect(MockWorker.instances).toHaveLength(1);
    const assertion = expect(second).rejects.toThrow('cancelled'); client.cancel(); await assertion;
  });
  it('ends a stuck model download rather than leaving a permanent spinner', async () => {
    vi.useFakeTimers(); const client = new InspectionWorkerClient();
    const pending = client.inspect(sample, 'dinov2', manifest, () => {});
    const assertion = expect(pending).rejects.toThrow('timed out');
    await vi.advanceTimersByTimeAsync(180_001); await assertion;
    expect(MockWorker.instances[0].terminated).toBe(true);
  });
});
