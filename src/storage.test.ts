import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { InspectionResult, ReviewRecord } from './types';
import { checkedFetch } from './ml/artifacts';

const mocks = vi.hoisted(() => ({ openDB: vi.fn() }));
vi.mock('idb', () => ({ openDB: mocks.openDB }));

const review: ReviewRecord = {
  schemaVersion: 1, sampleId: 'reviewable-photo', disposition: 'uncertain',
  note: 'Keep this manual observation', inspectionVersions: {},
  createdAt: '2026-09-20T00:00:00.000Z', updatedAt: '2026-09-20T00:00:00.000Z',
};
const result = { sampleId: review.sampleId, method: 'dinov2' } as InspectionResult;

beforeEach(() => { vi.resetModules(); mocks.openDB.mockReset(); });
afterEach(() => { vi.unstubAllGlobals(); });

describe('persistent storage failures reach the caller', () => {
  it('reports unavailable IndexedDB without altering the supplied review', async () => {
    mocks.openDB.mockImplementation(() => { throw new ReferenceError('indexedDB is not defined'); });
    const storage = await import('./storage');
    await expect(storage.loadSession()).rejects.toThrow('indexedDB is not defined');
    await expect(storage.persistReview(review)).rejects.toThrow('indexedDB is not defined');
    expect(review.note).toBe('Keep this manual observation');
  });

  it('propagates an asynchronous browser permission denial for reads and writes', async () => {
    const denied = new DOMException('IndexedDB access denied', 'SecurityError');
    mocks.openDB.mockRejectedValue(denied);
    const storage = await import('./storage');
    await expect(storage.loadSession()).rejects.toBe(denied);
    await expect(storage.persistReview(review)).rejects.toBe(denied);
    await expect(storage.persistResult(result)).rejects.toBe(denied);
    await expect(storage.clearSession()).rejects.toBe(denied);
  });

  it.each(['review', 'result'] as const)('does not report a successful %s save when quota is exhausted', async kind => {
    const full = new DOMException('Browser storage quota exceeded', 'QuotaExceededError');
    const put = vi.fn().mockRejectedValue(full);
    mocks.openDB.mockResolvedValue({ put });
    const storage = await import('./storage');
    const operation = kind === 'review' ? storage.persistReview(review) : storage.persistResult(result);
    await expect(operation).rejects.toBe(full);
    expect(put).toHaveBeenCalledWith(kind === 'review' ? 'reviews' : 'results', kind === 'review' ? review : result, expect.any(String));
    expect(review.disposition).toBe('uncertain');
    expect(review.note).toBe('Keep this manual observation');
  });

  it('reports a rejected clear operation rather than claiming saved data was removed', async () => {
    const failure = new DOMException('Clear could not be committed', 'QuotaExceededError');
    const clear = vi.fn().mockRejectedValue(failure);
    mocks.openDB.mockResolvedValue({ transaction: () => ({ objectStore: () => ({ clear }), done: Promise.resolve() }) });
    const storage = await import('./storage');
    await expect(storage.clearSession()).rejects.toBe(failure);
  });

  it('also reports a transaction abort after individual clear requests resolve', async () => {
    const aborted = new DOMException('Transaction aborted', 'AbortError');
    mocks.openDB.mockResolvedValue({ transaction: () => ({
      objectStore: () => ({ clear: () => Promise.resolve() }),
      done: Promise.reject(aborted),
    }) });
    const storage = await import('./storage');
    await expect(storage.clearSession()).rejects.toBe(aborted);
  });
});

describe('manual storage is independent from inference downloads', () => {
  it('can retain and update a manual review after an artifact fetch is blocked', async () => {
    const reviews = new Map<string, ReviewRecord>();
    mocks.openDB.mockResolvedValue({
      put: vi.fn(async (store: string, value: ReviewRecord, key: string) => { if (store === 'reviews') reviews.set(key, structuredClone(value)); }),
      getAll: vi.fn(async (store: string) => store === 'reviews' ? [...reviews.values()] : []),
    });
    vi.stubGlobal('caches', undefined);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Network request blocked')));
    const storage = await import('./storage');
    await storage.persistReview(review);
    await expect(checkedFetch('https://example.invalid/model.onnx', '0'.repeat(64))).rejects.toThrow('Network request blocked');
    const updated = { ...review, note: 'Updated manually while inference is unavailable' };
    await storage.persistReview(updated);
    const session = await storage.loadSession();
    expect(session.reviews[review.sampleId]).toEqual(updated);
    expect(session.results).toEqual({});
  });
});
