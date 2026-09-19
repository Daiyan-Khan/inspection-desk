import { afterEach, describe, expect, it, vi } from 'vitest';
import { checkedFetch, sha256 } from './artifacts';

afterEach(() => { vi.unstubAllGlobals(); });

describe('artifact cache denial does not require persistent storage', () => {
  it('falls back to verified network bytes when CacheStorage access is denied', async () => {
    const bytes = new Uint8Array([7, 11, 19, 23]).buffer;
    const digest = await sha256(bytes);
    vi.stubGlobal('caches', { open: vi.fn().mockRejectedValue(new DOMException('Cache denied', 'SecurityError')) });
    const fetch = vi.fn(async () => new Response(bytes)); vi.stubGlobal('fetch', fetch);
    expect(new Uint8Array(await checkedFetch('https://example.invalid/weights.onnx', digest))).toEqual(new Uint8Array(bytes));
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('falls back to a fresh integrity check when reading a cache entry fails', async () => {
    const bytes = new Uint8Array([13, 17, 31]).buffer;
    const digest = await sha256(bytes);
    vi.stubGlobal('caches', { open: vi.fn().mockResolvedValue({ match: vi.fn().mockRejectedValue(new Error('Cache read denied')) }) });
    vi.stubGlobal('fetch', vi.fn(async () => new Response(bytes)));
    expect(new Uint8Array(await checkedFetch('https://example.invalid/bank.bin', digest))).toEqual(new Uint8Array(bytes));
  });

  it('returns verified bytes even when caching the download exceeds quota', async () => {
    const bytes = new Uint8Array([41, 43, 47]).buffer;
    const digest = await sha256(bytes);
    const put = vi.fn().mockRejectedValue(new DOMException('Cache quota exceeded', 'QuotaExceededError'));
    vi.stubGlobal('caches', { open: vi.fn().mockResolvedValue({ match: vi.fn().mockResolvedValue(undefined), put }) });
    vi.stubGlobal('fetch', vi.fn(async () => new Response(bytes)));
    expect(new Uint8Array(await checkedFetch('https://example.invalid/bank.bin', digest))).toEqual(new Uint8Array(bytes));
    expect(put).toHaveBeenCalledTimes(1);
  });

  it('still rejects corrupted network bytes when the cache is inaccessible', async () => {
    vi.stubGlobal('caches', { open: vi.fn().mockRejectedValue(new Error('Cache unavailable')) });
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Uint8Array([1, 2, 3]))));
    await expect(checkedFetch('https://example.invalid/model.onnx', '0'.repeat(64))).rejects.toThrow('integrity');
  });

  it('propagates a blocked network request instead of fabricating an artifact', async () => {
    const blocked = new TypeError('Failed to fetch');
    vi.stubGlobal('caches', undefined);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(blocked));
    await expect(checkedFetch('https://example.invalid/model.onnx', '0'.repeat(64))).rejects.toBe(blocked);
  });
});
