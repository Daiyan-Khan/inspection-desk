import { openDB, type DBSchema } from 'idb';
import type { InspectionResult, ReviewRecord } from './types';
import { resultKey } from './desk-contract';

interface DeskDatabase extends DBSchema {
  results: { key: string; value: InspectionResult };
  reviews: { key: string; value: ReviewRecord };
}
let database: ReturnType<typeof openDB<DeskDatabase>> | undefined;
const getDatabase = () => database ??= openDB<DeskDatabase>('inspection-desk-v1', 1, {
  upgrade(db) { db.createObjectStore('results'); db.createObjectStore('reviews'); },
});
export async function loadSession() {
  const db = await getDatabase();
  const [results, reviews] = await Promise.all([db.getAll('results'), db.getAll('reviews')]);
  return { results: Object.fromEntries(results.map(r => [resultKey(r.sampleId, r.method), r])), reviews: Object.fromEntries(reviews.map(r => [r.sampleId, r])) };
}
export async function persistResult(result: InspectionResult) { const db = await getDatabase(); await db.put('results', result, resultKey(result.sampleId, result.method)); }
export async function persistReview(review: ReviewRecord) { const db = await getDatabase(); await db.put('reviews', review, review.sampleId); }
export async function clearSession() { const db = await getDatabase(); const tx = db.transaction(['results', 'reviews'], 'readwrite'); await Promise.all([tx.objectStore('results').clear(), tx.objectStore('reviews').clear(), tx.done]); }
