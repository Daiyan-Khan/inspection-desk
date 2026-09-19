import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ArtifactManifest, EvaluationReport, InspectionResult, Job, Method, ModelProgress, ReviewRecord, SampleManifest } from '../types';
import { resultKey, type InspectionDesk } from '../desk-contract';
import { clearSession, loadSession, persistResult, persistReview } from '../storage';
import { exportSession } from '../export';
import { InspectionWorkerClient } from '../worker-client';
import { currentEvaluation, currentResults, mergeByTimestamp } from '../session';

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}${path}`, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status}).`);
  return response.json() as Promise<T>;
}
const storageMessage = 'Local storage is unavailable or full. Your work remains in this tab; export it before leaving.';
export function useInspectionDesk(): InspectionDesk {
  const [loading, setLoading] = useState(true), [loadError, setLoadError] = useState<string | null>(null);
  const [sampleManifest, setSampleManifest] = useState<SampleManifest | null>(null);
  const [artifact, setArtifact] = useState<ArtifactManifest | null>(null), [artifactError, setArtifactError] = useState<string | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationReport | null>(null), [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [collection, setCollectionState] = useState('starter'), [selectedId, setSelectedId] = useState<string | null>(null);
  const [method, setMethod] = useState<Method>('dinov2');
  const [results, setResults] = useState<Record<string, InspectionResult>>({}), [reviews, setReviews] = useState<Record<string, ReviewRecord>>({});
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [progress, setProgress] = useState<ModelProgress>({ phase: 'idle', message: 'Ready when you are. Images are processed on this device.' });
  const [running, setRunning] = useState(false), [storageWarning, setStorageWarning] = useState<string | null>(null), [notice, setNotice] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const client = useRef(new InspectionWorkerClient()), generation = useRef(0), active = useRef(false), reviewRef = useRef(reviews);
  const storageGeneration = useRef(0);
  reviewRef.current = reviews;

  useEffect(() => {
    let live = true; setLoading(true); setLoadError(null);
    const readSamples = getJson<SampleManifest>('data/sample-manifest.json').then(value => {
      if (value.schemaVersion !== 1 || !Array.isArray(value.samples) || !value.samples.length) throw new Error('The sample collection is missing or incompatible.');
      if (live) { setSampleManifest(value); setSelectedId(old => old ?? value.samples.find(s => s.collections.includes('starter'))?.id ?? value.samples[0].id); }
    }).catch(error => { if (live) setLoadError(String(error.message)); });
    const readArtifacts = getJson<ArtifactManifest>('artifacts/manifest.json').then(value => {
      if (value.schemaVersion !== 1 || !value.version || !value.methods) throw new Error('The inference artifacts are incompatible.');
      if (live) { setArtifact(value); setArtifactError(null); }
    }).catch(() => { if (live) { setArtifact(null); setArtifactError('Inference artifacts are unavailable. You can still review every image manually.'); } });
    const readEvaluation = getJson<EvaluationReport>('data/evaluation.json').then(value => {
      if (value.schemaVersion !== 1 || value.status !== 'complete') throw new Error('Evaluation is not complete.');
      if (live) { setEvaluation(value); setEvaluationError(null); }
    }).catch(() => { if (live) { setEvaluation(null); setEvaluationError('The frozen evaluation report is not available. No benchmark results are being estimated.'); } });
    void Promise.all([readSamples, readArtifacts, readEvaluation]).finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [reload]);
  useEffect(() => {
    let live = true;
    const epoch = storageGeneration.current;
    void loadSession().then(value => {
      if (!live || epoch !== storageGeneration.current) return;
      setResults(current => mergeByTimestamp(value.results, current, 'completedAt'));
      setReviews(current => { const merged = mergeByTimestamp(value.reviews, current, 'updatedAt'); reviewRef.current = merged; return merged; });
    }).catch(() => { if (live) setStorageWarning(storageMessage); });
    return () => { live = false; };
  }, []);
  useEffect(() => () => { generation.current++; active.current = false; client.current.cancel(); }, []);
  const samples = useMemo(() => sampleManifest?.samples.filter(s => s.collections.includes(collection)) ?? [], [sampleManifest, collection]);
  const visibleResults = useMemo(() => currentResults(results, artifact, sampleManifest?.splitVersion), [results, artifact, sampleManifest]);
  const visibleEvaluation = currentEvaluation(evaluation, artifact, sampleManifest?.splitVersion);
  const visibleEvaluationError = evaluation && artifact && sampleManifest && !visibleEvaluation ? 'The evaluation report does not match the current model and dataset. Retry loading the complete release.' : evaluationError;
  const setCollection = (value: string) => { setCollectionState(value); setSelectedId(sampleManifest?.samples.find(s => s.collections.includes(value))?.id ?? null); };
  const processSamples = useCallback(async (ids: string[], force: boolean) => {
    if (active.current) return;
    if (!artifact || !sampleManifest) { setNotice('Inference is unavailable. Manual review is ready to use.'); return; }
    if (artifact.splitVersion !== sampleManifest.splitVersion) { setNotice('Dataset and model versions do not match. Reload the app to update the artifacts.'); return; }
    const selectedMethod = method;
    const todo = ids.slice(0, 10).filter(id => force || results[resultKey(id, selectedMethod)]?.artifactVersion !== artifact.version);
    if (!todo.length) { setNotice('This batch already has results for this method. Select an image to run it again.'); return; }
    const run = ++generation.current; active.current = true; setRunning(true); setNotice(null);
    setJobs(old => ({ ...old, ...Object.fromEntries(todo.map(id => [resultKey(id, selectedMethod), { sampleId: id, method: selectedMethod, status: 'queued' as const }])) }));
    for (const id of todo) {
      if (run !== generation.current) break;
      const sample = sampleManifest.samples.find(s => s.id === id); if (!sample) continue;
      const key = resultKey(id, selectedMethod);
      setJobs(old => ({ ...old, [key]: { sampleId: id, method: selectedMethod, status: 'processing' } }));
      try {
        const result = await client.current.inspect(sample, selectedMethod, artifact, value => { if (run === generation.current) setProgress(value); });
        if (run !== generation.current) break;
        setResults(old => ({ ...old, [key]: result }));
        setJobs(old => ({ ...old, [key]: { sampleId: id, method: selectedMethod, status: 'ready' } }));
        try { await persistResult(result); } catch { setStorageWarning(storageMessage); }
      } catch (error) {
        if (run !== generation.current) break;
        const message = error instanceof Error ? error.message : 'Unable to process this image.';
        setJobs(old => ({ ...old, [key]: { sampleId: id, method: selectedMethod, status: 'failed', error: message } }));
        setProgress({ phase: 'error', message });
        // Avoid retrying a broken model download for every remaining image.
        setJobs(old => Object.fromEntries(Object.entries(old).map(([k, job]) => [k, job.status === 'queued' ? { ...job, status: 'cancelled' as const } : job])));
        break;
      }
    }
    if (run === generation.current) { active.current = false; setRunning(false); setProgress(old => old.phase === 'error' ? old : { phase: 'ready', message: 'Processing complete. Review the suggestions and record your decision.' }); }
  }, [artifact, sampleManifest, method, results]);
  const cancel = () => {
    generation.current++; active.current = false; client.current.cancel(); setRunning(false);
    setJobs(old => Object.fromEntries(Object.entries(old).map(([key, job]) => [key, ['processing', 'queued'].includes(job.status) ? { ...job, status: 'cancelled' as const } : job])));
    setProgress({ phase: 'idle', message: 'Processing cancelled. Completed results and your reviews are saved.' });
  };
  const saveReview: InspectionDesk['saveReview'] = (id, patch) => {
    const now = new Date().toISOString();
    const previous = reviewRef.current[id];
    const linked = Object.fromEntries(Object.values(visibleResults).filter(r => r.sampleId === id).map(r => [r.method, r.artifactVersion]));
    const review: ReviewRecord = { ...(previous ?? { schemaVersion: 1, sampleId: id, disposition: null, note: '', createdAt: now }), ...patch, inspectionVersions: { ...previous?.inspectionVersions, ...linked }, updatedAt: now };
    reviewRef.current = { ...reviewRef.current, [id]: review }; setReviews(reviewRef.current);
    void persistReview(review).catch(() => setStorageWarning(storageMessage));
  };
  const resetSession = async () => {
    cancel();
    storageGeneration.current++;
    let persistentCleared = true;
    try { await clearSession(); setStorageWarning(null); } catch { persistentCleared = false; setStorageWarning('This tab was cleared, but saved browser records could not be removed. They may return after reloading; clear this site’s data in browser settings to remove them.'); }
    setResults({}); setReviews({}); reviewRef.current = {}; setJobs({}); setNotice(persistentCleared ? 'Your local review session has been reset.' : 'The current tab has been reset. Persistent storage could not be cleared.');
  };
  return { loading, loadError, samples, sampleManifest, collection, setCollection, selectedId, selectSample: setSelectedId, method, setMethod,
    results: visibleResults, reviews, jobs, progress, running, storageWarning, notice, artifact, artifactError, evaluation: visibleEvaluation, evaluationError: visibleEvaluationError,
    runBatch: () => { void processSamples(samples.map(s => s.id), false); }, inspectSample: id => { void processSamples([id], true); }, cancel, saveReview,
    exportReviews: format => exportSession(format, sampleManifest?.samples ?? [], reviews, results), resetSession, retryLoad: () => setReload(n => n + 1) };
}
