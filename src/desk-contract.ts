import type { ArtifactManifest, Disposition, EvaluationReport, InspectionResult, Job, Method, ModelProgress, ReviewRecord, Sample, SampleManifest } from './types';
export interface InspectionDesk {
  loading: boolean; loadError: string | null; samples: Sample[]; sampleManifest: SampleManifest | null;
  collection: string; setCollection: (value: string) => void; selectedId: string | null; selectSample: (id: string) => void;
  method: Method; setMethod: (value: Method) => void;
  results: Record<string, InspectionResult>; reviews: Record<string, ReviewRecord>; jobs: Record<string, Job>;
  progress: ModelProgress; running: boolean; storageWarning: string | null; notice: string | null;
  artifact: ArtifactManifest | null; artifactError: string | null; evaluation: EvaluationReport | null; evaluationError: string | null;
  runBatch: () => void; inspectSample: (id: string) => void; cancel: () => void;
  saveReview: (id: string, patch: { disposition?: Disposition | null; note?: string }) => void;
  exportReviews: (format: 'json' | 'csv') => void; resetSession: () => Promise<void>; retryLoad: () => void;
}
export const resultKey = (sampleId: string, method: Method) => `${sampleId}:${method}`;
