export type Method = 'dinov2' | 'classical' | 'pixel';
export type Disposition = 'defect' | 'acceptable' | 'uncertain';
export type JobStatus = 'queued' | 'processing' | 'ready' | 'failed' | 'cancelled';
export interface Transform { originalWidth: number; originalHeight: number; resizedWidth: number; resizedHeight: number; padLeft: number; padTop: number; inputSize: number }
export interface Sample { id: string; name: string; imageUrl: string; inputUrl: string; sourcePath: string; width: number; height: number; collections: string[]; transform: Transform }
export interface SampleManifest { schemaVersion: 1; dataset: string; sourceUrl: string; license: string; splitVersion: string; samples: Sample[] }
export interface InspectionResult { schemaVersion: 1; sampleId: string; method: Method; artifactVersion: string; score: number; threshold: number; flagged: boolean; heatmap: number[]; gridWidth: number; gridHeight: number; heatmapScale: number; transform: Transform; durationMs: number; completedAt: string }
export interface ReviewRecord { schemaVersion: 1; sampleId: string; disposition: Disposition | null; note: string; inspectionVersions: Partial<Record<Method, string>>; createdAt: string; updatedAt: string }
export interface Job { sampleId: string; method: Method; status: JobStatus; error?: string }
export interface ModelProgress { phase: 'idle' | 'loading' | 'ready' | 'processing' | 'error'; message: string; percent?: number }
export interface MetricEstimate { value: number; low: number; high: number }
export interface MethodEvaluation { method: Method; label: string; threshold: number; truePositives: number; falseNegatives: number; falsePositives: number; trueNegatives: number; recall: MetricEstimate; falseAlarmRate: MetricEstimate; averagePrecision: number; pixelAveragePrecision: number | null; medianLatencyMs: number; p95LatencyMs: number }
export interface EvaluationReport { schemaVersion: 1; status: 'complete'; artifactVersion: string; splitVersion: string; evaluatedAt: string; counts: { fit: number; calibration: number; testNormal: number; testAnomaly: number }; calibrationTarget: number; methods: MethodEvaluation[]; pairedRecallDifference: MetricEstimate; conclusion: string; limitations: string[]; environment: string; results: { sampleId: string; label: 'normal' | 'anomaly'; scores: Partial<Record<Method, number>>; flagged: Partial<Record<Method, boolean>> }[] }
export interface ArtifactMethod { bankUrl: string; bankSha256: string; dimensions: number; count: number; threshold: number; heatmapScale: number; standardizationUrl?: string; standardizationSha256?: string }
export interface ArtifactManifest { schemaVersion: 1; version: string; splitVersion: string; model: { id: string; revision: string; dtype: 'q8'; localPath?: string; sha256: string }; preprocessing: { size: 224; patchSize: 14; padding: number[]; mean: number[]; std: number[] }; methods: Record<Method, ArtifactMethod>; createdAt: string }
export type WorkerRequest = { type: 'inspect'; requestId: string; sample: Sample; method: Method; manifest: ArtifactManifest };
export type WorkerResponse = { type: 'progress'; requestId: string; progress: ModelProgress } | { type: 'result'; requestId: string; result: InspectionResult } | { type: 'error'; requestId: string; error: string };
