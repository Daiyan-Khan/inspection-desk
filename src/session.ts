import type { ArtifactManifest, EvaluationReport, InspectionResult } from './types';

/** Restore saved records without replacing newer decisions made while storage loads. */
export function mergeByTimestamp<T extends object>(saved: Record<string, T>, current: Record<string, T>, field: keyof T): Record<string, T> {
  const merged = { ...saved };
  for (const [key, record] of Object.entries(current)) {
    if (!merged[key] || String(record[field]) >= String(merged[key][field])) merged[key] = record;
  }
  return merged;
}
export function currentResults(results: Record<string, InspectionResult>, manifest: ArtifactManifest | null, splitVersion?: string) {
  if (!manifest || manifest.splitVersion !== splitVersion) return {};
  return Object.fromEntries(Object.entries(results).filter(([, result]) => result.schemaVersion === 1 && result.artifactVersion === manifest.version));
}
export function currentEvaluation(report: EvaluationReport | null, manifest: ArtifactManifest | null, splitVersion?: string): EvaluationReport | null {
  return report && manifest && report.artifactVersion === manifest.version && report.splitVersion === manifest.splitVersion && manifest.splitVersion === splitVersion ? report : null;
}
