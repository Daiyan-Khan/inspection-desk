import { describe, expect, it } from 'vitest';
import { currentEvaluation, currentResults, mergeByTimestamp } from './session';
import type { ArtifactManifest, EvaluationReport, InspectionResult, ReviewRecord } from './types';

const earlier = '2026-09-20T01:00:00.000Z';
const later = '2026-09-20T01:01:00.000Z';
const artifact = { version: 'release-current', splitVersion: 'split-current' } as ArtifactManifest;
const review = (sampleId: string, note: string, updatedAt = earlier): ReviewRecord => ({
  schemaVersion: 1, sampleId, disposition: 'uncertain', note,
  inspectionVersions: {}, createdAt: earlier, updatedAt,
});
const result = (sampleId: string, artifactVersion = artifact.version, completedAt = earlier): InspectionResult => ({
  schemaVersion: 1, sampleId, method: 'dinov2', artifactVersion,
  score: 0.4, threshold: 0.5, flagged: false, heatmap: [0.4], gridWidth: 1, gridHeight: 1,
  heatmapScale: 1, durationMs: 120, completedAt,
  transform: { originalWidth: 224, originalHeight: 224, resizedWidth: 224, resizedHeight: 224, padLeft: 0, padTop: 0, inputSize: 224 },
});
const report = { artifactVersion: artifact.version, splitVersion: artifact.splitVersion } as EvaluationReport;

describe('session restoration', () => {
  it('restores saved reviews without overwriting decisions made while storage was loading', () => {
    const saved = { first: review('first', 'Old note'), second: review('second', 'Saved review') };
    const current = { first: review('first', 'New tab-only note', later), third: review('third', 'New review', later) };
    expect(mergeByTimestamp(saved, current, 'updatedAt')).toEqual({
      first: current.first, second: saved.second, third: current.third,
    });
    expect(saved.first.note).toBe('Old note');
    expect(Object.keys(current)).toEqual(['first', 'third']);
  });

  it('keeps the current decision when two edits share the same millisecond', () => {
    const saved = { first: review('first', 'Before this click') };
    const current = { first: { ...review('first', 'After this click'), disposition: 'defect' as const } };
    expect(mergeByTimestamp(saved, current, 'updatedAt').first).toBe(current.first);
  });

  it('keeps a newer saved record and supports result completion timestamps', () => {
    const saved = { 'first:dinov2': result('first', artifact.version, later) };
    const current = { 'first:dinov2': result('first'), 'second:dinov2': result('second', artifact.version, later) };
    const merged = mergeByTimestamp(saved, current, 'completedAt');
    expect(merged['first:dinov2']).toBe(saved['first:dinov2']);
    expect(merged['second:dinov2']).toBe(current['second:dinov2']);
  });
});

describe('displayed artifact compatibility', () => {
  it('hides stale and incompatible cached scores while preserving current results', () => {
    const current = result('current');
    const cached = {
      'current:dinov2': current,
      'old:dinov2': result('old', 'release-old'),
      'incompatible:dinov2': { ...result('incompatible'), schemaVersion: 2 } as unknown as InspectionResult,
    };
    expect(currentResults(cached, artifact, artifact.splitVersion)).toEqual({ 'current:dinov2': current });
    expect(Object.keys(cached)).toHaveLength(3);
  });

  it('does not display cached scores until the model and dataset agree', () => {
    const cached = { 'first:dinov2': result('first') };
    expect(currentResults(cached, null, artifact.splitVersion)).toEqual({});
    expect(currentResults(cached, artifact)).toEqual({});
    expect(currentResults(cached, artifact, 'split-previous')).toEqual({});
  });

  it('shows the original evaluation only when all release identifiers match', () => {
    expect(currentEvaluation(report, artifact, artifact.splitVersion)).toBe(report);
    expect(currentEvaluation({ ...report, artifactVersion: 'release-old' }, artifact, artifact.splitVersion)).toBeNull();
    expect(currentEvaluation({ ...report, splitVersion: 'split-old' }, artifact, artifact.splitVersion)).toBeNull();
    expect(currentEvaluation(report, artifact, 'split-old')).toBeNull();
  });

  it('keeps evaluation pending while any required release artifact is unavailable', () => {
    expect(currentEvaluation(null, artifact, artifact.splitVersion)).toBeNull();
    expect(currentEvaluation(report, null, artifact.splitVersion)).toBeNull();
    expect(currentEvaluation(report, artifact)).toBeNull();
  });
});
