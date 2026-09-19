import type { InspectionResult, ReviewRecord, Sample } from './types';
export function csvCell(value: unknown): string {
  let text = value == null ? '' : String(value);
  if (/^[\s]*[=+@-]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}
export function exportSession(format: 'json' | 'csv', samples: Sample[], reviews: Record<string, ReviewRecord>, results: Record<string, InspectionResult>) {
  const rows = Object.values(reviews).map(review => ({ ...review, sourcePath: samples.find(s => s.id === review.sampleId)?.sourcePath ?? '', results: Object.values(results).filter(r => r.sampleId === review.sampleId) }));
  let content: string;
  if (format === 'json') content = JSON.stringify({ schemaVersion: 1, exportedAt: new Date().toISOString(), storage: 'device-local', attribution: 'VisA / Amazon Science, CC BY 4.0. Reviewer decisions are not dataset ground truth.', reviews: rows }, null, 2);
  else {
    const headers = ['sample_id', 'source_path', 'disposition', 'note', 'created_at', 'updated_at', 'inspection_versions'];
    content = [headers.map(csvCell).join(','), ...rows.map(r => [r.sampleId, r.sourcePath, r.disposition, r.note, r.createdAt, r.updatedAt, JSON.stringify(r.inspectionVersions)].map(csvCell).join(','))].join('\r\n');
  }
  const url = URL.createObjectURL(new Blob([content], { type: format === 'json' ? 'application/json' : 'text/csv;charset=utf-8' }));
  const a = document.createElement('a'); a.href = url; a.download = `inspection-desk-${new Date().toISOString().slice(0, 10)}.${format}`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
