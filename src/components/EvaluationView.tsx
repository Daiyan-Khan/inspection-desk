import { useState } from 'react';
import { ArrowRight, ArrowUpRight, BarChart3, Check, ChevronDown, CircleAlert, FlaskConical, Info, RefreshCw } from 'lucide-react';
import type { EvaluationReport, MetricEstimate } from '../types';

interface EvaluationViewProps {
  report: EvaluationReport | null;
  loading?: boolean;
  error?: string | null;
  onRetry: () => void;
  onOpenSample: (id: string) => void;
  sampleIds: Set<string>;
}

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const interval = (metric: MetricEstimate) => `${percent(metric.low)}–${percent(metric.high)}`;
const methodNames = { dinov2: 'AI reference model', classical: 'Classical vision', pixel: 'Pixel reference' };

export function EvaluationView({ report, loading, error, onRetry, onOpenSample, sampleIds }: EvaluationViewProps) {
  const [tab, setTab] = useState<'results' | 'protocol' | 'errors'>('results');
  const testCount = report ? report.counts.testNormal + report.counts.testAnomaly : 0;
  const errors = report?.results.filter(item => item.flagged.dinov2 !== undefined && item.flagged.dinov2 !== (item.label === 'anomaly')) ?? [];

  return <div className="evaluation-view">
    <div className="page-heading evaluation-heading"><div><div className="eyebrow"><FlaskConical size={13} /> MEASURED, NOT ASSUMED</div><h1>The evidence behind the model.</h1><p>A fixed holdout. Two non-neural baselines. The mistakes included.</p></div><span className={`status-pill ${report ? 'teal' : ''}`}><span className="status-dot" />{report ? 'Evaluation complete' : 'Results pending'}</span></div>
    <nav className="section-tabs" aria-label="Evaluation sections">
      <button className={tab === 'results' ? 'active' : ''} onClick={() => setTab('results')}>Results</button>
      <button className={tab === 'protocol' ? 'active' : ''} onClick={() => setTab('protocol')}>Protocol & limits</button>
      <button className={tab === 'errors' ? 'active' : ''} onClick={() => setTab('errors')}>Error analysis {report && <span>{errors.length}</span>}</button>
    </nav>

    {!report && <section className="evaluation-empty panel">
      <span className="empty-symbol"><BarChart3 size={28} /></span><div className="eyebrow">NO ESTIMATED NUMBERS</div><h2>{loading ? 'Loading the evaluation report' : 'Measured results will appear here.'}</h2><p>{error || 'This view reads the versioned benchmark report. It never substitutes illustrative performance figures.'}</p>
      <button className="button secondary" onClick={onRetry} disabled={loading}><RefreshCw size={15} className={loading ? 'spin' : ''} />{loading ? 'Loading report' : 'Check for results'}</button>
      <div className="empty-report-preview"><span><Check size={14} /> Fixed dataset splits</span><span><Check size={14} /> Same test images</span><span><Check size={14} /> False alarms reported</span></div>
    </section>}

    {report && tab === 'results' && <>
      <div className="evaluation-summary-grid">
        <section className="metric-card primary-metric"><div className="eyebrow">RECALL DIFFERENCE · AI VS CLASSICAL</div><div className="metric-value">{report.pairedRecallDifference.value > 0 ? '+' : ''}{(report.pairedRecallDifference.value * 100).toFixed(1)}<span>pp</span></div><p>Paired interval: {(report.pairedRecallDifference.low * 100).toFixed(1)} to {(report.pairedRecallDifference.high * 100).toFixed(1)} points</p></section>
        <section className="metric-card"><div className="eyebrow">HELD-OUT IMAGES</div><div className="metric-value">{testCount}<span>samples</span></div><p>{report.counts.testNormal} normal · {report.counts.testAnomaly} anomaly</p></section>
        <section className="metric-card"><div className="eyebrow">CALIBRATION TARGET</div><div className="metric-value">{percent(report.calibrationTarget)}</div><p>Normal-image false alarms on calibration data</p></section>
      </div>
      <section className="panel comparison-panel"><div className="panel-heading"><div><div className="eyebrow">SAME IMAGES. DIFFERENT METHODS.</div><h2>Does AI improve the inspection?</h2></div><span className="small-muted">Test performance</span></div>
        <div className="comparison-table-wrap"><table className="comparison-table"><thead><tr><th scope="col">Method</th><th scope="col">Defect recall</th><th scope="col">False alarms</th><th scope="col">Image AP</th><th scope="col">Median latency</th></tr></thead><tbody>{report.methods.map(method => <tr key={method.method} className={method.method === 'dinov2' ? 'highlight-row' : ''}><th scope="row"><span className={`method-mark ${method.method}`} />{methodNames[method.method]}<small>{method.label}</small></th><td><strong>{percent(method.recall.value)}</strong><small>{interval(method.recall)} interval</small><div className="mini-bar"><span style={{ width: `${method.recall.value * 100}%` }} /></div></td><td><strong>{percent(method.falseAlarmRate.value)}</strong><small>{interval(method.falseAlarmRate)} interval</small></td><td>{percent(method.averagePrecision)}</td><td className="mono">{method.medianLatencyMs.toFixed(0)} ms</td></tr>)}</tbody></table></div>
        <div className="table-note"><Info size={14} /><p>Thresholds are chosen on calibration images. Actual test false-alarm rates can differ. These timings describe the evaluation environment, not your browser.</p></div>
        <details className="evaluation-details">
          <summary><div><strong>Localisation, latency & raw counts</strong><span>Inspect the supporting measurements for every method.</span></div><ChevronDown size={17} /></summary>
          <div className="method-evidence-grid">
            {report.methods.map(method => <section className="method-evidence-card" key={method.method}>
              <h3><span className={`method-mark ${method.method}`} />{methodNames[method.method]}</h3>
              <dl className="supporting-metrics">
                <div><dt>Pixel average precision</dt><dd>{method.pixelAveragePrecision === null ? 'Not measured' : percent(method.pixelAveragePrecision)}</dd></div>
                <div><dt>95th-percentile latency</dt><dd>{method.p95LatencyMs.toFixed(0)}<span> ms</span></dd></div>
              </dl>
              <div className="count-grid" aria-label={`${methodNames[method.method]} confusion counts`}>
                <div><span>Detected anomalies <abbr title="True positives">TP</abbr></span><strong>{method.truePositives}</strong></div>
                <div><span>Missed anomalies <abbr title="False negatives">FN</abbr></span><strong>{method.falseNegatives}</strong></div>
                <div><span>False alarms <abbr title="False positives">FP</abbr></span><strong>{method.falsePositives}</strong></div>
                <div><span>Normal, below threshold <abbr title="True negatives">TN</abbr></span><strong>{method.trueNegatives}</strong></div>
              </div>
              <p className="evidence-threshold">Image-level counts at threshold <code>{method.threshold.toFixed(3)}</code></p>
            </section>)}
          </div>
          <div className="evidence-explainer"><Info size={14} /><p>Pixel average precision compares coarse anomaly maps with dataset masks on valid image pixels. It measures localisation ranking, not defect severity. The 95th-percentile latency describes the slower end of the benchmark timings; it is not a browser speed guarantee.</p></div>
        </details>
      </section>
      <section className="conclusion-panel"><div className="conclusion-icon"><FlaskConical size={20} /></div><div><div className="eyebrow">WHAT THE RESULTS SUPPORT</div><h3>{report.conclusion}</h3><p>Benchmark correctness is separate from saved reviewer time or manufacturing impact. No factory validation is claimed.</p></div><button className="text-button" onClick={() => setTab('errors')}>Inspect the mistakes <ArrowRight size={16} /></button></section>
    </>}

    {tab === 'protocol' && <div className="protocol-grid">
      <section className="panel protocol-panel"><div className="eyebrow">01 / DATA BOUNDARIES</div><h2>Learn from normal. Test separately.</h2><p>The reference bank contains normal images. Calibration selects thresholds; a separate holdout measures performance. The interactive collection is a curated demonstration, not a representative production stream.</p><div className="split-diagram"><div><span>Reference fit</span><strong>{report?.counts.fit ?? '—'}</strong><small>Normal appearance</small></div><ArrowRight size={15} /><div><span>Calibration</span><strong>{report?.counts.calibration ?? '—'}</strong><small>Threshold selection</small></div><ArrowRight size={15} /><div><span>Test holdout</span><strong>{report ? testCount : '—'}</strong><small>Final measurement</small></div></div></section>
      <section className="panel protocol-panel"><div className="eyebrow">02 / FAIR COMPARISON</div><h2>More than an easy baseline.</h2><p>AI reference matching is compared with classical appearance features and a pixel reference method. Each method uses its own calibration threshold and the same held-out images.</p><ul className="check-list"><li><Check size={15} /> Source splits and artifact versions are recorded</li><li><Check size={15} /> Recall is shown alongside false alarms</li><li><Check size={15} /> Review decisions do not alter benchmark labels</li></ul></section>
      <section className="panel protocol-panel wide"><div className="eyebrow">03 / HONEST LIMITS</div><h2>A benchmark is a beginning.</h2><ul className="limitation-list">{(report?.limitations ?? ['Results apply to the selected VisA product category and capture conditions.', 'An anomaly score does not establish defect cause, severity or production acceptability.', 'Curated public images do not establish time savings or performance in a real factory.']).map((limitation, index) => <li key={index}><CircleAlert size={15} /><span>{limitation}</span></li>)}</ul></section>
      {report && <section className="provenance-strip"><div><span>Artifact</span><code>{report.artifactVersion}</code></div><div><span>Split</span><code>{report.splitVersion}</code></div><div><span>Evaluated</span><code>{new Date(report.evaluatedAt).toLocaleDateString('en-AU')}</code></div><div><span>Environment</span><code>{report.environment}</code></div></section>}
    </div>}

    {report && tab === 'errors' && <div className="error-analysis"><div className="error-intro"><div><div className="eyebrow">THE USEFUL PART OF AN EVALUATION</div><h2>Where the model gets it wrong.</h2><p>False negatives need attention. False alarms cost reviewer time. Both belong in the result.</p></div><span className="status-pill orange">{errors.length} errors at the calibrated threshold</span></div><section className="panel"><div className="comparison-table-wrap"><table className="comparison-table error-table"><thead><tr><th>Sample</th><th>Benchmark label</th><th>AI result</th><th>Score</th><th>Review</th></tr></thead><tbody>{errors.map(item => <tr key={item.sampleId}><th className="mono">{item.sampleId}</th><td>{item.label === 'anomaly' ? 'Anomaly' : 'Normal'}</td><td><span className={`status-pill ${item.label === 'anomaly' ? 'orange' : ''}`}>{item.label === 'anomaly' ? 'Missed anomaly' : 'False alarm'}</span></td><td className="mono">{item.scores.dinov2?.toFixed(3) ?? '—'}</td><td>{sampleIds.has(item.sampleId) ? <button className="text-button" onClick={() => onOpenSample(item.sampleId)}>Open image <ArrowUpRight size={14} /></button> : <span className="small-muted">Not in demo collection</span>}</td></tr>)}</tbody></table></div>{errors.length === 0 && <div className="no-errors"><Check size={20} /><p>No image-level errors in this reported holdout. A small error-free sample does not establish production reliability.</p></div>}</section></div>}
  </div>;
}
