import { Provenance, RunSummary } from '../api'
import { fmtInt, shortHash } from '../format'

export default function ProvenanceFooter({
  provenance,
  environment,
  runName,
}: {
  provenance: Provenance | null
  environment: RunSummary['environment']
  runName: string
}) {
  if (!provenance) return null
  const ns = provenance.news_sentiment ?? {}
  const newsPath = ns.source_path ?? ns.path
  const newsHash = ns.source_hash ?? ns.hash
  const align = provenance.alignment_report ?? {}
  const alignEntries = Object.entries(align)

  return (
    <footer className="panel provenance">
      <h2>Nguồn gốc dữ liệu</h2>
      <div className="prov-grid">
        <div className="prov-item">
          <span className="prov-label">Lần chạy</span>
          <code>{runName}</code>
        </div>
        <div className="prov-item">
          <span className="prov-label">Chế độ tin tức</span>
          <code>{ns.mode ?? '—'}</code>
        </div>
        <div className="prov-item">
          <span className="prov-label">Nguồn tin tức</span>
          <code title={newsPath}>{newsPath ?? '—'}</code>
          {ns.rows !== undefined && (
            <span className="prov-sub">{fmtInt(ns.rows)} dòng</span>
          )}
        </div>
        <div className="prov-item">
          <span className="prov-label">Giờ chốt phiên</span>
          <code>
            {provenance.session_cutoff ?? '—'} {provenance.timezone ?? ''}
          </code>
        </div>
        <div className="prov-item">
          <span className="prov-label">Hash giá</span>
          <code title={provenance.prices_hash}>{shortHash(provenance.prices_hash)}</code>
        </div>
        <div className="prov-item">
          <span className="prov-label">Hash tin tức</span>
          <code title={newsHash}>{shortHash(newsHash)}</code>
        </div>
        {ns.feature_hash && (
          <div className="prov-item">
            <span className="prov-label">Hash đặc trưng</span>
            <code title={ns.feature_hash}>{shortHash(ns.feature_hash)}</code>
          </div>
        )}
        <div className="prov-item">
          <span className="prov-label">Hash panel</span>
          <code title={provenance.panel_hash}>{shortHash(provenance.panel_hash)}</code>
        </div>
        {environment?.python != null && (
          <div className="prov-item">
            <span className="prov-label">Môi trường</span>
            <code>
              Python {String(environment.python)} · numpy{' '}
              {String(environment.numpy ?? '?')} · pandas{' '}
              {String(environment.pandas ?? '?')}
            </code>
          </div>
        )}
      </div>
      {alignEntries.length > 0 && (
        <div className="align-row">
          <span className="prov-label">Báo cáo căn chỉnh phiên:</span>
          {alignEntries.map(([k, v]) => (
            <span className="align-chip" key={k}>
              {k}: <strong>{fmtInt(v)}</strong>
            </span>
          ))}
        </div>
      )}
    </footer>
  )
}
