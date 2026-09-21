import { useEffect, useState } from 'react'
import {
  api,
  InformationGain,
  MetricsRow,
  RunInfo,
  RunSummary,
  StratifiedRow,
} from './api'
import OverviewCards from './components/OverviewCards'
import MetricsTable from './components/MetricsTable'
import InformationGainSection from './components/InformationGainSection'
import StratifiedSection from './components/StratifiedSection'
import PredictionsExplorer from './components/PredictionsExplorer'
import ProvenanceFooter from './components/ProvenanceFooter'

function ModeBadge({ mode }: { mode: RunInfo['mode'] }) {
  if (mode === 'real') return <span className="badge badge-real">tin tức thật</span>
  if (mode === 'neutral_prior')
    return <span className="badge badge-neutral">neutral prior</span>
  return <span className="badge badge-muted">không tin tức</span>
}

export default function App() {
  const [runs, setRuns] = useState<RunInfo[] | null>(null)
  const [runsError, setRunsError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)

  const [summary, setSummary] = useState<RunSummary | null>(null)
  const [metricRows, setMetricRows] = useState<MetricsRow[]>([])
  const [stratRows, setStratRows] = useState<StratifiedRow[]>([])
  const [ig, setIg] = useState<InformationGain | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .runs()
      .then((data) => {
        if (cancelled) return
        setRuns(data.runs)
        const preferred =
          data.runs.find((r) => r.has_information_gain && r.mode === 'real') ??
          data.runs.find((r) => r.has_information_gain) ??
          data.runs[0]
        if (preferred) setSelected(preferred.name)
      })
      .catch((e: Error) => {
        if (!cancelled) setRunsError(e.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selected) return
    let cancelled = false
    setDetailLoading(true)
    setDetailError(null)
    setSummary(null)
    setIg(null)
    Promise.all([
      api.summary(selected),
      api.metrics(selected),
      api.stratified(selected),
      api.informationGain(selected),
    ])
      .then(([s, m, st, gain]) => {
        if (cancelled) return
        setSummary(s)
        setMetricRows(m.rows)
        setStratRows(st.rows)
        setIg(gain)
      })
      .catch((e: Error) => {
        if (!cancelled) setDetailError(e.message)
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selected])

  const currentRun = runs?.find((r) => r.name === selected) ?? null

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Dự báo xu hướng cổ phiếu</h1>
          <p className="subtitle">
            Bảng kết quả thực nghiệm — so sánh mô hình chỉ dùng giá và mô hình hai nhánh
            kết hợp tin tức
          </p>
        </div>
        <div className="run-picker">
          <label htmlFor="run-select">Lần chạy</label>
          <div className="run-picker-row">
            <select
              id="run-select"
              value={selected ?? ''}
              onChange={(e) => setSelected(e.target.value)}
              disabled={!runs || runs.length === 0}
            >
              {!runs && <option>Đang tải…</option>}
              {runs?.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name}
                </option>
              ))}
            </select>
            {currentRun && <ModeBadge mode={currentRun.mode} />}
          </div>
        </div>
      </header>

      {runsError && (
        <div className="alert alert-error">
          Không tải được danh sách lần chạy: {runsError}. Kiểm tra API tại{' '}
          <code>/api/runs</code>.
        </div>
      )}
      {runs && runs.length === 0 && (
        <div className="alert">Chưa có lần chạy nào trong thư mục outputs.</div>
      )}

      {detailError && (
        <div className="alert alert-error">Lỗi tải dữ liệu: {detailError}</div>
      )}
      {detailLoading && <div className="alert">Đang tải dữ liệu lần chạy…</div>}

      {summary && !detailLoading && (
        <main>
          <OverviewCards summary={summary} run={currentRun} />
          <MetricsTable summary={summary} metricRows={metricRows} />
          {ig && <InformationGainSection ig={ig} />}
          <StratifiedSection
            stratifiedByNews={summary.stratified_by_news}
            rows={stratRows}
          />
          <PredictionsExplorer
            runName={summary.name}
            arms={Object.keys(summary.summary ?? {})}
            tickers={summary.panel_tickers ?? []}
            nWindows={
              typeof summary.config?.n_windows === 'number'
                ? (summary.config.n_windows as number)
                : stratRows.reduce((m, r) => Math.max(m, r.window), 0)
            }
          />
          <ProvenanceFooter
            provenance={summary.provenance}
            environment={summary.environment}
            runName={summary.name}
          />
        </main>
      )}
    </div>
  )
}
