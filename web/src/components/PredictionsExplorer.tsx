import { useEffect, useState } from 'react'
import { api, PredictionsResp } from '../api'
import { armLabel, fmtInt } from '../format'

const PAGE_SIZE = 100

export default function PredictionsExplorer({
  runName,
  arms,
  tickers,
  nWindows,
}: {
  runName: string
  arms: string[]
  tickers: string[]
  nWindows: number
}) {
  const [arm, setArm] = useState('')
  const [ticker, setTicker] = useState('')
  const [window, setWindow_] = useState('')
  const [page, setPage] = useState(0)
  const [data, setData] = useState<PredictionsResp | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setPage(0)
  }, [runName, arm, ticker, window])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    api
      .predictions(runName, {
        arm: arm || undefined,
        ticker: ticker || undefined,
        window: window || undefined,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      })
      .then((d) => {
        if (!cancelled) setData(d)
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [runName, arm, ticker, window, page])

  const total = data?.total ?? 0
  const nPages = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const windows = Array.from({ length: nWindows }, (_, i) => i + 1)

  return (
    <section className="panel">
      <h2>Tra cứu dự đoán</h2>
      <div className="filters">
        <label>
          Nhánh{' '}
          <select value={arm} onChange={(e) => setArm(e.target.value)}>
            <option value="">Tất cả</option>
            {arms.map((a) => (
              <option key={a} value={a}>
                {armLabel(a)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Mã{' '}
          <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
            <option value="">Tất cả</option>
            {tickers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <label>
          Cửa sổ{' '}
          <select value={window} onChange={(e) => setWindow_(e.target.value)}>
            <option value="">Tất cả</option>
            {windows.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
        </label>
        <span className="filters-meta">
          {loading ? 'Đang tải…' : `${fmtInt(total)} dòng`}
        </span>
      </div>

      {error && <div className="alert alert-error">Lỗi tải dự đoán: {error}</div>}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Mã</th>
              <th>Ngày quan sát</th>
              <th>Ngày mục tiêu</th>
              <th>Cửa sổ</th>
              <th>Seed</th>
              <th>Nhánh</th>
              <th>Thực tế</th>
              <th>Dự đoán</th>
              <th>Có tin</th>
            </tr>
          </thead>
          <tbody>
            {(data?.rows ?? []).map((r, i) => (
              <tr key={`${r.ticker}-${r.observation_date}-${r.arm}-${r.seed}-${i}`}>
                <td>{r.ticker}</td>
                <td>{r.observation_date}</td>
                <td>{r.target_date}</td>
                <td>{r.window}</td>
                <td>{r.seed}</td>
                <td>{armLabel(r.arm)}</td>
                <td>{r.y_true}</td>
                <td className={r.y_pred === r.y_true ? 'pos' : 'neg'}>{r.y_pred}</td>
                <td>{r.has_news ? 'Có' : '—'}</td>
              </tr>
            ))}
            {data && data.rows.length === 0 && (
              <tr>
                <td colSpan={9} className="empty-cell">
                  Không có dòng nào khớp bộ lọc.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="pager">
        <button disabled={page === 0 || loading} onClick={() => setPage((p) => p - 1)}>
          ← Trước
        </button>
        <span>
          Trang {page + 1} / {nPages}
        </span>
        <button
          disabled={page + 1 >= nPages || loading}
          onClick={() => setPage((p) => p + 1)}
        >
          Sau →
        </button>
      </div>
    </section>
  )
}
