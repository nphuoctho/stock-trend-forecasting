import { MetricsRow, RunSummary } from '../api'
import {
  armLabel,
  fmtCI,
  fmtMeanStd,
  fmtSigned,
  isSentimentArm,
  metricLabel,
  prettyPair,
} from '../format'

const ARM_ORDER = [
  'majority',
  'random',
  'logreg_price',
  'logreg_price_sentiment',
  'lstm_price',
  'lstm_price_sentiment',
]

function num(v: string | number | null | undefined): number | null {
  if (v === null || v === undefined || v === '') return null
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isNaN(n) ? null : n
}

export default function MetricsTable({
  summary,
  metricRows,
}: {
  summary: RunSummary
  metricRows: MetricsRow[]
}) {
  const arms = Object.keys(summary.summary ?? {}).sort(
    (a, b) => ARM_ORDER.indexOf(a) - ARM_ORDER.indexOf(b),
  )
  const byArm = new Map(metricRows.map((r) => [r.arm, r]))
  const ablation = summary.ablation ?? {}
  const ablationPairs = Object.keys(ablation)

  return (
    <section className="panel">
      <h2>Kết quả theo nhánh mô hình</h2>
      <p className="panel-note">
        Trung bình ± độ lệch chuẩn trên các cửa sổ walk-forward × seed. Hàng tô sáng là
        nhánh hai luồng có đặc trưng tin tức.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Nhánh</th>
              <th>macro-F1</th>
              <th>balanced accuracy</th>
              <th>accuracy</th>
              <th>macro OvR AUC</th>
            </tr>
          </thead>
          <tbody>
            {arms.map((arm) => {
              const s = summary.summary?.[arm]
              const row = byArm.get(arm)
              const f1ci = row
                ? fmtCI({
                    low: num(row.macro_f1_window_ci_low),
                    high: num(row.macro_f1_window_ci_high),
                  })
                : '—'
              return (
                <tr key={arm} className={isSentimentArm(arm) ? 'row-sentiment' : ''}>
                  <td>
                    {armLabel(arm)}
                    {isSentimentArm(arm) && <span className="tag">+ tin tức</span>}
                  </td>
                  <td title={`CI theo cửa sổ: ${f1ci}`}>{fmtMeanStd(s?.macro_f1)}</td>
                  <td>{fmtMeanStd(s?.balanced_accuracy)}</td>
                  <td>{fmtMeanStd(s?.accuracy)}</td>
                  <td>{fmtMeanStd(s?.macro_ovr_auc)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {ablationPairs.length > 0 && (
        <>
          <h3>Ablation: nhánh tin tức − nhánh chỉ giá</h3>
          <p className="panel-note">
            Chênh lệch ghép cặp theo cửa sổ; khoảng tin cậy bootstrap theo cửa sổ và theo
            khối ngày.
          </p>
          {ablationPairs.map((pair) => {
            const { left, right } = prettyPair(pair)
            const metrics = ablation[pair]
            return (
              <div className="table-wrap" key={pair}>
                <table>
                  <thead>
                    <tr>
                      <th colSpan={4}>
                        {left} − {right}
                      </th>
                    </tr>
                    <tr>
                      <th>Chỉ số</th>
                      <th>Δ trung bình (cửa sổ)</th>
                      <th>CI bootstrap cửa sổ</th>
                      <th>CI bootstrap khối ngày</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics).map(([metric, m]) => (
                      <tr key={metric}>
                        <td>{metricLabel(metric)}</td>
                        <td className={deltaClass(m.window_bootstrap?.mean)}>
                          {fmtSigned(m.window_bootstrap?.mean)}
                        </td>
                        <td>{fmtCI(m.window_bootstrap)}</td>
                        <td>{fmtCI(m.date_block_bootstrap)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          })}
        </>
      )}
    </section>
  )
}

function deltaClass(v: number | null | undefined): string {
  if (v === null || v === undefined) return ''
  return v > 0 ? 'pos' : v < 0 ? 'neg' : ''
}
