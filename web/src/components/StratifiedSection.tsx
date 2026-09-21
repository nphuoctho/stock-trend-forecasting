import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { StratifiedByNews, StratifiedRow, StratumStat } from '../api'
import { armLabel, fmt, fmtInt, fmtSigned, metricLabel } from '../format'

type StratMetric = 'macro_f1' | 'balanced_accuracy' | 'accuracy'
const STRATIFIED_METRICS: StratMetric[] = ['macro_f1', 'balanced_accuracy', 'accuracy']

export default function StratifiedSection({
  stratifiedByNews,
  rows,
}: {
  stratifiedByNews: StratifiedByNews | null
  rows: StratifiedRow[]
}) {
  const [metric, setMetric] = useState<StratMetric>('macro_f1')
  const metricKey = `${metric}_mean` as keyof StratumStat

  const chartData = useMemo(() => {
    if (!stratifiedByNews) return []
    return Object.entries(stratifiedByNews).map(([arm, strata]) => ({
      arm: armLabel(arm),
      'Có tin tức': strata.has_news?.[metricKey] ?? null,
      'Không tin tức': strata.no_news?.[metricKey] ?? null,
    }))
  }, [stratifiedByNews, metricKey])

  const windowRows = useMemo(() => {
    type Agg = {
      arm: string
      window: number
      hasN: number
      noN: number
      hasSum: number
      hasCnt: number
      noSum: number
      noCnt: number
    }
    const acc: Record<string, Agg> = {}
    for (const r of rows) {
      const key = `${r.arm}|${r.window}`
      const a =
        acc[key] ??
        (acc[key] = {
          arm: r.arm,
          window: r.window,
          hasN: 0,
          noN: 0,
          hasSum: 0,
          hasCnt: 0,
          noSum: 0,
          noCnt: 0,
        })
      const v = r[metric]
      if (r.stratum === 'has_news') {
        a.hasN += r.n
        if (v !== null && v !== undefined) {
          a.hasSum += v
          a.hasCnt += 1
        }
      } else if (r.stratum === 'no_news') {
        a.noN += r.n
        if (v !== null && v !== undefined) {
          a.noSum += v
          a.noCnt += 1
        }
      }
    }
    return Object.values(acc)
      .map((a) => ({
        ...a,
        hasMean: a.hasCnt ? a.hasSum / a.hasCnt : null,
        noMean: a.noCnt ? a.noSum / a.noCnt : null,
      }))
      .sort((a, b) => a.arm.localeCompare(b.arm) || a.window - b.window)
  }, [rows, metric])

  if (!stratifiedByNews && rows.length === 0) return null

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Phân tầng theo tin tức</h2>
        <label className="inline-filter">
          Chỉ số{' '}
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as StratMetric)}
          >
            {STRATIFIED_METRICS.map((m) => (
              <option key={m} value={m}>
                {metricLabel(m)}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="panel-note">
        Hiệu năng trên các quan sát có tin tức trong ngày so với ngày không tin.
      </p>

      {chartData.length > 0 && (
        <div className="chart-block">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#26314a" />
              <XAxis
                dataKey="arm"
                stroke="#8b98b8"
                tick={{ fill: '#8b98b8', fontSize: 12 }}
                interval={0}
                angle={-18}
                textAnchor="end"
                height={64}
              />
              <YAxis
                stroke="#8b98b8"
                tick={{ fill: '#8b98b8' }}
                domain={['auto', 'auto']}
                tickFormatter={(v: number) => v.toFixed(3)}
              />
              <Tooltip
                contentStyle={{ background: '#141b2d', border: '1px solid #26314a' }}
                labelStyle={{ color: '#e6ebf5' }}
                formatter={(v) => (typeof v === 'number' ? v.toFixed(4) : v)}
              />
              <Legend wrapperStyle={{ color: '#8b98b8' }} />
              <Bar dataKey="Có tin tức" fill="#2dd4bf" />
              <Bar dataKey="Không tin tức" fill="#64748b" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {windowRows.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Nhánh</th>
                <th>Cửa sổ</th>
                <th>n có tin</th>
                <th>n không tin</th>
                <th>{metricLabel(metric)} — có tin</th>
                <th>{metricLabel(metric)} — không tin</th>
                <th>Δ</th>
              </tr>
            </thead>
            <tbody>
              {windowRows.map((r) => {
                const delta =
                  r.hasMean !== null && r.noMean !== null ? r.hasMean - r.noMean : null
                return (
                  <tr key={`${r.arm}-${r.window}`}>
                    <td>{armLabel(r.arm)}</td>
                    <td>{r.window}</td>
                    <td>{fmtInt(r.hasN)}</td>
                    <td>{fmtInt(r.noN)}</td>
                    <td>{fmt(r.hasMean)}</td>
                    <td>{fmt(r.noMean)}</td>
                    <td
                      className={
                        delta !== null ? (delta > 0 ? 'pos' : delta < 0 ? 'neg' : '') : ''
                      }
                    >
                      {fmtSigned(delta)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
