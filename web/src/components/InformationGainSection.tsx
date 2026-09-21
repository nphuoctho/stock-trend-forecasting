import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { IgEffect, InformationGain } from '../api'
import { armLabel, fmt, fmtCI, fmtSigned, metricLabel } from '../format'

const CHART_COLORS = {
  price_only: '#64748b',
  two_branch_neutral_prior: '#d97706',
  two_branch_real_sentiment: '#2dd4bf',
  macro_f1: '#2dd4bf',
  balanced_accuracy: '#818cf8',
  accuracy: '#f472b6',
}

function archEffect(e: IgEffect): number | null | undefined {
  return e.architecture_and_news_presence_volume_effect ?? e.architecture_effect
}

export default function InformationGainSection({ ig }: { ig: InformationGain }) {
  const metrics = Object.keys(ig.effects)

  const armScores = metrics.map((m) => ({
    metric: metricLabel(m),
    'Chỉ giá': ig.effects[m].price_only,
    'Hai nhánh — neutral prior': ig.effects[m].two_branch_neutral_prior,
    'Hai nhánh — tin tức thật': ig.effects[m].two_branch_real_sentiment,
  }))

  const nWindows = Math.max(
    ...metrics.map((m) => ig.effects[m].information_gain_per_window?.length ?? 0),
    0,
  )
  const perWindow = Array.from({ length: nWindows }, (_, i) => {
    const row: Record<string, number | string> = { window: `W${i + 1}` }
    for (const m of metrics) {
      const v = ig.effects[m].information_gain_per_window?.[i]
      if (v !== undefined && v !== null) row[metricLabel(m)] = v
    }
    return row
  })

  return (
    <section className="panel">
      <h2>Phân tách information gain</h2>
      <p className="panel-note">
        So sánh ghép cặp: <strong>{armLabel(ig.price_arm)}</strong> (chỉ giá) →{' '}
        <strong>{armLabel(ig.arm)}</strong> (hai nhánh). Information gain = mức tăng do
        tin tức thật sau khi tách hiệu ứng kiến trúc/độ phủ tin.
      </p>

      <div className="chart-block">
        <h3>Điểm theo nhánh</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={armScores} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#26314a" />
            <XAxis dataKey="metric" stroke="#8b98b8" tick={{ fill: '#8b98b8' }} />
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
            <Bar dataKey="Chỉ giá" fill={CHART_COLORS.price_only} />
            <Bar dataKey="Hai nhánh — neutral prior" fill={CHART_COLORS.two_branch_neutral_prior} />
            <Bar dataKey="Hai nhánh — tin tức thật" fill={CHART_COLORS.two_branch_real_sentiment} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Chỉ số</th>
              <th>Chỉ giá</th>
              <th>Neutral prior</th>
              <th>Tin tức thật</th>
              <th>Hiệu ứng kiến trúc</th>
              <th>Δ ngây thơ</th>
              <th>Information gain</th>
              <th>CI bootstrap cửa sổ</th>
              <th>CI bootstrap khối ngày</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => {
              const e = ig.effects[m]
              const gain = e.information_gain
              return (
                <tr key={m}>
                  <td>{metricLabel(m)}</td>
                  <td>{fmt(e.price_only)}</td>
                  <td>{fmt(e.two_branch_neutral_prior)}</td>
                  <td>{fmt(e.two_branch_real_sentiment)}</td>
                  <td>{fmtSigned(archEffect(e))}</td>
                  <td>{fmtSigned(e.naive_delta)}</td>
                  <td className={gain !== null && gain !== undefined ? (gain > 0 ? 'pos' : gain < 0 ? 'neg' : '') : ''}>
                    {fmtSigned(gain)}
                  </td>
                  <td>{fmtCI(e.information_gain_window_bootstrap)}</td>
                  <td>{fmtCI(e.information_gain_date_bootstrap)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {perWindow.length > 0 && (
        <div className="chart-block">
          <h3>Information gain theo cửa sổ walk-forward</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perWindow} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#26314a" />
              <XAxis dataKey="window" stroke="#8b98b8" tick={{ fill: '#8b98b8' }} />
              <YAxis
                stroke="#8b98b8"
                tick={{ fill: '#8b98b8' }}
                tickFormatter={(v: number) => v.toFixed(3)}
              />
              <Tooltip
                contentStyle={{ background: '#141b2d', border: '1px solid #26314a' }}
                labelStyle={{ color: '#e6ebf5' }}
                formatter={(v) => (typeof v === 'number' ? v.toFixed(4) : v)}
              />
              <Legend wrapperStyle={{ color: '#8b98b8' }} />
              <ReferenceLine y={0} stroke="#4a5878" />
              {metrics.map((m) => (
                <Bar
                  key={m}
                  dataKey={metricLabel(m)}
                  fill={
                    (CHART_COLORS as Record<string, string>)[m] ?? '#8b98b8'
                  }
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  )
}
