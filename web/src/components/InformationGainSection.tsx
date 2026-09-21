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
import { InformationGain } from '../api'
import { armLabel, fmt, fmtCI, fmtSigned, metricLabel } from '../format'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { cn } from '@/lib/utils'

const CHART_COLORS = {
  price_only: '#64748b',
  two_branch_neutral_prior: '#d97706',
  two_branch_real_sentiment: '#2dd4bf',
  macro_f1: '#2dd4bf',
  balanced_accuracy: '#818cf8',
  accuracy: '#f472b6',
}

const TOOLTIP_STYLE = {
  background: '#141b2d',
  border: '1px solid #26314a',
  borderRadius: '8px',
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
    <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500">
      <CardHeader>
        <CardTitle>Phân tách information gain</CardTitle>
        <CardDescription>
          So sánh ghép cặp: <strong>{armLabel(ig.price_arm)}</strong> (chỉ giá) →{' '}
          <strong>{armLabel(ig.arm)}</strong> (hai nhánh). Information gain = mức tăng
          do tin tức thật sau khi tách hiệu ứng kiến trúc/độ phủ tin.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Điểm theo nhánh</h3>
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
                contentStyle={TOOLTIP_STYLE}
                labelStyle={{ color: '#e6ebf5' }}
                formatter={(v) => (typeof v === 'number' ? v.toFixed(4) : v)}
              />
              <Legend wrapperStyle={{ color: '#8b98b8' }} />
              <Bar dataKey="Chỉ giá" fill={CHART_COLORS.price_only} />
              <Bar
                dataKey="Hai nhánh — neutral prior"
                fill={CHART_COLORS.two_branch_neutral_prior}
              />
              <Bar
                dataKey="Hai nhánh — tin tức thật"
                fill={CHART_COLORS.two_branch_real_sentiment}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Chỉ số</TableHead>
              <TableHead>Chỉ giá</TableHead>
              <TableHead>Neutral prior</TableHead>
              <TableHead>Tin tức thật</TableHead>
              <TableHead>Hiệu ứng kiến trúc</TableHead>
              <TableHead>Δ ngây thơ</TableHead>
              <TableHead>Information gain</TableHead>
              <TableHead>CI bootstrap cửa sổ</TableHead>
              <TableHead>CI bootstrap khối ngày</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {metrics.map((m) => {
              const e = ig.effects[m]
              const gain = e.information_gain
              return (
                <TableRow key={m}>
                  <TableCell>{metricLabel(m)}</TableCell>
                  <TableCell className="tabular-nums">{fmt(e.price_only)}</TableCell>
                  <TableCell className="tabular-nums">
                    {fmt(e.two_branch_neutral_prior)}
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {fmt(e.two_branch_real_sentiment)}
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {fmtSigned(e.architecture_and_news_presence_volume_effect)}
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {fmtSigned(e.naive_delta)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      'tabular-nums font-medium',
                      gain != null &&
                        (gain > 0 ? 'text-primary' : gain < 0 ? 'text-destructive' : ''),
                    )}
                  >
                    {fmtSigned(gain)}
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {fmtCI(e.information_gain_window_bootstrap)}
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {fmtCI(e.information_gain_date_bootstrap)}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>

        {perWindow.length > 0 && (
          <div>
            <h3 className="mb-2 text-sm font-semibold">
              Information gain theo cửa sổ walk-forward
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={perWindow}
                margin={{ top: 8, right: 16, bottom: 4, left: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#26314a" />
                <XAxis dataKey="window" stroke="#8b98b8" tick={{ fill: '#8b98b8' }} />
                <YAxis
                  stroke="#8b98b8"
                  tick={{ fill: '#8b98b8' }}
                  tickFormatter={(v: number) => v.toFixed(3)}
                />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  labelStyle={{ color: '#e6ebf5' }}
                  formatter={(v) => (typeof v === 'number' ? v.toFixed(4) : v)}
                />
                <Legend wrapperStyle={{ color: '#8b98b8' }} />
                <ReferenceLine y={0} stroke="#4a5878" />
                {metrics.map((m) => (
                  <Bar
                    key={m}
                    dataKey={metricLabel(m)}
                    fill={(CHART_COLORS as Record<string, string>)[m] ?? '#8b98b8'}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
