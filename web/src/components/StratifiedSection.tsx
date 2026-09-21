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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { cn } from '@/lib/utils'

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
    <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500">
      <CardHeader className="flex-row items-start justify-between gap-4 space-y-0">
        <div className="space-y-1">
          <CardTitle>Phân tầng theo tin tức</CardTitle>
          <CardDescription>
            Hiệu năng trên các quan sát có tin tức trong ngày so với ngày không tin.
          </CardDescription>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Chỉ số</span>
          <Select value={metric} onValueChange={(v) => setMetric(v as StratMetric)}>
            <SelectTrigger className="w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {STRATIFIED_METRICS.map((m) => (
                <SelectItem key={m} value={m}>
                  {metricLabel(m)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {chartData.length > 0 && (
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
                contentStyle={{
                  background: '#141b2d',
                  border: '1px solid #26314a',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: '#e6ebf5' }}
                formatter={(v) => (typeof v === 'number' ? v.toFixed(4) : v)}
              />
              <Legend wrapperStyle={{ color: '#8b98b8' }} />
              <Bar dataKey="Có tin tức" fill="#2dd4bf" />
              <Bar dataKey="Không tin tức" fill="#64748b" />
            </BarChart>
          </ResponsiveContainer>
        )}

        {windowRows.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Nhánh</TableHead>
                <TableHead>Cửa sổ</TableHead>
                <TableHead>n có tin</TableHead>
                <TableHead>n không tin</TableHead>
                <TableHead>{metricLabel(metric)} — có tin</TableHead>
                <TableHead>{metricLabel(metric)} — không tin</TableHead>
                <TableHead>Δ</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {windowRows.map((r) => {
                const delta =
                  r.hasMean !== null && r.noMean !== null ? r.hasMean - r.noMean : null
                return (
                  <TableRow key={`${r.arm}-${r.window}`}>
                    <TableCell>{armLabel(r.arm)}</TableCell>
                    <TableCell className="tabular-nums">{r.window}</TableCell>
                    <TableCell className="tabular-nums">{fmtInt(r.hasN)}</TableCell>
                    <TableCell className="tabular-nums">{fmtInt(r.noN)}</TableCell>
                    <TableCell className="tabular-nums">{fmt(r.hasMean)}</TableCell>
                    <TableCell className="tabular-nums">{fmt(r.noMean)}</TableCell>
                    <TableCell
                      className={cn(
                        'tabular-nums',
                        delta !== null &&
                          (delta > 0 ? 'text-primary' : delta < 0 ? 'text-destructive' : ''),
                      )}
                    >
                      {fmtSigned(delta)}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
