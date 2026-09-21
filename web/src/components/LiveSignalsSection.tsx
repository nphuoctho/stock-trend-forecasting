import { useQuery } from '@tanstack/react-query'
import { HugeiconsIcon } from '@hugeicons/react'
import { DashboardSpeed02Icon, NewsIcon } from '@hugeicons/core-free-icons'
import {
  Line,
  LineChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, LiveArmHistory, LiveArmLatest } from '../api'
import { armLabel, fmt, fmtPct } from '../format'
import { Badge } from '@/components/ui/badge'
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

const LABEL_VARIANT: Record<string, 'real' | 'destructive' | 'muted'> = {
  UP: 'real',
  DOWN: 'destructive',
  FLAT: 'muted',
}

function LatestTable({ arm }: { arm: LiveArmLatest }) {
  return (
    <div>
      <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold">
        {armLabel(arm.arm)}
        <span className="text-xs font-normal text-muted-foreground">
          phiên {arm.observation_date}
        </span>
      </h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Mã</TableHead>
            <TableHead>Dự đoán</TableHead>
            <TableHead>P(UP)</TableHead>
            <TableHead>P(FLAT)</TableHead>
            <TableHead>P(DOWN)</TableHead>
            <TableHead>Có tin</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {arm.rows.map((r) => (
            <TableRow key={r.ticker}>
              <TableCell className="font-medium">{r.ticker}</TableCell>
              <TableCell>
                <Badge variant={LABEL_VARIANT[r.y_pred] ?? 'muted'}>{r.y_pred}</Badge>
              </TableCell>
              <TableCell className="tabular-nums">{fmt(r.prob_up)}</TableCell>
              <TableCell className="tabular-nums">{fmt(r.prob_flat)}</TableCell>
              <TableCell className="tabular-nums">{fmt(r.prob_down)}</TableCell>
              <TableCell>
                {r.has_news ? (
                  <HugeiconsIcon icon={NewsIcon} className="size-4 text-primary" />
                ) : (
                  '—'
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function HistoryChart({ arms }: { arms: LiveArmHistory[] }) {
  const dates = new Set<string>()
  for (const a of arms) for (const d of a.by_date) dates.add(d.target_date)
  if (dates.size === 0) return null
  const data = [...dates].sort().map((date) => {
    const row: Record<string, string | number> = { date }
    for (const a of arms) {
      const hit = a.by_date.find((d) => d.target_date === date)
      if (hit) row[armLabel(a.arm)] = hit.accuracy
    }
    return row
  })
  const colors = ['#2dd4bf', '#818cf8', '#f472b6', '#d97706']
  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold">Accuracy theo ngày mục tiêu</h3>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#26314a" />
          <XAxis dataKey="date" stroke="#8b98b8" tick={{ fill: '#8b98b8', fontSize: 11 }} />
          <YAxis
            stroke="#8b98b8"
            tick={{ fill: '#8b98b8' }}
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <Tooltip
            contentStyle={{
              background: '#141b2d',
              border: '1px solid #26314a',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#e6ebf5' }}
            formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : v)}
          />
          <Legend wrapperStyle={{ color: '#8b98b8' }} />
          {arms.map((a, i) => (
            <Line
              key={a.arm}
              type="monotone"
              dataKey={armLabel(a.arm)}
              stroke={colors[i % colors.length]}
              dot={{ r: 3 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function LiveSignalsSection() {
  const latest = useQuery({ queryKey: ['live-latest'], queryFn: api.liveLatest })
  const history = useQuery({ queryKey: ['live-history'], queryFn: api.liveHistory })

  if (!latest.data && !history.data) return null

  return (
    <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HugeiconsIcon icon={DashboardSpeed02Icon} className="size-4 text-primary" />
          Tín hiệu phiên mới nhất
        </CardTitle>
        <CardDescription>
          Dự đoán xu hướng phiên kế tiếp từ các mô hình đã refit trên toàn bộ dữ liệu.
          Đây là demo pipeline end-to-end — không phải khuyến nghị giao dịch.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {latest.data?.arms.map((a) => <LatestTable key={a.arm} arm={a} />)}

        {history.data && history.data.arms.length > 0 && (
          <div className="space-y-3 border-t border-border pt-4">
            <h3 className="text-sm font-semibold">Theo dõi trực tiếp</h3>
            <div className="flex flex-wrap gap-3">
              {history.data.arms.map((a) => (
                <div
                  key={a.arm}
                  className="rounded-md border border-border bg-muted px-3 py-2 text-sm"
                >
                  <span className="font-medium">{armLabel(a.arm)}</span>
                  <span className="ml-2 text-muted-foreground">
                    {a.resolved} đã đối chiếu · {a.pending} chờ
                  </span>
                  {a.accuracy !== null && (
                    <span
                      className={cn(
                        'ml-2 font-medium tabular-nums',
                        a.accuracy >= 1 / 3 ? 'text-primary' : 'text-destructive',
                      )}
                    >
                      acc {fmtPct(a.accuracy)}
                    </span>
                  )}
                </div>
              ))}
            </div>
            <HistoryChart arms={history.data.arms} />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
