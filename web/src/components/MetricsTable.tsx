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
    <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500">
      <CardHeader>
        <CardTitle>Kết quả theo nhánh mô hình</CardTitle>
        <CardDescription>
          Trung bình ± độ lệch chuẩn trên các cửa sổ walk-forward × seed. Hàng tô sáng
          là nhánh hai luồng có đặc trưng tin tức.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Nhánh</TableHead>
              <TableHead>macro-F1</TableHead>
              <TableHead>balanced accuracy</TableHead>
              <TableHead>accuracy</TableHead>
              <TableHead>macro OvR AUC</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
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
                <TableRow
                  key={arm}
                  className={cn(isSentimentArm(arm) && 'bg-primary/10 hover:bg-primary/15')}
                >
                  <TableCell className="font-medium">
                    {armLabel(arm)}
                    {isSentimentArm(arm) && (
                      <Badge variant="real" className="ml-2">
                        + tin tức
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell title={`CI theo cửa sổ: ${f1ci}`} className="tabular-nums">
                    {fmtMeanStd(s?.macro_f1)}
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {fmtMeanStd(s?.balanced_accuracy)}
                  </TableCell>
                  <TableCell className="tabular-nums">{fmtMeanStd(s?.accuracy)}</TableCell>
                  <TableCell className="tabular-nums">
                    {fmtMeanStd(s?.macro_ovr_auc)}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>

        {ablationPairs.length > 0 && (
          <div className="space-y-3">
            <div>
              <h3 className="text-sm font-semibold">
                Ablation: nhánh tin tức − nhánh chỉ giá
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Chênh lệch ghép cặp theo cửa sổ; khoảng tin cậy bootstrap theo cửa sổ và
                theo khối ngày.
              </p>
            </div>
            {ablationPairs.map((pair) => {
              const { left, right } = prettyPair(pair)
              const metrics = ablation[pair]
              return (
                <Table key={pair}>
                  <TableHeader>
                    <TableRow>
                      <TableHead colSpan={4} className="normal-case">
                        {left} − {right}
                      </TableHead>
                    </TableRow>
                    <TableRow>
                      <TableHead>Chỉ số</TableHead>
                      <TableHead>Δ trung bình (cửa sổ)</TableHead>
                      <TableHead>CI bootstrap cửa sổ</TableHead>
                      <TableHead>CI bootstrap khối ngày</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(metrics).map(([metric, m]) => (
                      <TableRow key={metric}>
                        <TableCell>{metricLabel(metric)}</TableCell>
                        <TableCell
                          className={cn(
                            'tabular-nums',
                            m.window_bootstrap?.mean != null &&
                              (m.window_bootstrap.mean > 0
                                ? 'text-primary'
                                : m.window_bootstrap.mean < 0
                                  ? 'text-destructive'
                                  : ''),
                          )}
                        >
                          {fmtSigned(m.window_bootstrap?.mean)}
                        </TableCell>
                        <TableCell className="tabular-nums">
                          {fmtCI(m.window_bootstrap)}
                        </TableCell>
                        <TableCell className="tabular-nums">
                          {fmtCI(m.date_block_bootstrap)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
