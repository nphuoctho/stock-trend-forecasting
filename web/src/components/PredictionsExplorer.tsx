import { useEffect, useState } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { HugeiconsIcon } from '@hugeicons/react'
import {
  AlertCircleIcon,
  ArrowLeft01Icon,
  ArrowRight01Icon,
  FilterIcon,
  Loading03Icon,
} from '@hugeicons/core-free-icons'
import { api } from '../api'
import { armLabel, fmtInt } from '../format'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
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

const PAGE_SIZE = 100
const ALL = '__all__'

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
  const [arm, setArm] = useState(ALL)
  const [ticker, setTicker] = useState(ALL)
  const [window, setWindow_] = useState(ALL)
  const [page, setPage] = useState(0)

  useEffect(() => {
    setPage(0)
  }, [runName, arm, ticker, window])

  const query = useQuery({
    queryKey: ['predictions', runName, arm, ticker, window, page],
    placeholderData: keepPreviousData,
    queryFn: () =>
      api.predictions(runName, {
        arm: arm === ALL ? undefined : arm,
        ticker: ticker === ALL ? undefined : ticker,
        window: window === ALL ? undefined : window,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
  })

  const data = query.data
  const total = data?.total ?? 0
  const nPages = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const windows = Array.from({ length: nWindows }, (_, i) => i + 1)

  return (
    <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HugeiconsIcon icon={FilterIcon} className="size-4 text-primary" />
          Tra cứu dự đoán
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Nhánh</span>
            <Select value={arm} onValueChange={setArm}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL}>Tất cả</SelectItem>
                {arms.map((a) => (
                  <SelectItem key={a} value={a}>
                    {armLabel(a)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Mã</span>
            <Select value={ticker} onValueChange={setTicker}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL}>Tất cả</SelectItem>
                {tickers.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Cửa sổ</span>
            <Select value={window} onValueChange={setWindow_}>
              <SelectTrigger className="w-28">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL}>Tất cả</SelectItem>
                {windows.map((w) => (
                  <SelectItem key={w} value={String(w)}>
                    {w}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <span className="flex items-center gap-1.5 pb-2 text-sm text-muted-foreground">
            {query.isFetching && (
              <HugeiconsIcon icon={Loading03Icon} className="size-3.5 animate-spin" />
            )}
            {fmtInt(total)} dòng
          </span>
        </div>

        {query.isError && (
          <div className="flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <HugeiconsIcon icon={AlertCircleIcon} className="size-4 shrink-0" />
            Lỗi tải dự đoán: {query.error.message}
          </div>
        )}

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Mã</TableHead>
              <TableHead>Ngày quan sát</TableHead>
              <TableHead>Ngày mục tiêu</TableHead>
              <TableHead>Cửa sổ</TableHead>
              <TableHead>Seed</TableHead>
              <TableHead>Nhánh</TableHead>
              <TableHead>Thực tế</TableHead>
              <TableHead>Dự đoán</TableHead>
              <TableHead>Có tin</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {(data?.rows ?? []).map((r, i) => (
              <TableRow key={`${r.ticker}-${r.observation_date}-${r.arm}-${r.seed}-${i}`}>
                <TableCell className="font-medium">{r.ticker}</TableCell>
                <TableCell className="tabular-nums">{r.observation_date}</TableCell>
                <TableCell className="tabular-nums">{r.target_date}</TableCell>
                <TableCell className="tabular-nums">{r.window}</TableCell>
                <TableCell className="tabular-nums">{r.seed}</TableCell>
                <TableCell>{armLabel(r.arm)}</TableCell>
                <TableCell>{r.y_true}</TableCell>
                <TableCell
                  className={cn(
                    'font-medium',
                    r.y_pred === r.y_true ? 'text-primary' : 'text-destructive',
                  )}
                >
                  {r.y_pred}
                </TableCell>
                <TableCell>{r.has_news ? 'Có' : '—'}</TableCell>
              </TableRow>
            ))}
            {data && data.rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={9} className="py-8 text-center text-muted-foreground">
                  Không có dòng nào khớp bộ lọc.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>

        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            size="sm"
            disabled={page === 0 || query.isFetching}
            onClick={() => setPage((p) => p - 1)}
          >
            <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
            Trước
          </Button>
          <span className="text-sm text-muted-foreground tabular-nums">
            Trang {page + 1} / {nPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page + 1 >= nPages || query.isFetching}
            onClick={() => setPage((p) => p + 1)}
          >
            Sau
            <HugeiconsIcon icon={ArrowRight01Icon} className="size-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
