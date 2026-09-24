import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { HugeiconsIcon } from '@hugeicons/react'
import {
  ArrowDown01Icon,
  ArrowUp01Icon,
  ChartLineData01Icon,
  MinusSignIcon,
  NewsIcon,
} from '@hugeicons/core-free-icons'
import { api, LiveTodayTicker } from '../api'
import { fmtPct } from '../format'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

const DIRECTION = {
  UP: { label: 'TĂNG', icon: ArrowUp01Icon, className: 'text-primary' },
  DOWN: { label: 'GIẢM', icon: ArrowDown01Icon, className: 'text-destructive' },
  FLAT: { label: 'ĐI NGANG', icon: MinusSignIcon, className: 'text-muted-foreground' },
} as const

function sentimentLabel(n: LiveTodayTicker['news'][number]) {
  const best = Math.max(n.prob_negative, n.prob_neutral, n.prob_positive)
  if (best === n.prob_positive) return { text: 'tích cực', cls: 'text-primary' }
  if (best === n.prob_negative) return { text: 'tiêu cực', cls: 'text-destructive' }
  return { text: 'trung lập', cls: 'text-muted-foreground' }
}

function TickerCard({ t }: { t: LiveTodayTicker }) {
  const [open, setOpen] = useState(false)
  const dir = DIRECTION[t.y_pred as keyof typeof DIRECTION] ?? DIRECTION.FLAT
  return (
    <div className="rounded-lg border border-border p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-base font-bold">{t.ticker}</span>
          <HugeiconsIcon icon={dir.icon} className={cn('size-5', dir.className)} />
          <span className={cn('text-sm font-semibold', dir.className)}>{dir.label}</span>
        </div>
        <span className="text-xs tabular-nums text-muted-foreground">
          tin cậy {fmtPct(t.confidence)}
        </span>
      </div>

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {t.last_close !== null && (
          <span>
            đóng cửa <span className="font-medium text-foreground">{t.last_close.toFixed(2)}</span>
          </span>
        )}
        {t.flat_band && (
          <span>
            ngưỡng đi ngang{' '}
            <span className="font-medium text-foreground">
              {t.flat_band.low.toFixed(2)} – {t.flat_band.high.toFixed(2)}
            </span>
          </span>
        )}
        <span>
          xác suất{' '}
          <span className="text-primary">T {fmtPct(t.prob_up)}</span> ·{' '}
          <span className="text-muted-foreground">N {fmtPct(t.prob_flat)}</span> ·{' '}
          <span className="text-destructive">G {fmtPct(t.prob_down)}</span>
        </span>
      </div>

      {t.news.length > 0 ? (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            <HugeiconsIcon icon={NewsIcon} className="size-3.5" />
            {t.news.length} tin liên quan {open ? '▴' : '▾'}
          </button>
          {open && (
            <ul className="mt-2 space-y-1.5">
              {t.news.map((n) => {
                const s = sentimentLabel(n)
                return (
                  <li key={n.url} className="text-xs">
                    <a
                      href={n.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-foreground underline-offset-2 hover:underline"
                    >
                      {n.title ?? n.url}
                    </a>
                    <span className={cn('ml-1.5', s.cls)}>({s.text})</span>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      ) : (
        <p className="mt-3 text-xs text-muted-foreground">
          không có tin mới trong cửa sổ 5 phiên — dự đoán chủ yếu từ giá
        </p>
      )}
    </div>
  )
}

export default function TodaySection() {
  const today = useQuery({ queryKey: ['live-today'], queryFn: api.liveToday })

  if (today.isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Dự đoán phiên tới</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-40 w-full" />
        </CardContent>
      </Card>
    )
  }
  if (!today.data) return null

  const d = today.data
  return (
    <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HugeiconsIcon icon={ChartLineData01Icon} className="size-4 text-primary" />
          Dự đoán phiên tới
          <Badge variant="muted" className="text-xs font-normal">
            quan sát {d.observation_date}
          </Badge>
        </CardTitle>
        <CardDescription>
          Mô hình LSTM kết hợp giá và cảm xúc tin tức, dự đoán hướng phiên kế tiếp
          (3 lớp). "Ngưỡng đi ngang" là ranh giới phân lớp học từ dữ liệu quá khứ —
          mô hình không dự đoán biên độ cụ thể. Tin liên quan là các bài trong cửa
          sổ 5 phiên mà mô hình đọc, không hẳn là nguyên nhân của dự đoán. Không
          phải khuyến nghị giao dịch.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-2">
          {d.tickers.map((t) => (
            <TickerCard key={t.ticker} t={t} />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
