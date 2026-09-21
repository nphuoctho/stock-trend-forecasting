import { HugeiconsIcon, type IconSvgElement } from '@hugeicons/react'
import {
  Calendar03Icon,
  ChartLineData01Icon,
  DashboardSpeed02Icon,
  DatabaseIcon,
  NewsIcon,
} from '@hugeicons/core-free-icons'
import { RunInfo, RunSummary } from '../api'
import { fmt, fmtInt, fmtPct } from '../format'
import { Card, CardContent } from '@/components/ui/card'

export default function OverviewCards({
  summary,
  run,
}: {
  summary: RunSummary
  run: RunInfo | null
}) {
  const prov = summary.provenance
  const coverage = prov?.news_coverage
  const dateStart = prov?.date_start ?? run?.date_start
  const dateEnd = prov?.date_end ?? run?.date_end
  const tickers = prov?.tickers ?? summary.panel_tickers ?? []

  const cards: {
    label: string
    value: string
    hint?: string
    icon: IconSvgElement
  }[] = [
    {
      label: 'Số dòng panel',
      value: fmtInt(summary.panel_rows),
      hint: 'quan sát mã × ngày',
      icon: DatabaseIcon,
    },
    {
      label: 'Số mã cổ phiếu',
      value: fmtInt(tickers.length),
      hint: tickers.slice(0, 6).join(', ') + (tickers.length > 6 ? ', …' : ''),
      icon: ChartLineData01Icon,
    },
    {
      label: 'Khoảng dữ liệu',
      value: dateStart && dateEnd ? `${dateStart} → ${dateEnd}` : '—',
      hint: prov?.timezone,
      icon: Calendar03Icon,
    },
    {
      label: 'Mức ngẫu nhiên (chance)',
      value: fmt(summary.chance_level),
      hint: 'accuracy của dự đoán đồng đều',
      icon: DashboardSpeed02Icon,
    },
    {
      label: 'Độ phủ tin tức',
      value: fmtPct(coverage?.fraction),
      hint: coverage
        ? `${fmtInt(coverage.rows_with_news)} / ${fmtInt(coverage.panel_rows)} dòng có tin`
        : undefined,
      icon: NewsIcon,
    },
  ]

  return (
    <section className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
      {cards.map((c, i) => (
        <Card
          key={c.label}
          className="gap-2 py-4 animate-in fade-in slide-in-from-bottom-2 duration-500"
          style={{ animationDelay: `${i * 60}ms`, animationFillMode: 'backwards' }}
        >
          <CardContent className="flex flex-col gap-1.5 px-4">
            <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              <HugeiconsIcon icon={c.icon} className="size-3.5 text-primary" />
              {c.label}
            </div>
            <div className="text-2xl font-semibold tabular-nums">{c.value}</div>
            {c.hint && <div className="text-xs text-muted-foreground">{c.hint}</div>}
          </CardContent>
        </Card>
      ))}
    </section>
  )
}
