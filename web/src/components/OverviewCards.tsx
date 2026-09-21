import { RunInfo, RunSummary } from '../api'
import { fmt, fmtInt, fmtPct } from '../format'

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

  const cards: { label: string; value: string; hint?: string }[] = [
    {
      label: 'Số dòng panel',
      value: fmtInt(summary.panel_rows),
      hint: 'quan sát mã × ngày',
    },
    {
      label: 'Số mã cổ phiếu',
      value: fmtInt(tickers.length),
      hint: tickers.slice(0, 6).join(', ') + (tickers.length > 6 ? ', …' : ''),
    },
    {
      label: 'Khoảng dữ liệu',
      value: dateStart && dateEnd ? `${dateStart} → ${dateEnd}` : '—',
      hint: prov?.timezone,
    },
    {
      label: 'Mức ngẫu nhiên (chance)',
      value: fmt(summary.chance_level),
      hint: 'accuracy của dự đoán đồng đều',
    },
    {
      label: 'Độ phủ tin tức',
      value: fmtPct(coverage?.fraction),
      hint: coverage
        ? `${fmtInt(coverage.rows_with_news)} / ${fmtInt(coverage.panel_rows)} dòng có tin`
        : undefined,
    },
  ]

  return (
    <section className="cards">
      {cards.map((c) => (
        <div className="card" key={c.label}>
          <div className="card-label">{c.label}</div>
          <div className="card-value">{c.value}</div>
          {c.hint && <div className="card-hint">{c.hint}</div>}
        </div>
      ))}
    </section>
  )
}
