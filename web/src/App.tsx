import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { HugeiconsIcon } from '@hugeicons/react'
import {
  AlertCircleIcon,
  ChartLineData01Icon,
  Loading03Icon,
} from '@hugeicons/core-free-icons'
import { api, RunInfo } from './api'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import TodaySection from './components/TodaySection'
import LiveSignalsSection from './components/LiveSignalsSection'
import OverviewCards from './components/OverviewCards'
import MetricsTable from './components/MetricsTable'
import InformationGainSection from './components/InformationGainSection'
import StratifiedSection from './components/StratifiedSection'
import PredictionsExplorer from './components/PredictionsExplorer'
import ProvenanceFooter from './components/ProvenanceFooter'

function ModeBadge({ mode }: { mode: RunInfo['mode'] }) {
  if (mode === 'real') return <Badge variant="real">tin tức thật</Badge>
  if (mode === 'neutral_prior') return <Badge variant="neutral">neutral prior</Badge>
  return <Badge variant="muted">không tin tức</Badge>
}

export default function App() {
  const [selected, setSelected] = useState<string | null>(null)

  const runsQuery = useQuery({ queryKey: ['runs'], queryFn: api.runs })
  const runs = runsQuery.data?.runs

  useEffect(() => {
    if (selected || !runs?.length) return
    const preferred =
      runs.find((r) => r.has_information_gain && r.mode === 'real') ??
      runs.find((r) => r.has_information_gain) ??
      runs[0]
    setSelected(preferred.name)
  }, [runs, selected])

  const detailQuery = useQuery({
    queryKey: ['run-detail', selected],
    enabled: !!selected,
    queryFn: async () => {
      const [summary, metrics, stratified, ig] = await Promise.all([
        api.summary(selected!),
        api.metrics(selected!),
        api.stratified(selected!),
        api.informationGain(selected!),
      ])
      return { summary, metrics, stratified, ig }
    },
  })

  const summary = detailQuery.data?.summary ?? null
  const currentRun = runs?.find((r) => r.name === selected) ?? null

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4 animate-in fade-in slide-in-from-top-2 duration-500">
        <div className="flex items-start gap-3">
          <div className="mt-1 flex size-10 items-center justify-center rounded-lg bg-primary/15 text-primary">
            <HugeiconsIcon icon={ChartLineData01Icon} className="size-6" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">
              Dự báo xu hướng cổ phiếu
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Bảng kết quả thực nghiệm — so sánh mô hình chỉ dùng giá và mô hình hai
              nhánh kết hợp tin tức
            </p>
          </div>
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Lần chạy
          </span>
          <div className="flex items-center gap-2">
            <Select
              value={selected ?? ''}
              onValueChange={setSelected}
              disabled={!runs || runs.length === 0}
            >
              <SelectTrigger className="w-64">
                <SelectValue placeholder="Đang tải…" />
              </SelectTrigger>
              <SelectContent>
                {runs?.map((r) => (
                  <SelectItem key={r.name} value={r.name}>
                    {r.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {currentRun && <ModeBadge mode={currentRun.mode} />}
          </div>
        </div>
      </header>

      {runsQuery.isError && (
        <div className="mb-6 flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive animate-in fade-in">
          <HugeiconsIcon icon={AlertCircleIcon} className="size-4 shrink-0" />
          Không tải được danh sách lần chạy: {runsQuery.error.message}. Kiểm tra API tại{' '}
          <code className="font-mono">/api/runs</code>.
        </div>
      )}
      {runs && runs.length === 0 && (
        <div className="mb-6 rounded-lg border border-border bg-card px-4 py-3 text-sm text-muted-foreground">
          Chưa có lần chạy nào trong thư mục outputs.
        </div>
      )}

      {detailQuery.isError && (
        <div className="mb-6 flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive animate-in fade-in">
          <HugeiconsIcon icon={AlertCircleIcon} className="size-4 shrink-0" />
          Lỗi tải dữ liệu: {detailQuery.error.message}
        </div>
      )}
      {detailQuery.isLoading && (
        <div className="space-y-4" aria-busy="true">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <HugeiconsIcon icon={Loading03Icon} className="size-4 animate-spin" />
            Đang tải dữ liệu lần chạy…
          </div>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-24" />
            ))}
          </div>
          <Skeleton className="h-64" />
        </div>
      )}

      <TodaySection />

      <LiveSignalsSection />

      {summary && !detailQuery.isLoading && (
        <main className="space-y-6">
          <OverviewCards summary={summary} run={currentRun} />
          <MetricsTable
            summary={summary}
            metricRows={detailQuery.data?.metrics.rows ?? []}
          />
          {detailQuery.data?.ig && (
            <InformationGainSection ig={detailQuery.data.ig} />
          )}
          <StratifiedSection
            stratifiedByNews={summary.stratified_by_news}
            rows={detailQuery.data?.stratified.rows ?? []}
          />
          <PredictionsExplorer
            runName={summary.name}
            arms={Object.keys(summary.summary ?? {})}
            tickers={summary.panel_tickers ?? []}
            nWindows={
              typeof summary.config?.n_windows === 'number'
                ? (summary.config.n_windows as number)
                : (detailQuery.data?.stratified.rows ?? []).reduce(
                    (m, r) => Math.max(m, r.window),
                    0,
                  )
            }
          />
          <ProvenanceFooter
            provenance={summary.provenance}
            environment={summary.environment}
            runName={summary.name}
          />
        </main>
      )}
    </div>
  )
}
