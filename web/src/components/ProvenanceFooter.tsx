import { Provenance } from '../api'
import { fmtInt, shortHash } from '../format'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

export default function ProvenanceFooter({
  provenance,
  environment,
  runName,
}: {
  provenance: Provenance | null
  environment: Record<string, unknown> | null
  runName: string
}) {
  if (!provenance && !environment) return null

  const ns = provenance?.news_sentiment
  const align = provenance?.alignment_report
  const alignEntries = align ? Object.entries(align) : []

  return (
    <Card className="animate-in fade-in duration-500">
      <CardHeader>
        <CardTitle className="text-sm">Nguồn dữ liệu &amp; môi trường</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-1 gap-x-8 gap-y-2 text-sm sm:grid-cols-2 lg:grid-cols-3">
          <div className="flex justify-between gap-4">
            <dt className="text-muted-foreground">Lần chạy</dt>
            <dd className="font-mono text-xs">{runName}</dd>
          </div>
          {provenance?.date_start && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Khoảng dữ liệu</dt>
              <dd className="tabular-nums">
                {provenance.date_start} → {provenance.date_end}
              </dd>
            </div>
          )}
          {provenance?.session_cutoff && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Giờ chốt phiên</dt>
              <dd>{provenance.session_cutoff}</dd>
            </div>
          )}
          {ns?.mode && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Chế độ tin tức</dt>
              <dd>{ns.mode}</dd>
            </div>
          )}
          {ns?.source_path && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Nguồn tin tức</dt>
              <dd className="font-mono text-xs" title={ns.source_path}>
                {ns.source_path}
              </dd>
            </div>
          )}
          {ns?.source_hash && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Hash nguồn tin</dt>
              <dd className="font-mono text-xs">{shortHash(ns.source_hash)}</dd>
            </div>
          )}
          {ns?.feature_hash && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Hash đặc trưng</dt>
              <dd className="font-mono text-xs">{shortHash(ns.feature_hash)}</dd>
            </div>
          )}
          {ns?.rows != null && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Số dòng tin tức</dt>
              <dd className="tabular-nums">{fmtInt(ns.rows)}</dd>
            </div>
          )}
          {provenance?.prices_hash && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Hash dữ liệu giá</dt>
              <dd className="font-mono text-xs">{shortHash(provenance.prices_hash)}</dd>
            </div>
          )}
          {provenance?.panel_hash && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Hash panel</dt>
              <dd className="font-mono text-xs">{shortHash(provenance.panel_hash)}</dd>
            </div>
          )}
          {environment && (
            <div className="flex justify-between gap-4">
              <dt className="text-muted-foreground">Môi trường</dt>
              <dd className="font-mono text-xs">
                py {String(environment.python ?? '?')} · numpy{' '}
                {String(environment.numpy ?? '?')} · pandas{' '}
                {String(environment.pandas ?? '?')}
              </dd>
            </div>
          )}
        </dl>
        {alignEntries.length > 0 && (
          <div className="mt-4 border-t border-border pt-3">
            <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Báo cáo canh chỉnh phiên
            </div>
            <div className="flex flex-wrap gap-2">
              {alignEntries.map(([k, v]) => (
                <span
                  key={k}
                  className={cn(
                    'inline-flex items-center gap-1.5 rounded-md border border-border bg-muted px-2 py-1 text-xs',
                    v === 0 && 'text-muted-foreground',
                  )}
                >
                  {k}: <strong className="tabular-nums">{fmtInt(v)}</strong>
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
