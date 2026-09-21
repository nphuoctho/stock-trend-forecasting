export const ARM_LABELS: Record<string, string> = {
  majority: 'Đa số (majority)',
  random: 'Ngẫu nhiên',
  logreg_price: 'LogReg — giá',
  logreg_price_sentiment: 'LogReg — giá + tin tức',
  lstm_price: 'LSTM — giá',
  lstm_price_sentiment: 'LSTM — giá + tin tức',
}

export const METRIC_LABELS: Record<string, string> = {
  macro_f1: 'macro-F1',
  balanced_accuracy: 'balanced accuracy',
  accuracy: 'accuracy',
  macro_ovr_auc: 'macro OvR AUC',
}

export const METRICS: string[] = ['macro_f1', 'balanced_accuracy', 'accuracy', 'macro_ovr_auc']

export function armLabel(arm: string): string {
  return ARM_LABELS[arm] ?? arm
}

export function metricLabel(metric: string): string {
  return METRIC_LABELS[metric] ?? metric
}

export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return v.toFixed(digits)
}

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return v.toLocaleString('vi-VN')
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

export function fmtSigned(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`
}

export function fmtMeanStd(ms: { mean: number | null; std: number | null } | undefined): string {
  if (!ms || ms.mean === null || ms.mean === undefined) return '—'
  if (ms.std === null || ms.std === undefined) return fmt(ms.mean)
  return `${fmt(ms.mean)} ± ${fmt(ms.std)}`
}

export function fmtCI(ci: { low: number | null; high: number | null } | undefined): string {
  if (!ci || ci.low === null || ci.low === undefined || ci.high === null || ci.high === undefined)
    return '—'
  return `[${fmt(ci.low)}, ${fmt(ci.high)}]`
}

export function shortHash(h: string | undefined | null): string {
  if (!h) return '—'
  return h.length > 16 ? `${h.slice(0, 12)}…` : h
}

export function isSentimentArm(arm: string): boolean {
  return arm.includes('sentiment')
}

export function prettyPair(pair: string): { left: string; right: string } {
  const [left, right] = pair.split('__minus__')
  return { left: armLabel(left ?? pair), right: armLabel(right ?? '') }
}
