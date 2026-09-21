export interface RunInfo {
  name: string
  panel_rows: number | null
  mode: 'real' | 'neutral_prior' | null
  date_start: string | null
  date_end: string | null
  has_information_gain: boolean
}

export interface MeanStd {
  mean: number | null
  std: number | null
  n: number
}

export type ArmSummary = Record<string, MeanStd>
export type Summary = Record<string, ArmSummary>

export interface BootstrapCI {
  mean: number | null
  low: number | null
  high: number | null
  n_blocks?: number
}

export interface AblationMetric {
  per_window: number[]
  window_bootstrap: BootstrapCI
  date_block_bootstrap: BootstrapCI
}

export type Ablation = Record<string, Record<string, AblationMetric>>

export interface StratumStat {
  n_mean: number | null
  macro_f1_mean: number | null
  balanced_accuracy_mean: number | null
  accuracy_mean: number | null
}

export type StratifiedByNews = Record<
  string,
  { has_news?: StratumStat; no_news?: StratumStat }
>

export interface NewsSentimentProv {
  mode?: 'real' | 'neutral_prior' | null
  source_path?: string
  source_hash?: string
  feature_hash?: string
  path?: string
  hash?: string
  rows?: number
}

export interface Provenance {
  tickers?: string[]
  date_start?: string
  date_end?: string
  session_cutoff?: string
  timezone?: string
  prices_hash?: string
  news_sentiment?: NewsSentimentProv
  alignment_report?: Record<string, number>
  panel_hash?: string
  news_coverage?: {
    panel_rows?: number
    rows_with_news?: number
    fraction?: number
  }
}

export interface RunSummary {
  name: string
  config: Record<string, unknown> | null
  chance_level: number | null
  panel_rows: number | null
  panel_tickers: string[] | null
  summary: Summary | null
  ablation: Ablation | null
  stratified_by_news: StratifiedByNews | null
  environment: Record<string, unknown> | null
  provenance: Provenance | null
}

export interface MetricsRow {
  arm: string
  [key: string]: string | number | null
}

export interface Prediction {
  ticker: string
  observation_date: string
  target_date: string
  window: number
  seed: number
  arm: string
  y_true: string
  y_pred: string
  has_news: number
}

export interface PredictionsResp {
  total: number
  rows: Prediction[]
}

export interface StratifiedRow {
  window: number
  seed: number
  arm: string
  stratum: string
  n: number
  macro_f1: number | null
  balanced_accuracy: number | null
  accuracy: number | null
}

export interface IgEffect {
  price_only: number | null
  two_branch_neutral_prior: number | null
  two_branch_real_sentiment: number | null
  architecture_effect?: number | null
  architecture_and_news_presence_volume_effect?: number | null
  information_gain: number | null
  naive_delta: number | null
  information_gain_per_window: number[]
  information_gain_window_bootstrap: BootstrapCI
  information_gain_date_bootstrap: BootstrapCI
}

export interface InformationGain {
  arm: string
  price_arm: string
  config: Record<string, unknown> | null
  effects: Record<string, IgEffect>
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`)
  return (await res.json()) as T
}

async function getOr404<T>(path: string): Promise<T | null> {
  const res = await fetch(path)
  if (res.status === 404) return null
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`)
  return (await res.json()) as T
}

export const api = {
  runs: () => get<{ runs: RunInfo[] }>('/api/runs'),
  summary: (name: string) => get<RunSummary>(`/api/runs/${name}/summary`),
  metrics: (name: string) => get<{ rows: MetricsRow[] }>(`/api/runs/${name}/metrics`),
  stratified: (name: string) =>
    get<{ rows: StratifiedRow[] }>(`/api/runs/${name}/stratified`),
  informationGain: (name: string) =>
    getOr404<InformationGain>(`/api/runs/${name}/information-gain`),
  predictions: (
    name: string,
    params: { arm?: string; ticker?: string; window?: string; limit: number; offset: number },
  ) => {
    const q = new URLSearchParams()
    if (params.arm) q.set('arm', params.arm)
    if (params.ticker) q.set('ticker', params.ticker)
    if (params.window) q.set('window', params.window)
    q.set('limit', String(params.limit))
    q.set('offset', String(params.offset))
    return get<PredictionsResp>(`/api/runs/${name}/predictions?${q.toString()}`)
  },
}
