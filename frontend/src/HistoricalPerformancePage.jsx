import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle, CalendarDays,
  ChartNoAxesCombined, CircleHelp, FilterX, Newspaper, RefreshCw,
} from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { getHistoricalPerformanceDashboard } from './api.js';

const HORIZONS = ['1d', '3d', '5d'];
const STOCKS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'ITC', 'LT', 'BHARTIARTL', 'TATAMOTORS'];
const SENTIMENTS = ['Positive', 'Negative', 'Neutral'];
const DIRECTIONS = ['Positive', 'Negative', 'Neutral'];
const pct = (value) => typeof value === 'number' && Number.isFinite(value) ? `${value > 0 ? '+' : ''}${value.toFixed(2)}%` : '—';
const count = (value) => Number.isFinite(value) ? value.toLocaleString() : '0';
const alignmentPct = (value) => typeof value === 'number' ? `${value.toFixed(1)}%` : 'Insufficient directional sample';
const signed = (value) => typeof value === 'number' ? (value >= 0 ? 'text-emerald-700' : 'text-rose-700') : 'text-slate-400';
const dateLabel = (value) => new Date(value).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });

function MetricCard({ label, value, detail, icon: Icon, help }) {
  return <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
    <div className="flex items-start justify-between gap-2"><p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p><span title={help} className="text-slate-400"><Icon size={18}/></span></div>
    <p className="mt-3 text-2xl font-bold tracking-tight text-slate-900">{value}</p>
    <p className="mt-1 text-xs leading-relaxed text-slate-500">{detail}</p>
  </article>;
}

function ChartCard({ title, subtitle, children }) {
  return <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
    <div className="mb-4"><h2 className="font-semibold text-slate-900">{title}</h2><p className="mt-1 text-xs text-slate-500">{subtitle}</p></div>
    <div className="h-64">{children}</div>
  </article>;
}

export default function HistoricalPerformancePage() {
  const [filters, setFilters] = useState({ stock_symbol: '', date_from: '', date_to: '', sentiment: '', impact_direction: '', horizon: '1d', page: '1', page_size: '50' });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const queryKey = JSON.stringify(filters);
  const requestFilters = useMemo(() => JSON.parse(queryKey), [queryKey]);
  const viewerTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');
    getHistoricalPerformanceDashboard({ ...requestFilters, timezone: viewerTimezone })
      .then((result) => { if (active) setData(result); })
      .catch((requestError) => { if (active) setError(requestError.message || 'Historical performance data is unavailable.'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [requestFilters, refresh, viewerTimezone]);

  const horizonMetrics = useMemo(() => Object.fromEntries((data?.performance_by_horizon || []).map((item) => [item.horizon, item])), [data]);
  const evaluations = useMemo(() => Object.fromEntries((data?.evaluation_by_horizon || []).map((item) => [item.horizon, item])), [data]);
  const selectedEval = evaluations[filters.horizon];
  const selectedHorizonMetrics = horizonMetrics[filters.horizon];
  const chartReturns = useMemo(() => HORIZONS.map((horizon) => ({
    horizon: horizon.toUpperCase(),
    stock: horizonMetrics[horizon]?.average_stock_return_pct ?? null,
    nifty: horizonMetrics[horizon]?.average_nifty_return_pct ?? null,
    excess: horizonMetrics[horizon]?.average_excess_return_pct ?? null,
    sample: horizonMetrics[horizon]?.eligible_events ?? 0,
  })), [horizonMetrics]);
  const chartAlignment = useMemo(() => HORIZONS.map((horizon) => ({
    horizon: horizon.toUpperCase(),
    alignment: evaluations[horizon]?.sentiment_return_alignment.alignment_pct ?? null,
    sample: evaluations[horizon]?.sentiment_return_alignment.sample_size ?? 0,
    eligible: evaluations[horizon]?.sentiment_return_alignment.eligible_events ?? 0,
    neutralSentiment: evaluations[horizon]?.sentiment_return_alignment.neutral_sentiment_events ?? 0,
    neutralReturns: evaluations[horizon]?.sentiment_return_alignment.neutral_excess_return_events ?? 0,
  })), [evaluations]);
  const impactReturnGroups = selectedEval?.business_impact_returns || [];
  const chartImpactReturns = impactReturnGroups.map((group) => ({
    direction: group.impact_direction,
    stock: group.average_stock_return_pct,
    excess: group.average_excess_return_pct,
    sample: group.sample_size,
  }));
  const chartOutcomes = useMemo(() => HORIZONS.map((horizon) => ({
    horizon: horizon.toUpperCase(),
    Positive: horizonMetrics[horizon]?.positive_outcomes ?? 0,
    Negative: horizonMetrics[horizon]?.negative_outcomes ?? 0,
  })), [horizonMetrics]);
  const supportedStocks = useMemo(() => (data?.stock_comparison || []).filter((stock) => stock.sample_supported && stock.average_excess_return_pct != null), [data]);
  const bestStock = supportedStocks[0];
  const worstStock = [...supportedStocks].reverse()[0];

  function updateFilter(key, value) {
    setFilters((current) => ({ ...current, [key]: value, page: '1' }));
  }

  function clearFilters() {
    setFilters({ stock_symbol: '', date_from: '', date_to: '', sentiment: '', impact_direction: '', horizon: '1d', page: '1', page_size: '50' });
  }

  const sharedAxis = { tick: { fill: '#64748b', fontSize: 11 }, axisLine: { stroke: '#cbd5e1' }, tickLine: false };

  return <main className="mx-auto max-w-[1500px] space-y-6">
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div><p className="text-xs font-bold uppercase tracking-[0.16em] text-emerald-700">StockPilot · Event study</p><h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">Historical Performance</h1><p className="mt-2 max-w-2xl text-sm text-slate-600">Measure observed returns after persisted news analyses. Completed outcomes only; pending events are excluded from completed-outcome summaries.</p></div>
      <button type="button" onClick={() => setRefresh((value) => value + 1)} disabled={loading} className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"><RefreshCw size={15} className={loading ? 'animate-spin' : ''}/>Refresh</button>
    </header>

    <section aria-label="Historical performance filters" className="grid gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:grid-cols-2 lg:grid-cols-7">
      <label className="text-xs font-semibold text-slate-600">From date<input type="date" value={filters.date_from} onChange={(event) => updateFilter('date_from', event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-200 px-2.5 py-2 text-sm font-normal text-slate-800"/></label>
      <label className="text-xs font-semibold text-slate-600">To date<input type="date" value={filters.date_to} onChange={(event) => updateFilter('date_to', event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-200 px-2.5 py-2 text-sm font-normal text-slate-800"/></label>
      <label className="text-xs font-semibold text-slate-600">Stock<select value={filters.stock_symbol} onChange={(event) => updateFilter('stock_symbol', event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm font-normal"><option value="">All stocks</option>{STOCKS.map((symbol) => <option key={symbol}>{symbol}</option>)}</select></label>
      <label className="text-xs font-semibold text-slate-600">FinBERT sentiment<select value={filters.sentiment} onChange={(event) => updateFilter('sentiment', event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm font-normal"><option value="">All sentiments</option>{SENTIMENTS.map((item) => <option key={item}>{item}</option>)}</select></label>
      <label className="text-xs font-semibold text-slate-600">AI impact direction<select value={filters.impact_direction} onChange={(event) => updateFilter('impact_direction', event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm font-normal"><option value="">All directions</option>{DIRECTIONS.map((item) => <option key={item}>{item}</option>)}</select></label>
      <label className="text-xs font-semibold text-slate-600" title="Selects the horizon for sentiment alignment, the business-impact chart, stock-wise eligibility and event interpretation. Overview returns and grouped tables retain all horizons.">Selected analysis horizon<select value={filters.horizon} onChange={(event) => updateFilter('horizon', event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-sm font-normal">{HORIZONS.map((item) => <option key={item} value={item}>{item.toUpperCase()}</option>)}</select></label>
      <button type="button" onClick={clearFilters} className="inline-flex items-center justify-center gap-2 self-end rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-50"><FilterX size={15}/>Clear</button>
    </section>
    <p className="-mt-4 px-1 text-xs leading-relaxed text-slate-500">The selected horizon controls sentiment alignment, the business-impact chart, stock-wise eligibility and event-row interpretation. Return overview cards, the impact-by-horizon table and event return columns continue to show 1D, 3D and 5D.</p>

    {loading && <div role="status" className="rounded-2xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-600"><RefreshCw size={18} className="mx-auto mb-2 animate-spin text-emerald-700"/>Loading persisted event outcomes…</div>}
    {error && <div role="alert" className="flex gap-3 rounded-2xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-800"><AlertTriangle size={18} className="shrink-0"/><div><p className="font-semibold">Historical performance is unavailable</p><p className="mt-1">{error}</p></div></div>}

    {!loading && !error && data && <>
      {data.total_events === 0 && <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"><p className="font-semibold">No persisted events match these filters.</p><p className="mt-1">New AI impact analyses are stored as article-symbol events. Outcome metrics appear after the selected market sessions complete and the worker records their prices.</p></div>}

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricCard label="News events analysed" value={count(data.total_events)} detail={`${count(data.eligible_events)} eligible at ${filters.horizon.toUpperCase()} · ${count(data.pending_outcomes)} pending`} icon={Newspaper} help="One event per article and affected stock symbol."/>
        <MetricCard label={`${filters.horizon.toUpperCase()} Sentiment–Return Alignment`} value={alignmentPct(selectedEval?.sentiment_return_alignment.alignment_pct)} detail={`${count(selectedEval?.sentiment_return_alignment.sample_size)} directional pairs from ${count(selectedEval?.sentiment_return_alignment.eligible_events)} eligible; neutral sentiment ${count(selectedEval?.sentiment_return_alignment.neutral_sentiment_events)}, neutral returns ${count(selectedEval?.sentiment_return_alignment.neutral_excess_return_events)}, unavailable sentiment ${count(selectedEval?.sentiment_return_alignment.unavailable_sentiment_events)}`} icon={ChartNoAxesCombined} help="Historical polarity alignment between directional FinBERT sentiment and directional excess returns. This is association, not a price forecast or causal effect."/>
        {HORIZONS.map((horizon) => <MetricCard key={horizon} label={`Average ${horizon.toUpperCase()} stock return`} value={pct(horizonMetrics[horizon]?.average_stock_return_pct)} detail={`${count(horizonMetrics[horizon]?.eligible_events)} completed · excess ${pct(horizonMetrics[horizon]?.average_excess_return_pct)}`} icon={ChartNoAxesCombined} help="Arithmetic mean of paired stock returns. Adj Close is used consistently when the dataset has that column; missing adjusted values are excluded. Close is used only if Adj Close is absent. NIFTY uses Close."/>)}
      </section>
      <div className="flex flex-wrap gap-x-5 gap-y-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs text-slate-600"><span className="font-semibold text-slate-800">Completed outcomes</span>{HORIZONS.map((horizon) => <span key={horizon}>{horizon.toUpperCase()} <b>{count(data.completed_outcomes[horizon])}</b></span>)}<span className="text-amber-700">Pending at {filters.horizon.toUpperCase()}: {count(data.pending_outcomes)}</span></div>

      <section className="grid gap-4 xl:grid-cols-2">
        <ChartCard title="Returns by holding horizon" subtitle="Average stock and NIFTY 50 returns; sample counts are shown on hover.">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={chartReturns} margin={{ top: 8, right: 8, left: -18, bottom: 4 }}><CartesianGrid stroke="#e2e8f0" vertical={false}/><XAxis dataKey="horizon" {...sharedAxis}/><YAxis tickFormatter={(value) => `${value}%`} {...sharedAxis}/><Tooltip formatter={(value, name, item) => [typeof value === 'number' ? `${value.toFixed(2)}% · n=${item.payload.sample}` : 'Insufficient data', name]}/><Legend/><Bar dataKey="stock" name="Stock return" fill="#15803d" radius={[4,4,0,0]}/><Bar dataKey="nifty" name="NIFTY 50 return" fill="#64748b" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Sentiment–Return Alignment by horizon" subtitle="Directional label agreement only; neutral sentiment, neutral excess returns, and missing sentiment are excluded and counted separately.">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={chartAlignment} margin={{ top: 8, right: 8, left: -18, bottom: 4 }}><CartesianGrid stroke="#e2e8f0" vertical={false}/><XAxis dataKey="horizon" {...sharedAxis}/><YAxis domain={[0,100]} tickFormatter={(value) => `${value}%`} {...sharedAxis}/><Tooltip formatter={(value, _name, item) => [typeof value === 'number' ? `${value.toFixed(1)}% · directional n=${item.payload.sample} · eligible=${item.payload.eligible} · neutral sentiment=${item.payload.neutralSentiment} · neutral excess=${item.payload.neutralReturns}` : `No directional sample · eligible=${item.payload.eligible}`, 'Sentiment–return alignment']}/><Bar dataKey="alignment" name="Directional alignment" fill="#0f766e" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer>
        </ChartCard>
        <ChartCard title={`Business Impact vs Subsequent Returns (${filters.horizon.toUpperCase()})`} subtitle="Historical average stock and excess returns grouped by business-impact direction. Groups are descriptive and are not BUY/SELL predictions.">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={chartImpactReturns} margin={{ top: 8, right: 8, left: -18, bottom: 4 }}><CartesianGrid stroke="#e2e8f0" vertical={false}/><XAxis dataKey="direction" {...sharedAxis}/><YAxis tickFormatter={(value) => `${value}%`} {...sharedAxis}/><Tooltip formatter={(value, name, item) => [typeof value === 'number' ? `${value.toFixed(2)}% · n=${item.payload.sample}` : `No observations · n=${item.payload.sample}`, name]}/><Legend/><Bar dataKey="stock" name="Average stock return" fill="#15803d" radius={[4,4,0,0]}/><Bar dataKey="excess" name="Average excess return" fill="#0f766e" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer>
        </ChartCard>
        <ChartCard title="Positive and negative stock outcomes" subtitle="Outcome distribution by stock-return sign; neutral observations are excluded from these two bars.">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={chartOutcomes} margin={{ top: 8, right: 8, left: -18, bottom: 4 }}><CartesianGrid stroke="#e2e8f0" vertical={false}/><XAxis dataKey="horizon" {...sharedAxis}/><YAxis allowDecimals={false} {...sharedAxis}/><Tooltip/><Legend/><Bar dataKey="Positive" fill="#15803d" radius={[4,4,0,0]}/><Bar dataKey="Negative" fill="#be123c" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer>
        </ChartCard>
        <ChartCard title="NIFTY 50 vs stock excess return" subtitle="The excess series is stock return minus NIFTY 50 return, not a separate asset return.">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={chartReturns} margin={{ top: 8, right: 8, left: -18, bottom: 4 }}><CartesianGrid stroke="#e2e8f0" vertical={false}/><XAxis dataKey="horizon" {...sharedAxis}/><YAxis tickFormatter={(value) => `${value}%`} {...sharedAxis}/><Tooltip formatter={(value, name, item) => [typeof value === 'number' ? `${value.toFixed(2)}% · n=${item.payload.sample}` : 'Insufficient data', name]}/><Legend/><Bar dataKey="nifty" name="NIFTY 50 return" fill="#64748b" radius={[4,4,0,0]}/><Bar dataKey="excess" name="Stock excess return" fill="#0f766e" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer>
        </ChartCard>
      </section>

      <section className="grid gap-3 sm:grid-cols-2">
        <article className="rounded-xl border border-slate-200 bg-white p-4"><p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{filters.horizon.toUpperCase()} stock outcome distribution</p><div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-sm"><span className="text-emerald-700">Positive {pct(selectedHorizonMetrics?.positive_outcome_pct)}</span><span className="text-rose-700">Negative {pct(selectedHorizonMetrics?.negative_outcome_pct)}</span><span className="text-slate-500">Neutral {count(selectedHorizonMetrics?.neutral_outcomes)}</span></div><p className="mt-1 text-xs text-slate-400">Denominator: {count(selectedHorizonMetrics?.eligible_events)} completed stock outcomes.</p></article>
        <article className="rounded-xl border border-slate-200 bg-white p-4"><p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Impact strength vs return magnitude</p><p className="mt-2 text-lg font-bold text-slate-900">{typeof selectedEval?.impact_strength_vs_magnitude.pearson_correlation === 'number' ? selectedEval.impact_strength_vs_magnitude.pearson_correlation.toFixed(3) : 'Insufficient data'}</p><p className="mt-1 text-xs text-slate-500">Pearson r · n={count(selectedEval?.impact_strength_vs_magnitude.sample_size)} · ordinal strength vs absolute excess return.</p></article>
      </section>

      <section className="overflow-x-auto rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
        <h2 className="font-semibold text-slate-900">Business Impact vs Subsequent Returns</h2>
        <p className="mt-1 text-xs text-slate-500">Average returns are descriptive historical associations, grouped by business-impact direction. These categories are not BUY/SELL signals.</p>
        <table className="mt-3 w-full min-w-[680px] text-left text-sm"><thead className="border-b border-slate-200 text-xs text-slate-500"><tr><th className="px-3 py-2">Horizon</th><th className="px-3 py-2">Business impact</th><th className="px-3 py-2">Eligible events</th><th className="px-3 py-2">Average stock return</th><th className="px-3 py-2">Average excess return</th></tr></thead><tbody>{(data.evaluation_by_horizon || []).flatMap((evaluation) => evaluation.business_impact_returns.map((group) => <tr key={`${evaluation.horizon}-${group.impact_direction}`} className="border-b border-slate-100 last:border-0"><td className="px-3 py-2 font-semibold">{evaluation.horizon.toUpperCase()}</td><td className="px-3 py-2">{group.impact_direction}</td><td className="px-3 py-2">{count(group.sample_size)}</td><td className={`px-3 py-2 ${signed(group.average_stock_return_pct)}`}>{pct(group.average_stock_return_pct)}</td><td className={`px-3 py-2 ${signed(group.average_excess_return_pct)}`}>{pct(group.average_excess_return_pct)}</td></tr>))}</tbody></table>
      </section>

      <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
        <div className="flex flex-wrap items-end justify-between gap-2"><div><h2 className="font-semibold text-slate-900">Stock-wise performance</h2><p className="mt-1 text-xs text-slate-500">Average excess returns use completed {filters.horizon.toUpperCase()} outcomes; highlights require at least {data.minimum_stock_sample_size} observations.</p></div><span className="text-xs text-slate-500">{supportedStocks.length} stocks meet the sample threshold</span></div>
        <div className="grid gap-3 sm:grid-cols-2">
          {[['Highest sample-supported excess return', bestStock, 'text-emerald-700'], ['Lowest sample-supported excess return', worstStock, 'text-rose-700']].map(([title, stock, color]) => <article key={title} className="rounded-xl border border-slate-200 bg-slate-50 p-4"><p className="text-xs font-semibold text-slate-500">{title}</p>{stock ? <div className="mt-2 flex items-end justify-between gap-2"><div><p className="text-lg font-bold text-slate-900">{stock.symbol}</p><p className="text-xs text-slate-500">n={stock.eligible_events} completed events</p></div><p className={`text-xl font-bold ${color}`}>{pct(stock.average_excess_return_pct)}</p></div> : <p className="mt-2 text-sm text-slate-500">Insufficient completed events for a sample-supported comparison.</p>}</article>)}
        </div>
        <div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-sm"><thead className="border-b border-slate-200 text-xs text-slate-500"><tr><th className="px-3 py-2">Symbol</th><th className="px-3 py-2">Events</th><th className="px-3 py-2">Eligible at {filters.horizon.toUpperCase()}</th><th className="px-3 py-2">Avg excess return</th><th className="px-3 py-2">Sample support</th></tr></thead><tbody>{data.stock_comparison.map((stock) => <tr key={stock.symbol} className="border-b border-slate-100 last:border-0"><td className="px-3 py-2 font-bold text-slate-800">{stock.symbol}</td><td className="px-3 py-2">{count(stock.event_count)}</td><td className="px-3 py-2">{count(stock.eligible_events)}</td><td className={`px-3 py-2 font-medium ${signed(stock.average_excess_return_pct)}`}>{pct(stock.average_excess_return_pct)}</td><td className="px-3 py-2">{stock.sample_supported ? <span className="text-emerald-700">Supported</span> : <span className="text-slate-400">Low sample</span>}</td></tr>)}</tbody></table></div>
      </section>

      <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-wrap items-end justify-between gap-3 border-b border-slate-100 p-4 sm:p-5"><div><h2 className="font-semibold text-slate-900">Historical news events</h2><p className="mt-1 text-xs text-slate-500">Sentiment-return alignment is descriptive, not predictive. Actual excess direction and alignment use the selected {filters.horizon.toUpperCase()} horizon; all three return horizons remain visible. Pending outcomes remain unscored.</p></div><div className="flex items-center gap-2 text-xs text-slate-500"><CalendarDays size={14}/>Page {data.page} of {data.total_pages || 1} · {count(data.total_events)} article-symbol observations</div></div>
        {data.events.length === 0 ? <div className="p-10 text-center"><Newspaper size={25} className="mx-auto mb-2 text-slate-400"/><p className="font-semibold text-slate-700">No historical events match these filters</p><p className="mt-1 text-sm text-slate-500">Clear one or more filters to view available events.</p></div> : <div className="overflow-x-auto"><table className="w-full min-w-[1380px] text-left text-sm"><thead className="border-b border-slate-200 bg-slate-50 text-xs text-slate-500"><tr><th className="px-3 py-3">Headline</th><th className="px-3 py-3">Stock</th><th className="px-3 py-3">Published</th><th className="px-3 py-3">FinBERT</th><th className="px-3 py-3">Business impact</th><th className="px-3 py-3">1D stock / NIFTY / excess</th><th className="px-3 py-3">3D stock / NIFTY / excess</th><th className="px-3 py-3">5D stock / NIFTY / excess</th><th className="px-3 py-3">Actual excess direction ({filters.horizon.toUpperCase()})</th><th className="px-3 py-3">Sentiment-return alignment ({filters.horizon.toUpperCase()})</th></tr></thead><tbody>{data.events.map((event) => <tr key={event.event_id} className="border-b border-slate-100 align-top last:border-0"><td className="max-w-[320px] px-3 py-3"><a href={event.article_url} target="_blank" rel="noopener noreferrer" className="line-clamp-2 font-medium text-slate-800 hover:underline">{event.headline}</a></td><td className="px-3 py-3 font-bold">{event.symbol}</td><td className="whitespace-nowrap px-3 py-3 text-xs text-slate-500">{dateLabel(event.published_at)}</td><td className="px-3 py-3">{event.finbert_sentiment || 'Unavailable'}</td><td className="px-3 py-3">{event.ai_impact_direction} <span className="text-xs text-slate-500">· {event.ai_impact_strength}</span></td>{HORIZONS.map((horizon) => <td key={horizon} className="whitespace-nowrap px-3 py-3 text-xs"><span className={signed(event.stock_returns_pct[horizon])}>{pct(event.stock_returns_pct[horizon])}</span><span className="text-slate-400"> / </span><span className={signed(event.nifty_returns_pct[horizon])}>{pct(event.nifty_returns_pct[horizon])}</span><span className="text-slate-400"> / </span><span className={signed(event.excess_returns_pct[horizon])}>{pct(event.excess_returns_pct[horizon])}</span></td>)}<td className="px-3 py-3">{event.actual_excess_direction || "Pending"}</td><td className="px-3 py-3">{event.sentiment_return_alignment}</td></tr>)}</tbody></table></div>}
        <div className="flex justify-end gap-2 border-t border-slate-100 p-3"><button type="button" disabled={data.page <= 1} onClick={() => setFilters((current) => ({ ...current, page: String(Math.max(1, Number(current.page) - 1)) }))} className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 disabled:opacity-40">Previous</button><button type="button" disabled={data.page >= data.total_pages} onClick={() => setFilters((current) => ({ ...current, page: String(Math.min(data.total_pages, Number(current.page) + 1)) }))} className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 disabled:opacity-40">Next</button></div>
      </section>

      <details className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-600"><summary className="flex cursor-pointer list-none items-center gap-2 font-semibold text-slate-800"><CircleHelp size={16} className="text-emerald-700"/>Methodology and interpretation</summary><ul className="mt-3 list-disc space-y-2 pl-5 text-xs leading-relaxed">{data.methodology.map((item) => <li key={item}>{item}</li>)}</ul><p className="mt-3 border-t border-slate-100 pt-3 text-xs text-slate-500">Observed performance describes historical association. AI impact is an assessment of potential business impact, not a share-price forecast; these results do not establish causation.</p></details>
      <footer className="text-xs text-slate-400">Data as of {data.data_as_of}. Alignment uses directional sentiment and excess-return pairs only; insufficient samples and missing prices are shown without imputation.</footer>
    </>}
  </main>;
}
