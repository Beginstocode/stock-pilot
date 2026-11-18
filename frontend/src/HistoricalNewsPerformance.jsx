import { useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, BarChart3, Newspaper, RefreshCw } from 'lucide-react';
import { getHistoricalNewsPerformance } from './api.js';

const pct = (value) => typeof value === 'number' && Number.isFinite(value) ? `${value > 0 ? '+' : ''}${value.toFixed(2)}%` : 'Insufficient data';
const count = (value) => Number.isFinite(value) ? value.toLocaleString() : '—';
const signedClass = (value) => typeof value === 'number' ? (value >= 0 ? 'text-emerald-700' : 'text-rose-700') : 'text-slate-400';

function MetricCard({ label, value, detail, icon: Icon }) {
  return <article className="rounded-xl border border-slate-200 bg-white p-4">
    <div className="flex items-center justify-between gap-2"><p className="text-xs font-medium text-slate-500">{label}</p><Icon size={16} className="text-slate-400"/></div>
    <p className="mt-3 text-xl font-bold text-slate-900">{value}</p>
    <p className="mt-1 text-xs text-slate-500">{detail}</p>
  </article>;
}

function ReturnTable({ rows, symbolMode = false }) {
  return <div className="overflow-x-auto">
    <table className="w-full min-w-[620px] text-left text-sm">
      <thead className="border-b border-slate-200 text-xs text-slate-500"><tr>
        <th className="px-3 py-2 font-semibold">{symbolMode ? 'Stock' : 'FinBERT sentiment'}</th>
        <th className="px-3 py-2 font-semibold">1 trading day</th><th className="px-3 py-2 font-semibold">3 trading days</th><th className="px-3 py-2 font-semibold">5 trading days</th>
        {!symbolMode&&<th className="px-3 py-2 font-semibold">Event pairs</th>}
      </tr></thead>
      <tbody>{rows.map((row) => <tr key={symbolMode ? row.symbol : row.sentiment} className="border-b border-slate-100 last:border-0">
        <td className="px-3 py-2 font-semibold text-slate-800">{symbolMode ? row.symbol : row.sentiment}</td>
        {[row.average_excess_return_1d_pct, row.average_excess_return_3d_pct, row.average_excess_return_5d_pct].map((value, index) => <td key={index} className={`px-3 py-2 font-medium ${signedClass(value)}`}>{pct(value)}</td>)}
        {!symbolMode&&<td className="px-3 py-2 text-slate-600">1d {count(row.event_observations_1d)} · 3d {count(row.event_observations_3d)} · 5d {count(row.event_observations_5d)}</td>}
      </tr>)}</tbody>
    </table>
  </div>;
}

export default function HistoricalNewsPerformance() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');
    getHistoricalNewsPerformance()
      .then((result) => { if (active) setData(result); })
      .catch((requestError) => { if (active) setError(requestError.message || 'Historical news performance is unavailable.'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [refreshKey]);

  const metrics = useMemo(() => Object.fromEntries((data?.returns_by_horizon || []).map((item) => [item.horizon, item])), [data]);
  const eligibleEvents = useMemo(() => (data?.events || []).filter((event) => event.eligible).sort((a, b) => b.published_at.localeCompare(a.published_at)), [data]);
  const oneDay = metrics['1d'];

  return <section aria-labelledby="historical-news-performance-title" className="space-y-5 rounded-2xl border border-slate-200 bg-slate-50 p-4 md:p-6">
    <header className="flex flex-wrap items-center justify-between gap-3">
      <div><p className="text-xs font-semibold uppercase tracking-wide text-indigo-700">Market News · Event study</p><h2 id="historical-news-performance-title" className="mt-1 text-xl font-bold text-slate-900">Historical News Performance</h2><p className="mt-1 text-sm text-slate-500">Observed stock returns after publication compared with NIFTY 50.</p></div>
      <button type="button" onClick={() => setRefreshKey((key) => key + 1)} disabled={loading} className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 disabled:opacity-50"><RefreshCw size={14} className={loading ? 'animate-spin' : ''}/>Refresh</button>
    </header>

    {loading&&<div role="status" className="rounded-xl bg-white p-6 text-sm text-slate-600">Calculating historical event outcomes from available daily prices…</div>}
    {error&&<div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700"><p className="font-semibold">Historical analysis unavailable</p><p className="mt-1">{error}</p></div>}
    {!loading&&!error&&data&&<>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricCard label="News articles analysed" value={count(data.total_events_analyzed)} detail={`${count(data.excluded_events)} without a complete 1-day stock + NIFTY result`} icon={Newspaper}/>
        {['1d','3d','5d'].map((horizon) => <MetricCard key={horizon} label={`Average ${horizon} stock return`} value={pct(metrics[horizon]?.average_stock_return_pct)} detail={`NIFTY ${pct(metrics[horizon]?.average_nifty_return_pct)} · Excess ${pct(metrics[horizon]?.average_excess_return_pct)} · ${count(metrics[horizon]?.event_observations)} pairs`} icon={BarChart3}/>)}
        <MetricCard label="1D Sentiment-return alignment" value={oneDay?.finbert_alignment_pct == null ? 'Insufficient directional sample' : `${oneDay.finbert_alignment_pct.toFixed(1)}%`} detail={`Directional n=${count(oneDay?.finbert_alignment_sample_size)} / eligible=${count(oneDay?.event_observations)} · neutral sentiment=${count(oneDay?.neutral_sentiment_observations)} · neutral excess=${count(oneDay?.neutral_excess_return_observations)} · unavailable=${count(oneDay?.unavailable_sentiment_observations)}`} icon={Activity}/>
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs text-slate-600"><span className="font-semibold text-slate-800">1-day stock outcomes</span><span className="text-emerald-700">Positive {count(oneDay?.positive_outcomes)}</span><span className="text-rose-700">Negative {count(oneDay?.negative_outcomes)}</span><span>Flat {count(oneDay?.neutral_outcomes)}</span></div>

      <div className="grid gap-4 xl:grid-cols-2">
        <article className="rounded-xl border border-slate-200 bg-white p-4"><h3 className="mb-1 font-semibold text-slate-900">Excess return by FinBERT sentiment</h3>{!data.finbert_available&&<p className="mb-3 text-xs text-amber-700">FinBERT history is unavailable: {data.finbert_unavailable_reason || 'No scored news records were found.'}</p>}<ReturnTable rows={data.sentiment_performance}/></article>
        <article className="rounded-xl border border-slate-200 bg-white p-4"><h3 className="mb-1 font-semibold text-slate-900">Stock-wise excess return comparison</h3><p className="mb-3 text-xs text-slate-500">Average stock return minus the matching NIFTY 50 return.</p><ReturnTable rows={data.stock_comparison} symbolMode/></article>
      </div>

      {data.ai_impact_performance.available ? <article className="overflow-x-auto rounded-xl border border-slate-200 bg-white p-4"><h3 className="font-semibold text-slate-900">Business Impact vs Subsequent 1D Returns</h3><p className="mb-3 mt-1 text-xs text-slate-500">Descriptive historical averages by business-impact category, not BUY/SELL predictions. {data.ai_impact_performance.reason}</p><table className="w-full min-w-[600px] text-left text-sm"><thead className="border-b border-slate-200 text-xs text-slate-500"><tr><th className="px-3 py-2">Business impact</th><th className="px-3 py-2">Eligible events</th><th className="px-3 py-2">Average stock return</th><th className="px-3 py-2">Average excess return</th></tr></thead><tbody>{data.ai_impact_performance.returns_by_impact_direction.map((group) => <tr key={group.impact_direction} className="border-b border-slate-100 last:border-0"><td className="px-3 py-2">{group.impact_direction}</td><td className="px-3 py-2">{count(group.event_observations)}</td><td className={`px-3 py-2 ${signedClass(group.average_stock_return_1d_pct)}`}>{pct(group.average_stock_return_1d_pct)}</td><td className={`px-3 py-2 ${signedClass(group.average_excess_return_1d_pct)}`}>{pct(group.average_excess_return_1d_pct)}</td></tr>)}</tbody></table></article> : <article className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950"><AlertTriangle size={18} className="mt-0.5 shrink-0 text-amber-700"/><div><h3 className="font-semibold">Historical business-impact return grouping is unavailable</h3><p className="mt-1 leading-relaxed">{data.ai_impact_performance.reason}</p></div></article>}

      <article className="overflow-hidden rounded-xl border border-slate-200 bg-white">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-4 py-3"><div><h3 className="font-semibold text-slate-900">Eligible historical events</h3><p className="mt-1 text-xs text-slate-500">Daily close-to-close returns; future horizons may be unavailable near the dataset boundary.</p></div><span className="text-xs text-slate-500">Price data through {data.nifty_coverage_end}</span></div>
        {eligibleEvents.length===0?<p className="p-6 text-sm text-slate-500">No events have complete stock and benchmark data.</p>:<div className="overflow-x-auto"><table className="w-full min-w-[980px] text-left text-sm">
          <thead className="border-b border-slate-200 bg-slate-50 text-xs text-slate-500"><tr><th className="px-3 py-2 font-semibold">Published</th><th className="px-3 py-2 font-semibold">Stock</th><th className="px-3 py-2 font-semibold">Article</th><th className="px-3 py-2 font-semibold">FinBERT</th><th className="px-3 py-2 font-semibold">1d stock / excess</th><th className="px-3 py-2 font-semibold">3d stock / excess</th><th className="px-3 py-2 font-semibold">5d stock / excess</th></tr></thead>
          <tbody>{eligibleEvents.map((event) => <tr key={`${event.event_id}-${event.symbol}`} className="border-b border-slate-100 last:border-0"><td className="whitespace-nowrap px-3 py-2 text-xs text-slate-500">{new Date(event.published_at).toLocaleDateString()}</td><td className="px-3 py-2 font-bold text-slate-800">{event.symbol}</td><td className="max-w-sm px-3 py-2"><a href={event.article_url} target="_blank" rel="noopener noreferrer" className="line-clamp-2 font-medium text-slate-700 hover:underline">{event.headline}</a></td><td className="px-3 py-2 text-slate-600">{event.finbert_sentiment || 'Unavailable'}</td>{['1d','3d','5d'].map((horizon) => { const outcome = event.outcomes[horizon]; return <td key={horizon} className="whitespace-nowrap px-3 py-2"><span className={signedClass(outcome.stock_return_pct)}>{pct(outcome.stock_return_pct)}</span><span className="text-slate-400"> / </span><span className={signedClass(outcome.excess_return_pct)}>{pct(outcome.excess_return_pct)}</span></td>; })}</tr>)}</tbody>
        </table></div>}
      </article>

      <footer className="space-y-1 text-xs leading-relaxed text-slate-500"><p>Price coverage: stocks {data.stock_price_coverage_start}–{data.stock_price_coverage_end}; NIFTY 50 {data.nifty_coverage_start}–{data.nifty_coverage_end}. Analysis cut-off: {data.data_as_of}.</p><p>{data.disclaimer}</p></footer>
    </>}
  </section>;
}
