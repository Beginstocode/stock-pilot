import { useEffect, useMemo, useRef, useState } from 'react';
import { ExternalLink, Search, Newspaper, TrendingUp, TrendingDown, Minus, RefreshCw, Sparkles } from 'lucide-react';
import { getNews, getNewsImpact } from './api.js';
import HistoricalNewsPerformance from './HistoricalNewsPerformance.jsx';

const STOCKS = ['RELIANCE','TCS','INFY','HDFCBANK','ICICIBANK','SBIN','ITC','LT','BHARTIARTL','TATAMOTORS'];
const TABS = ['All', 'Positive', 'Negative', 'Neutral'];
const PAGE_SIZE = 20;
const contentSourceLabel = { full_article: 'Full article', rss_summary: 'RSS summary', headline_only: 'Headline only' };
const tone = { Positive: 'bg-emerald-50 text-emerald-700', Negative: 'bg-rose-50 text-rose-700', Neutral: 'bg-amber-50 text-amber-700' };
const formatDate = (value) => { const date = new Date(value); return Number.isNaN(date.getTime()) ? 'Publication time unavailable' : date.toLocaleString(); };

function SummaryCard({ label, count, icon: Icon, color }) {
  return <article className="rounded-2xl border border-slate-200 bg-white p-5"><div className="flex items-center justify-between"><p className="text-sm text-slate-500">{label}</p><span className={`rounded-xl p-2 ${color}`}><Icon size={18}/></span></div><p className="mt-4 text-2xl font-bold">{count}</p><p className="mt-1 text-xs text-slate-400">In current results</p></article>;
}

function SentimentBadge({ sentiment }) {
  if (!sentiment) return <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-500">Sentiment unavailable</span>;
  const Icon = sentiment === 'Positive' ? TrendingUp : sentiment === 'Negative' ? TrendingDown : Minus;
  return <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${tone[sentiment] || tone.Neutral}`}><Icon size={13}/>{sentiment}</span>;
}

function ArticleCard({ article }) {
  const [impactExpanded,setImpactExpanded]=useState(false),[impactLoading,setImpactLoading]=useState(false),[impactError,setImpactError]=useState(''),[impact,setImpact]=useState(null);
  const impactRequestStarted=useRef(false);
  let validUrl = null;
  try { const parsed = new URL(article.article_url); if (parsed.protocol === 'https:' || parsed.protocol === 'http:') validUrl = parsed.href; } catch { /* Render unavailable state below. */ }
  const probs = [['Positive', article.positive_probability], ['Negative', article.negative_probability], ['Neutral', article.neutral_probability]];
  async function loadImpact() {
    if (impactRequestStarted.current) return;
    impactRequestStarted.current=true;
    setImpactLoading(true);
    setImpactError('');
    try { setImpact(await getNewsImpact(article.article_url)); }
    catch (error) { setImpactError(error.message || 'AI impact analysis is unavailable.'); }
    finally { impactRequestStarted.current=false;setImpactLoading(false); }
  }
  function toggleImpact() {
    const expanded = !impactExpanded;
    setImpactExpanded(expanded);
    if (expanded && !impact && !impactLoading) void loadImpact();
  }
  return <article className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white p-5 transition hover:border-slate-300 hover:shadow-sm">
    <div className="flex flex-wrap items-center gap-2"><span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-bold text-slate-700">{article.symbol}</span><SentimentBadge sentiment={article.sentiment}/><time className="ml-auto text-xs text-slate-500">{formatDate(article.published_at)}</time></div>
    <h2 className="mt-4 text-lg font-semibold leading-snug text-slate-900">{article.headline}</h2>
    <div className="mt-2 flex flex-wrap items-center gap-2"><span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${article.content_source === 'full_article' ? 'bg-emerald-50 text-emerald-700' : article.content_source === 'headline_only' ? 'bg-amber-50 text-amber-700' : 'bg-slate-100 text-slate-600'}`}>{contentSourceLabel[article.content_source] || 'RSS summary'}</span>{article.article_author&&<span className="text-xs text-slate-500">By {article.article_author}</span>}</div>
    {article.article_title&&article.article_title!==article.headline&&<p className="mt-2 text-sm font-medium text-slate-600">Publisher title: {article.article_title}</p>}
    <p className="mt-2 text-sm text-slate-500">{article.source || 'Source unavailable'}</p>
    <div className="mt-5 rounded-xl bg-slate-50 p-3"><div className="flex justify-between text-xs"><span className="font-medium text-slate-600">FinBERT sentiment score</span><span className="font-semibold text-slate-900">{typeof article.sentiment_score === 'number' ? article.sentiment_score.toFixed(3) : 'Unavailable'}</span></div><div className="mt-3 grid grid-cols-3 gap-2">{probs.map(([name,value])=><div key={name} className="text-center"><p className="text-[11px] text-slate-500">{name}</p><p className="mt-1 text-sm font-semibold">{typeof value === 'number' ? `${(value*100).toFixed(1)}%` : '—'}</p></div>)}</div></div>
    <div className="mt-auto pt-5">{validUrl?<a href={validUrl} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 text-sm font-semibold text-slate-800 hover:underline">Read article <ExternalLink size={15}/></a>:<span className="text-sm text-slate-400">Article link unavailable</span>}</div>
    <section className="mt-4 border-t border-slate-100 pt-4">
      <button type="button" onClick={toggleImpact} aria-expanded={impactExpanded} className="inline-flex w-full items-center justify-between gap-3 rounded-xl border border-indigo-100 bg-indigo-50/70 px-4 py-3 text-left text-sm font-semibold text-indigo-950 transition hover:bg-indigo-50">
        <span className="inline-flex items-center gap-2"><Sparkles size={16} className="text-indigo-600"/>{impactExpanded?'Hide':'AI Impact Analysis'}</span><span className="text-xs font-medium text-indigo-700">{impactLoading?'Analyzing…':impactExpanded?'Close':'Expand'}</span>
      </button>
      {impactExpanded&&<div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
        {impactLoading&&<p role="status" className="text-sm text-slate-600">Reviewing article context, company relevance, and FinBERT sentiment…</p>}
        {impactError&&<div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700"><p className="font-semibold">Impact analysis unavailable</p><p className="mt-1">{impactError}</p><button type="button" onClick={()=>void loadImpact()} className="mt-2 font-semibold underline">Retry</button></div>}
        {impact&&<div className="space-y-4">
          {impact.content_basis==='headline_only'?<p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">Headline only · low content quality. The publisher article and RSS summary were unavailable.</p>:impact.content_basis==='rss_summary'?<p className="rounded-lg bg-slate-100 px-3 py-2 text-xs text-slate-600">RSS summary · medium content quality. A publisher article body was unavailable.</p>:<p className="rounded-lg bg-emerald-50 px-3 py-2 text-xs text-emerald-800">Full article · high content quality. Analysis uses the extracted publisher article.</p>}
          <div><p className="text-xs font-bold uppercase tracking-wide text-slate-500">AI summary</p><p className="mt-1 text-sm leading-relaxed text-slate-800">{impact.summary}</p></div>
          <div className="flex flex-wrap gap-2"><span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700">Direction: {impact.impact_direction}</span><span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700">Strength: {impact.impact_strength}</span><span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700">Horizon: {impact.time_horizon}</span></div>
          <div><p className="text-xs font-bold uppercase tracking-wide text-slate-500">Affected stocks</p>{impact.affected_symbols.length?<div className="mt-2 flex flex-wrap gap-2">{impact.affected_symbols.map(symbol=><span key={symbol} className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-bold text-slate-800">{symbol}</span>)}</div>:<p className="mt-1 text-sm text-slate-600">No supported listed company was assessed as materially affected.</p>}</div>
          <div><p className="text-xs font-bold uppercase tracking-wide text-slate-500">Why it matters</p><p className="mt-1 text-sm leading-relaxed text-slate-700">{impact.explanation}</p></div>
          <div><p className="text-xs font-bold uppercase tracking-wide text-slate-500">Risks and uncertainties</p><ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-slate-700">{impact.risks.map((risk,index)=><li key={`${index}-${risk}`}>{risk}</li>)}</ul></div>
          {impact.recommendations.length>0&&<div><p className="text-xs font-bold uppercase tracking-wide text-slate-500">Current model recommendations</p><div className="mt-2 grid gap-2 sm:grid-cols-2">{impact.recommendations.map(item=><div key={item.symbol} className="rounded-lg border border-slate-200 bg-white p-3 text-sm"><p className="font-semibold text-slate-800">{item.symbol}</p>{item.recommendation?<p className="mt-1 text-slate-700">{item.recommendation}{typeof item.confidence==='number'?` · ${(item.confidence*100).toFixed(1)}% confidence`:''}</p>:<p className="mt-1 text-slate-500">Recommendation unavailable</p>}{item.prediction_date&&<p className="mt-1 text-xs text-slate-400">Market data date: {item.prediction_date}</p>}</div>)}</div><p className="mt-2 text-xs leading-relaxed text-slate-500">These existing model outputs are contextual only and are not adjusted by this news analysis.</p></div>}
          <p className="border-t border-slate-200 pt-3 text-xs leading-relaxed text-slate-500">{impact.disclaimer}</p>
        </div>}
      </div>}
    </section>
  </article>;
}

export default function MarketNewsPage() {
  const [articles,setArticles]=useState([]),[sentiment,setSentiment]=useState('All'),[symbol,setSymbol]=useState('All stocks'),[search,setSearch]=useState(''),[loading,setLoading]=useState(true),[error,setError]=useState(''),[page,setPage]=useState(1),[pagination,setPagination]=useState({total_articles:0,current_page:1,page_size:PAGE_SIZE,total_pages:0,sentiment_counts:{Positive:0,Negative:0,Neutral:0}}),[refreshKey,setRefreshKey]=useState(0);
  useEffect(()=>{let active=true;setLoading(true);setError('');getNews({symbol:symbol==='All stocks'?'':symbol,sentiment:sentiment==='All'?'':sentiment,search:search.trim(),page:String(page),page_size:String(PAGE_SIZE)}).then(data=>{if(active){setArticles(Array.isArray(data.articles)?data.articles:[]);setPagination(data);}}).catch(err=>{if(active){setArticles([]);setPagination({total_articles:0,current_page:1,page_size:PAGE_SIZE,total_pages:0,sentiment_counts:{Positive:0,Negative:0,Neutral:0}});setError(err.message||'Could not load market news.')}}).finally(()=>{if(active)setLoading(false)});return()=>{active=false}},[sentiment,symbol,search,page,refreshKey]);
  const counts=useMemo(()=>({All:pagination.total_articles,...pagination.sentiment_counts}),[pagination]);
  return <div className="space-y-6">
    <header className="flex flex-wrap items-end justify-between gap-3"><div><p className="text-sm text-slate-500">Latest financial news and market sentiment</p><h1 className="mt-1 text-2xl font-bold">Market News</h1></div><button onClick={()=>setRefreshKey(k=>k+1)} disabled={loading} className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium hover:bg-slate-50 disabled:opacity-50"><RefreshCw size={15} className={loading?'animate-spin':''}/>Refresh</button></header>
    <HistoricalNewsPerformance/>
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><SummaryCard label="Total articles" count={loading?'…':counts.All} icon={Newspaper} color="bg-slate-100 text-slate-700"/><SummaryCard label="Positive news" count={loading?'…':counts.Positive} icon={TrendingUp} color="bg-emerald-50 text-emerald-700"/><SummaryCard label="Negative news" count={loading?'…':counts.Negative} icon={TrendingDown} color="bg-rose-50 text-rose-700"/><SummaryCard label="Neutral news" count={loading?'…':counts.Neutral} icon={Minus} color="bg-amber-50 text-amber-700"/></section>
    <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4 md:p-5"><label className="relative block"><Search size={17} className="absolute left-3 top-3 text-slate-400"/><input value={search} onChange={e=>{setPage(1);setSearch(e.target.value)}} placeholder="Search by headline, company or stock symbol" className="w-full rounded-xl border border-slate-200 py-2.5 pl-10 pr-3 text-sm outline-none focus:border-slate-400"/></label>
      <div className="flex gap-2 overflow-x-auto pb-1">{TABS.map(tab=><button key={tab} onClick={()=>{setPage(1);setSentiment(tab)}} className={`whitespace-nowrap rounded-xl px-4 py-2 text-sm font-semibold ${sentiment===tab?'bg-slate-900 text-white':'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>{tab}{!loading&&<span className="ml-2 opacity-70">{counts[tab]}</span>}</button>)}</div>
      <div><p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Filter by stock</p><div className="flex flex-wrap gap-2"><button onClick={()=>{setPage(1);setSymbol('All stocks')}} className={`rounded-lg border px-3 py-1.5 text-xs font-medium ${symbol==='All stocks'?'border-slate-900 bg-slate-900 text-white':'border-slate-200 text-slate-600'}`}>All stocks</button>{STOCKS.map(item=><button key={item} onClick={()=>{setPage(1);setSymbol(item)}} className={`rounded-lg border px-3 py-1.5 text-xs font-medium ${symbol===item?'border-slate-900 bg-slate-900 text-white':'border-slate-200 text-slate-600 hover:border-slate-400'}`}>{item}</button>)}</div></div>
    </section>
    {error&&<div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700"><p className="font-semibold">News could not be loaded</p><p className="mt-1">{error}</p></div>}
    {loading&&<div role="status" className="grid gap-4 md:grid-cols-2"><div className="h-56 animate-pulse rounded-2xl bg-slate-200"/><div className="h-56 animate-pulse rounded-2xl bg-slate-200"/></div>}
    {!loading&&!error&&!pagination.total_articles&&<div className="rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center"><Newspaper size={30} className="mx-auto mb-3 text-slate-400"/><h2 className="font-semibold">No news available</h2><p className="mx-auto mt-2 max-w-md text-sm text-slate-500">There are no articles matching these filters in the available news feed.</p></div>}
    {!loading&&!error&&articles.length>0&&<><section className="grid gap-4 md:grid-cols-2">{articles.map((item,index)=><ArticleCard key={`${item.article_url}-${item.symbol}-${index}`} article={item}/>)}</section><nav aria-label="Market news pagination" className="flex flex-wrap items-center justify-center gap-3"><button onClick={()=>setPage(current=>Math.max(1,current-1))} disabled={page<=1||loading} className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50">Previous</button><span aria-live="polite" className="text-sm text-slate-600">Page {pagination.current_page} of {pagination.total_pages} · {pagination.total_articles} articles</span><button onClick={()=>setPage(current=>Math.min(pagination.total_pages,current+1))} disabled={page>=pagination.total_pages||loading} className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50">Next</button></nav></>}
  </div>;
}
