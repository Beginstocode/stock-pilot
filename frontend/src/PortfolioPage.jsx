import { useEffect, useMemo, useState } from 'react';
import { BriefcaseBusiness, Plus, Pencil, Trash2, X } from 'lucide-react';
import { addPortfolioHolding, createPortfolio, deletePortfolioHolding, getPortfolios, getPortfolioHoldings, getSupportedStocks, updatePortfolioHolding } from './api.js';

const money = (value) => value == null ? '—' : new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 }).format(value);
const initialForm = { symbol: '', quantity: '', purchase_price: '' };

export default function PortfolioPage() {
  const [token, setToken] = useState(localStorage.getItem('access_token') || '');
  const [email, setEmail] = useState(''); const [password, setPassword] = useState('');
  const [stocks, setStocks] = useState([]); const [portfolios, setPortfolios] = useState([]);
  const [portfolioId, setPortfolioId] = useState(''); const [summary, setSummary] = useState(null);
  const [form, setForm] = useState(initialForm); const [editing, setEditing] = useState(null);
  const [loading, setLoading] = useState(false); const [error, setError] = useState('');
  async function load() {
    setLoading(true); setError('');
    try {
      const [catalog, list] = await Promise.all([getSupportedStocks(), getPortfolios()]); setStocks(catalog); setPortfolios(list);
      let selected = list[0];
      if (!selected) selected = await createPortfolio('My Portfolio');
      setPortfolios((items) => items.length ? items : [selected]); setPortfolioId(String(selected.portfolio_id));
      setSummary(await getPortfolioHoldings(selected.portfolio_id));
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  }
  useEffect(() => { if (token) load(); }, [token]);
  async function login(e) {
    e.preventDefault(); setLoading(true); setError('');
    try { const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || '/api'}/auth/login`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) }); const body = await response.json(); if (!response.ok) throw new Error(body.detail || 'Login failed'); localStorage.setItem('access_token', body.access_token); setToken(body.access_token); }
    catch (e2) { setError(e2.message); } finally { setLoading(false); }
  }
  async function submit(e) {
    e.preventDefault(); setError('');
    const data = { quantity: Number(form.quantity), purchase_price: Number(form.purchase_price) };
    try {
      if (editing) await updatePortfolioHolding(editing.holding_id, data);
      else await addPortfolioHolding({ ...data, symbol: form.symbol, portfolio_id: Number(portfolioId) });
      setForm(initialForm); setEditing(null); await load();
    } catch (e2) { setError(e2.message); }
  }
  async function remove(id) { try { await deletePortfolioHolding(id); await load(); } catch (e) { setError(e.message); } }
  const allocations = useMemo(() => summary?.holdings || [], [summary]);
  if (!token) return <section className="mx-auto mt-12 max-w-md rounded-2xl border border-slate-200 bg-white p-7"><h2 className="text-xl font-bold">Sign in to view your portfolio</h2><p className="mt-2 text-sm text-slate-500">Portfolio data is private and requires your account.</p><form onSubmit={login} className="mt-5 space-y-3"><input required type="email" value={email} onChange={e=>setEmail(e.target.value)} placeholder="Email" className="w-full rounded-xl border p-3"/><input required type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="Password" className="w-full rounded-xl border p-3"/><button className="w-full rounded-xl bg-slate-900 p-3 font-semibold text-white">Sign in</button></form>{error&&<p role="alert" className="mt-3 text-sm text-rose-600">{error}</p>}</section>;
  return <div className="space-y-6">
    <header className="flex items-center justify-between"><div><p className="text-sm text-slate-500">Your investments</p><h1 className="mt-1 text-2xl font-bold">Portfolio</h1></div><div className="flex gap-2"><button disabled={loading} onClick={load} className="rounded-lg border px-3 py-2 text-sm disabled:opacity-50">{loading?'Refreshing…':'Refresh prices'}</button><button onClick={()=>{localStorage.removeItem('access_token');setToken('');}} className="rounded-lg border px-3 py-2 text-sm">Sign out</button></div></header>
    {error&&<div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">{error}</div>}
    {loading&&!summary&&<p role="status" className="text-slate-500">Loading portfolio…</p>}
    {summary&&<><section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{[['Invested value',money(summary.total_investment)],['Current value',money(summary.current_value)],['Profit / loss',money(summary.profit_loss)],['Return',summary.profit_loss==null||!summary.total_investment?'—':`${(summary.profit_loss/summary.total_investment*100).toFixed(2)}%`]].map(([label,value])=><article key={label} className="rounded-2xl border border-slate-200 bg-white p-5"><p className="text-sm text-slate-500">{label}</p><p className="mt-3 text-xl font-bold">{value}</p></article>)}</section>
    <section className="grid gap-5 xl:grid-cols-3"><article className="rounded-2xl border bg-white p-5 xl:col-span-2"><h2 className="mb-4 font-bold">Holdings</h2>{summary.holdings.length===0?<div className="py-12 text-center text-slate-500"><BriefcaseBusiness className="mx-auto mb-3"/><p>Your portfolio is empty. Add your first holding below.</p></div>:<div className="overflow-x-auto"><table className="w-full min-w-[900px] text-left text-sm"><thead className="border-b text-slate-500"><tr>{['Stock','Qty','Avg. price','Invested','Latest available closing price','Current value','P&L','Return',''].map(x=><th key={x} className="pb-3 pr-3">{x}</th>)}</tr></thead><tbody>{summary.holdings.map(h=>{const pct=h.profit_loss!=null&&h.total_investment>0?h.profit_loss/h.total_investment*100:null;return <tr key={h.holding_id} className="border-b last:border-0"><td className="py-3 font-semibold">{h.symbol}</td><td>{h.quantity}</td><td>{money(h.purchase_price)}</td><td>{money(h.total_investment)}</td><td><div>{h.current_price==null?'Price unavailable':money(h.current_price)}</div>{h.price_timestamp&&<div className="text-xs text-slate-500">{new Date(h.price_timestamp).toLocaleDateString()}</div>}</td><td>{money(h.current_value)}</td><td className={h.profit_loss>=0?'text-emerald-700':'text-rose-700'}>{money(h.profit_loss)}</td><td>{pct==null?'—':`${pct.toFixed(2)}%`}</td><td className="whitespace-nowrap"><button title="Edit" onClick={()=>{setEditing(h);setForm({symbol:h.symbol,quantity:String(h.quantity),purchase_price:String(h.purchase_price)});}} className="mr-2 rounded p-1 hover:bg-slate-100"><Pencil size={16}/></button><button title="Delete" onClick={()=>remove(h.holding_id)} className="rounded p-1 text-rose-600 hover:bg-rose-50"><Trash2 size={16}/></button></td></tr>})}</tbody></table></div>}</article>
    <article className="rounded-2xl border bg-white p-5"><h2 className="font-bold">Allocation by current value</h2>{allocations.length? <div className="mt-5 space-y-4">{allocations.map(h=>{const total=allocations.reduce((sum,item)=>sum+(item.current_value||0),0);const pct=h.current_value!=null&&total>0?h.current_value/total*100:null;return <div key={h.holding_id}><div className="mb-1 flex justify-between text-sm"><span>{h.symbol}</span><span>{pct==null?'Unavailable':`${pct.toFixed(1)}%`}</span></div><div className="h-2 rounded-full bg-slate-100"><div className="h-full rounded-full bg-slate-800" style={{width:`${pct||0}%`}}/></div></div>})}</div>:<p className="mt-5 text-sm text-slate-500">Add holdings to see allocation.</p>}</article></section>
    <section className="rounded-2xl border bg-white p-5"><h2 className="mb-4 font-bold">{editing?'Edit holding':'Add a holding'}</h2><form onSubmit={submit} className="grid gap-3 sm:grid-cols-4"><select required disabled={!!editing} value={form.symbol} onChange={e=>setForm({...form,symbol:e.target.value})} className="rounded-xl border p-3"><option value="">Select stock</option>{stocks.map(s=><option key={s} value={s}>{s}</option>)}</select><input required min="1" type="number" value={form.quantity} onChange={e=>setForm({...form,quantity:e.target.value})} placeholder="Quantity" className="rounded-xl border p-3"/><input required min="0.01" step="0.01" type="number" value={form.purchase_price} onChange={e=>setForm({...form,purchase_price:e.target.value})} placeholder="Average purchase price (₹)" className="rounded-xl border p-3"/><div className="flex gap-2"><button className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-slate-900 p-3 font-semibold text-white"><Plus size={17}/>{editing?'Save changes':'Add holding'}</button>{editing&&<button type="button" onClick={()=>{setEditing(null);setForm(initialForm)}} className="rounded-xl border px-3"><X size={17}/></button>}</div></form></section></>}
  </div>;
}
