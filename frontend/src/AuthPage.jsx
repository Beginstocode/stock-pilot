import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Activity } from 'lucide-react';
import { registerAccount, signIn } from './api.js';

export default function AuthPage({ mode }) {
  const signup = mode === 'signup'; const navigate = useNavigate();
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' });
  const [error, setError] = useState(''); const [success, setSuccess] = useState(''); const [loading, setLoading] = useState(false);
  const change = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }));
  async function submit(event) {
    event.preventDefault(); setError(''); setSuccess('');
    if (signup && form.password !== form.confirm) { setError('Passwords do not match.'); return; }
    setLoading(true);
    try {
      if (signup) {
        await registerAccount({ name: form.name, email: form.email, password: form.password });
        setSuccess('Account created. Sign in with your new account.');
        setTimeout(() => navigate('/signin', { replace: true }), 900);
      } else {
        const result = await signIn({ email: form.email, password: form.password });
        localStorage.setItem('access_token', result.access_token);
        navigate('/portfolio', { replace: true });
      }
    } catch (err) { setError(err.message || 'Authentication request failed.'); }
    finally { setLoading(false); }
  }
  return <main className="grid min-h-screen place-items-center bg-[#f7f8fa] p-5 text-slate-900"><section className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-7 shadow-sm"><div className="mb-6 flex items-center gap-3"><span className="rounded-xl bg-slate-900 p-2 text-white"><Activity size={22}/></span><div><h1 className="font-bold">StockPilot</h1><p className="text-xs text-slate-500">AI Market Intelligence</p></div></div><h2 className="text-2xl font-bold">{signup?'Create your account':'Welcome back'}</h2><p className="mt-1 text-sm text-slate-500">{signup?'Sign up to manage your portfolio and watch stocks.':'Sign in to continue to StockPilot.'}</p>
    <form onSubmit={submit} className="mt-6 space-y-4">{signup&&<label className="block text-sm font-medium">Full name<input required maxLength={150} autoComplete="name" value={form.name} onChange={change('name')} className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-normal"/></label>}<label className="block text-sm font-medium">Email address<input required type="email" autoComplete="email" value={form.email} onChange={change('email')} className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-normal"/></label><label className="block text-sm font-medium">Password<input required type="password" minLength={12} maxLength={128} autoComplete={signup?'new-password':'current-password'} value={form.password} onChange={change('password')} className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-normal"/>{signup&&<span className="mt-1 block text-xs text-slate-500">Use at least 12 characters.</span>}</label>{signup&&<label className="block text-sm font-medium">Confirm password<input required type="password" autoComplete="new-password" value={form.confirm} onChange={change('confirm')} className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-normal"/></label>}{error&&<p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{error}</p>}{success&&<p role="status" className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-700">{success}</p>}<button disabled={loading} className="w-full rounded-xl bg-slate-900 p-3 font-semibold text-white disabled:opacity-60">{loading?'Please wait…':signup?'Create Account':'Sign In'}</button></form>
    <p className="mt-5 text-center text-sm text-slate-500">{signup?'Already have an account?':'New to StockPilot?'} <Link className="font-semibold text-slate-900 underline" to={signup?'/signin':'/signup'}>{signup?'Sign In':'Create account'}</Link></p><p className="mt-5 text-center"><Link to="/" className="text-xs text-slate-500 hover:text-slate-900">Back to dashboard</Link></p></section></main>;
}
