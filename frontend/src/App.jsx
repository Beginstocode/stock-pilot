
import { lazy, Suspense } from "react";
import { NavLink, Outlet } from "react-router-dom";
import {
  LayoutDashboard, ChartNoAxesCombined, BriefcaseBusiness,
  Newspaper, Search, Bell, ChevronDown, TrendingUp,
  ArrowUpRight, ArrowDownRight, Activity,
  Wallet, Star, ChartLine
} from "lucide-react";

const MarketChart = lazy(() => import("./MarketChart.jsx"));

const stocks = [
  { name: "Reliance Industries", symbol: "RELIANCE", price: "1,468.20", change: "+2.45%", up: true },
  { name: "Tata Consultancy", symbol: "TCS", price: "3,245.60", change: "+1.82%", up: true },
  { name: "Infosys Limited", symbol: "INFY", price: "1,520.40", change: "-0.65%", up: false },
  { name: "HDFC Bank", symbol: "HDFCBANK", price: "1,785.30", change: "+0.94%", up: true },
];

export function DashboardPage() {
  return (
    <>
      <header className="flex flex-wrap items-center justify-between gap-4 mb-9">
        <div>
          <p className="text-sm text-slate-500">Welcome back</p>
          <h2 className="text-2xl font-bold mt-1">Market Overview</h2>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center bg-white border border-slate-200 rounded-xl px-3 py-2">
            <Search size={17} className="text-slate-400" />
            <input placeholder="Search stocks..." className="outline-none text-sm ml-2 w-36" />
          </div>
          <button className="p-3 bg-white border border-slate-200 rounded-xl">
            <Bell size={18} />
          </button>
          <button className="flex items-center gap-2 bg-white border border-slate-200 rounded-xl p-2">
            <div className="w-8 h-8 bg-slate-900 text-white rounded-lg grid place-items-center text-sm">S</div>
            <span className="text-sm font-medium hidden sm:block">My Account</span>
            <ChevronDown size={15} />
          </button>
        </div>
      </header>

      <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-7">
        {[
          { label: "NIFTY 50", value: "25,100.00", change: "+1.24%", up: true, icon: TrendingUp },
          { label: "BANK NIFTY", value: "57,450.25", change: "+0.86%", up: true, icon: Wallet },
          { label: "INDIA VIX", value: "14.82", change: "-2.10%", up: false, icon: Activity },
          { label: "WATCHLIST", value: "12 Stocks", change: "4 active signals", up: true, icon: Star },
        ].map((item) => (
          <div key={item.label} className="bg-white border border-slate-200 rounded-2xl p-5">
            <div className="flex justify-between items-center">
              <p className="text-sm text-slate-500">{item.label}</p>
              <item.icon size={19} className="text-slate-400" />
            </div>
            <h3 className="text-2xl font-bold mt-4">{item.value}</h3>
            <p className={`text-sm mt-2 ${item.up ? "text-emerald-600" : "text-rose-600"}`}>
              {item.change}
            </p>
          </div>
        ))}
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-3 gap-5 mb-7">
        <div className="xl:col-span-2 bg-white border border-slate-200 rounded-2xl p-6">
          <div className="flex flex-wrap justify-between gap-3 mb-7">
            <div>
              <h3 className="font-bold text-lg">Market Performance</h3>
              <p className="text-sm text-slate-500 mt-1">NIFTY 50 · Illustrative chart</p>
            </div>
            <span className="text-emerald-600 text-sm font-semibold">+4.06% this week</span>
          </div>
          <div className="h-64">
            <Suspense fallback={null}><MarketChart /></Suspense>
          </div>
        </div>

        <div className="bg-slate-900 text-white rounded-2xl p-6">
          <div className="flex items-center gap-2 text-slate-300 text-sm"><Activity size={17} /> AI Market Signal</div>
          <h3 className="text-3xl font-bold mt-7">Market Watch</h3>
          <p className="text-slate-400 text-sm mt-2">AI-generated market insights will appear here.</p>
          <div className="border-t border-slate-700 mt-7 pt-5">
            <p className="text-xs text-slate-400">MODEL STATUS</p>
            <p className="text-emerald-400 font-semibold mt-2">Ready for analysis</p>
          </div>
        </div>
      </section>

      <section className="bg-white border border-slate-200 rounded-2xl p-6">
        <div className="flex justify-between items-center mb-6">
          <div><h3 className="font-bold text-lg">Popular Stocks</h3><p className="text-sm text-slate-500 mt-1">Your market watchlist</p></div>
          <button className="text-sm font-semibold text-slate-600">View all</button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-slate-400 border-b border-slate-100"><tr>
              <th className="pb-4 font-medium">Company</th><th className="pb-4 font-medium">Symbol</th>
              <th className="pb-4 font-medium">Price (₹)</th><th className="pb-4 font-medium">Change</th><th className="pb-4 font-medium">Trend</th>
            </tr></thead>
            <tbody>{stocks.map((stock) => (
              <tr key={stock.symbol} className="border-b last:border-0 border-slate-100">
                <td className="py-4 font-semibold">{stock.name}</td><td className="py-4 text-slate-500">{stock.symbol}</td>
                <td className="py-4 font-medium">{stock.price}</td>
                <td className={`py-4 font-semibold ${stock.up ? "text-emerald-600" : "text-rose-600"}`}>{stock.change}</td>
                <td className="py-4">{stock.up ? <ArrowUpRight className="text-emerald-600" size={19} /> : <ArrowDownRight className="text-rose-600" size={19} />}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>

      <footer className="text-center text-xs text-slate-400 mt-8">StockPilot · AI-powered Indian stock intelligence</footer>
    </>
  );
}

function App() {
  return (
    <div className="min-h-screen bg-[#f7f8fa] text-slate-900 flex">

      <aside className="w-64 bg-white border-r border-slate-200 p-6 hidden md:flex flex-col">
        <div className="flex items-center gap-3 mb-12">
          <div className="bg-slate-900 text-white p-2 rounded-xl">
            <Activity size={23} />
          </div>
          <div>
            <h1 className="font-bold text-xl tracking-tight">StockPilot</h1>
            <p className="text-xs text-slate-400">AI Market Intelligence</p>
          </div>
        </div>

        <p className="text-xs font-semibold text-slate-400 mb-4">WORKSPACE</p>

        <nav className="space-y-2">
          <NavLink to="/" end className={({ isActive }) => `flex items-center gap-3 rounded-xl p-3 ${isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}>
            <LayoutDashboard size={19} /> Dashboard
          </NavLink>
          <NavLink to="/stock-analysis" className={({ isActive }) => `flex items-center gap-3 rounded-xl p-3 ${isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}>
            <ChartNoAxesCombined size={19} /> Stock Analysis
          </NavLink>
          <NavLink to="/portfolio" className={({ isActive }) => `flex items-center gap-3 rounded-xl p-3 ${isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}>
            <BriefcaseBusiness size={19} /> Portfolio
          </NavLink>
          <NavLink to="/market-news" className={({ isActive }) => `flex items-center gap-3 rounded-xl p-3 ${isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}>
            <Newspaper size={19} /> Market News
          </NavLink>
          <NavLink to="/historical-performance" className={({ isActive }) => `flex items-center gap-3 rounded-xl p-3 ${isActive ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}>
            <ChartLine size={19} /> Historical Performance
          </NavLink>
        </nav>

        <div className="mt-auto bg-slate-50 rounded-2xl p-4">
          <p className="font-semibold text-sm">AI Insights</p>
          <p className="text-xs text-slate-500 mt-2">
            Your market intelligence workspace.
          </p>
        </div>
      </aside>

      <main className="flex-1 min-w-0 p-5 md:p-9">
        <nav aria-label="Mobile workspace navigation" className="mb-5 flex gap-2 overflow-x-auto md:hidden">
          {[
            ["/", "Dashboard"],
            ["/stock-analysis", "Stock Analysis"],
            ["/portfolio", "Portfolio"],
            ["/market-news", "Market News"],
            ["/historical-performance", "Historical Performance"],
          ].map(([to, label]) => <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `whitespace-nowrap rounded-lg px-3 py-2 text-xs font-semibold ${isActive ? "bg-slate-900 text-white" : "border border-slate-200 bg-white text-slate-600"}`}>{label}</NavLink>)}
        </nav>
        <Outlet />
      </main>
    </div>
  );
}

export default App;
