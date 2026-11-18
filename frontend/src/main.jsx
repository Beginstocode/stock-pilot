import { StrictMode, lazy, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import './index.css'
import App, { DashboardPage } from './App.jsx'
import PlaceholderPage from './PlaceholderPage.jsx'
import StockAnalysisPage from './StockAnalysisPage.jsx'
import PortfolioPage from './PortfolioPage.jsx'
import AuthPage from './AuthPage.jsx'
import MarketNewsPage from './MarketNewsPage.jsx'
const HistoricalPerformancePage = lazy(() => import('./HistoricalPerformancePage.jsx'))

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/signup" element={<AuthPage mode="signup" />} />
        <Route path="/signin" element={<AuthPage mode="signin" />} />
        <Route element={<App />}>
          <Route index element={<DashboardPage />} />
          <Route path="stock-analysis" element={<StockAnalysisPage />} />
          <Route path="portfolio" element={<PortfolioPage />} />
          <Route path="market-news" element={<MarketNewsPage />} />
          <Route path="historical-performance" element={<Suspense fallback={<div role="status" className="p-8 text-center text-sm text-slate-600">Loading performance dashboard…</div>}><HistoricalPerformancePage /></Suspense>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
