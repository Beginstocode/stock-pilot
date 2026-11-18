const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

async function request(path, options = {}) {
  const token = localStorage.getItem('access_token');
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      // Keep the status based message when the response is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

export const getSupportedStocks = () => request('/stocks');
export const getStockDetails = (symbol) => request(`/stocks/${encodeURIComponent(symbol)}`);
export const getStockPrediction = (symbol, model = 'A') => request(
  `/predict/${encodeURIComponent(symbol)}?model=${encodeURIComponent(model)}`,
);
export const getStockHistory = (symbol, { startDate, endDate } = {}) => {
  const params = new URLSearchParams();
  if (startDate) params.set('start_date', startDate);
  if (endDate) params.set('end_date', endDate);
  const query = params.toString();
  return request(`/stocks/${encodeURIComponent(symbol)}/history${query ? `?${query}` : ''}`);
};
export const getStockIndicators = (symbol, { startDate, endDate } = {}) => {
  const params = new URLSearchParams();
  if (startDate) params.set('start_date', startDate);
  if (endDate) params.set('end_date', endDate);
  const query = params.toString();
  return request(`/stocks/${encodeURIComponent(symbol)}/indicators${query ? `?${query}` : ''}`);
};

export const getPortfolios = () => request('/portfolio');
export const createPortfolio = (portfolio_name) => request('/portfolio', { method: 'POST', body: JSON.stringify({ portfolio_name }) });
export const getPortfolioHoldings = (portfolioId) => request(`/portfolio/holdings?portfolio_id=${encodeURIComponent(portfolioId)}`);
export const addPortfolioHolding = (data) => request('/portfolio/holdings', { method: 'POST', body: JSON.stringify(data) });
export const updatePortfolioHolding = (id, data) => request(`/portfolio/holdings/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(data) });
export const deletePortfolioHolding = (id) => request(`/portfolio/holdings/${encodeURIComponent(id)}`, { method: 'DELETE' });
export const registerAccount = (data) => request('/auth/register', { method: 'POST', body: JSON.stringify(data) });
export const signIn = (data) => request('/auth/login', { method: 'POST', body: JSON.stringify(data) });
export const getNews = (params = {}) => {
  const query = new URLSearchParams(Object.entries(params).filter(([, value]) => value));
  return request(`/news${query.size ? `?${query}` : ''}`);
};
export const getNewsImpact = (article_url) => request('/news/impact', {
  method: 'POST',
  body: JSON.stringify({ article_url }),
});
export const getHistoricalNewsPerformance = () => request('/news/performance');
export const getHistoricalPerformanceDashboard = (filters = {}) => {
  const params = new URLSearchParams(
    Object.entries(filters).filter(([, value]) => value !== '' && value != null),
  );
  return request(`/news/historical-performance${params.size ? `?${params}` : ''}`);
};
