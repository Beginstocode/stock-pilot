import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { Activity, AlertCircle, BarChart3, Search, TrendingUp } from 'lucide-react';
import { getStockDetails, getStockIndicators, getStockPrediction, getSupportedStocks } from './api.js';

const StockPriceChart = lazy(() => import('./StockPriceChart.jsx'));

const indicatorGroups = [
  { title: 'RSI (14)', values: [{ label: 'RSI', key: 'rsi_14' }] },
  {
    title: 'MACD (12, 26, 9)',
    values: [
      { label: 'MACD', key: 'macd' },
      { label: 'Signal (9)', key: 'macd_signal' },
      { label: 'Histogram', key: 'macd_histogram' },
    ],
  },
  {
    title: 'Simple Moving Average',
    values: [
      { label: 'SMA 20', key: 'sma_20' },
      { label: 'SMA 50', key: 'sma_50' },
      { label: 'SMA 200', key: 'sma_200' },
    ],
  },
  {
    title: 'Exponential Moving Average',
    values: [
      { label: 'EMA 12', key: 'ema_12' },
      { label: 'EMA 26', key: 'ema_26' },
    ],
  },
];
const HISTORY_RANGES = {
  '1M': { unit: 'month', count: 1 },
  '3M': { unit: 'month', count: 3 },
  '6M': { unit: 'month', count: 6 },
  '1Y': { unit: 'year', count: 1 },
  '3Y': { unit: 'year', count: 3 },
  ALL: null,
};

function toApiDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function getHistoryDateRange(range) {
  const rangeConfig = HISTORY_RANGES[range];
  if (!rangeConfig) return {};

  const end = new Date();
  const start = new Date(end);
  if (rangeConfig.unit === 'month') start.setMonth(start.getMonth() - rangeConfig.count);
  else start.setFullYear(start.getFullYear() - rangeConfig.count);
  return { startDate: toApiDate(start), endDate: toApiDate(end) };
}

function PendingValue({ children }) {
  return (
    <p className="mt-2 text-sm text-slate-500">
      <span className="font-semibold text-amber-600">Pending</span>
      {children ? ` · ${children}` : ''}
    </p>
  );
}

function AnalysisError({ message }) {
  if (!message) return null;
  return (
    <div role="alert" className="mt-4 flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
      <AlertCircle size={18} className="mt-0.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

function latestIndicatorValue(records, key) {
  for (let index = records.length - 1; index >= 0; index -= 1) {
    const value = records[index][key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return { value, date: records[index].date };
    }
  }
  return null;
}

function PredictionCard({ model, prediction, state, error, recommendationStyle }) {
  return (
    <div className="rounded-xl border border-slate-100 p-4">
      <h4 className="text-sm font-semibold text-slate-700">Model {model}</h4>
      {state === 'loading' && <p className="mt-3 text-sm text-slate-500" role="status">Loading Model {model} prediction…</p>}
      {state === 'ready' && prediction && (
        <>
          <div className={`mt-3 inline-flex rounded-lg border px-3 py-1.5 text-sm font-bold ${recommendationStyle[prediction.prediction] || 'border-slate-200 bg-slate-50 text-slate-700'}`}>
            {prediction.prediction}
          </div>
          <p className="mt-3 text-2xl font-bold">
            {typeof prediction.confidence === 'number'
              ? new Intl.NumberFormat('en-IN', { style: 'percent', maximumFractionDigits: 1 }).format(prediction.confidence)
              : '—'}
          </p>
          <p className="text-sm text-slate-500">Confidence for the predicted class</p>
          <p className="mt-2 text-xs text-slate-400">
            {prediction.prediction_timestamp ? `Generated ${new Date(prediction.prediction_timestamp).toLocaleString()}` : 'Generation timestamp unavailable'}
          </p>
          <div className="mt-4 border-t border-slate-100 pt-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">BUY · HOLD · SELL probabilities</p>
            {prediction.probabilities ? (
              <div className="mt-3 space-y-3">
                {[
                  { label: 'BUY', color: 'bg-emerald-500', textColor: 'text-emerald-700' },
                  { label: 'HOLD', color: 'bg-amber-500', textColor: 'text-amber-700' },
                  { label: 'SELL', color: 'bg-rose-500', textColor: 'text-rose-700' },
                ].map(({ label, color, textColor }) => {
                  const probability = prediction.probabilities[label];
                  const available = typeof probability === 'number'
                    && Number.isFinite(probability)
                    && probability >= 0
                    && probability <= 1;
                  return (
                    <div key={label}>
                      <div className="mb-1 flex items-center justify-between text-xs">
                        <span className={`font-semibold ${textColor}`}>{label}</span>
                        <span className="font-medium text-slate-700">
                          {available
                            ? new Intl.NumberFormat('en-IN', { style: 'percent', maximumFractionDigits: 1 }).format(probability)
                            : 'Unavailable'}
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-slate-100" role="progressbar" aria-label={`Model ${model} ${label} probability`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={available ? probability * 100 : 0}>
                        <div className={`h-full rounded-full ${color}`} style={{ width: `${available ? probability * 100 : 0}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : <PendingValue>Class probabilities are not available from the prediction response.</PendingValue>}
          </div>
        </>
      )}
      {state === 'idle' && <PendingValue>Run an analysis to request this prediction.</PendingValue>}
      <AnalysisError message={error} />
    </div>
  );
}

function displayMarketDate(value) {
  if (!value) return 'Unavailable';
  const parsed = new Date(value);
  if (!Number.isFinite(parsed.getTime())) return 'Unavailable';
  return new Intl.DateTimeFormat('en-IN', { dateStyle: 'medium', timeZone: 'Asia/Kolkata' }).format(parsed);
}

export default function StockAnalysisPage() {
  const [symbols, setSymbols] = useState([]);
  const [catalogState, setCatalogState] = useState('loading');
  const [catalogError, setCatalogError] = useState('');
  const [searchError, setSearchError] = useState('');
  const [query, setQuery] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [details, setDetails] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionB, setPredictionB] = useState(null);
  const [indicators, setIndicators] = useState([]);
  const [historyRange, setHistoryRange] = useState('1Y');
  const [indicatorState, setIndicatorState] = useState('idle');
  const [indicatorError, setIndicatorError] = useState('');
  const [detailsState, setDetailsState] = useState('idle');
  const [predictionState, setPredictionState] = useState('idle');
  const [detailsError, setDetailsError] = useState('');
  const [predictionError, setPredictionError] = useState('');
  const [predictionBError, setPredictionBError] = useState('');
  const [predictionBState, setPredictionBState] = useState('idle');

  useEffect(() => {
    let isCurrent = true;
    getSupportedStocks()
      .then((result) => {
        if (!isCurrent) return;
        if (!Array.isArray(result)) throw new Error('The stock list response was not in the expected format.');
        setSymbols(result);
        setCatalogState(result.length ? 'ready' : 'empty');
      })
      .catch((error) => {
        if (!isCurrent) return;
        setCatalogError(error.message || 'Could not load supported stocks.');
        setCatalogState('error');
      });
    return () => { isCurrent = false; };
  }, []);

  useEffect(() => {
    if (!selectedSymbol) {
      setIndicators([]);
      setIndicatorState('idle');
      setIndicatorError('');
      return undefined;
    }

    let isCurrent = true;
    setIndicators([]);
    setIndicatorState('loading');
    setIndicatorError('');

    getStockIndicators(selectedSymbol, getHistoryDateRange(historyRange))
      .then((records) => {
        if (!isCurrent) return;
        if (!Array.isArray(records)) throw new Error('The indicator data response was not in the expected format.');

        const validRecords = records.filter((record) => (
          record
          && typeof record.date === 'string'
          && Number.isFinite(new Date(record.date).getTime())
          && typeof record.close === 'number'
          && Number.isFinite(record.close)
        ));
        setIndicators(validRecords);
        setIndicatorState(validRecords.length ? 'ready' : 'empty');
      })
      .catch((error) => {
        if (!isCurrent) return;
        if (error.message?.includes('No historical indicator data')) {
          setIndicators([]);
          setIndicatorState('empty');
          return;
        }
        setIndicatorError(error.message || 'Could not load historical indicators.');
        setIndicatorState('error');
      });

    return () => { isCurrent = false; };
  }, [selectedSymbol, historyRange]);

  const filteredSymbols = useMemo(() => {
    const normalizedQuery = query.trim().toUpperCase();
    if (!normalizedQuery) return symbols;
    return symbols.filter((symbol) => symbol.includes(normalizedQuery));
  }, [query, symbols]);

  async function analyzeStock(event) {
    event.preventDefault();
    const symbol = query.trim().toUpperCase();
    if (!symbols.includes(symbol)) {
      setSearchError('Choose a symbol from the supported stock list.');
      setSelectedSymbol('');
      setDetails(null);
      setPrediction(null);
      setPredictionB(null);
      setDetailsError('');
      setDetailsState('idle');
      setPredictionError('');
      setPredictionState('idle');
      setPredictionBError('');
      setPredictionBState('idle');
      return;
    }

    setSearchError('');
    setSelectedSymbol(symbol);
    setDetails(null);
    setPrediction(null);
    setPredictionB(null);
    setDetailsError('');
    setPredictionError('');
    setPredictionBError('');
    setDetailsState('loading');
    setPredictionState('loading');
    setPredictionBState('loading');

    const [detailsResult, predictionResult, predictionBResult] = await Promise.allSettled([
      getStockDetails(symbol),
      getStockPrediction(symbol, 'A'),
      getStockPrediction(symbol, 'B'),
    ]);

    if (detailsResult.status === 'fulfilled') {
      setDetails(detailsResult.value);
      setDetailsState('ready');
    } else {
      setDetailsError(detailsResult.reason.message || 'Could not load stock details.');
      setDetailsState('error');
    }

    if (predictionResult.status === 'fulfilled') {
      setPrediction(predictionResult.value);
      setPredictionState('ready');
    } else {
      setPredictionError(predictionResult.reason.message || 'Could not load a prediction.');
      setPredictionState('error');
    }

    if (predictionBResult.status === 'fulfilled') {
      setPredictionB(predictionBResult.value);
      setPredictionBState('ready');
    } else {
      setPredictionBError(predictionBResult.reason.message || 'Could not load Model B prediction.');
      setPredictionBState('error');
    }
  }

  const recommendationStyle = {
    BUY: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    HOLD: 'bg-amber-50 text-amber-700 border-amber-200',
    SELL: 'bg-rose-50 text-rose-700 border-rose-200',
  };

  return (
    <div>
      <header className="mb-7 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-slate-500">Research workspace</p>
          <h2 className="mt-1 text-2xl font-bold">Stock Analysis</h2>
          <p className="mt-1 text-sm text-slate-500">Explore supported Indian stocks and the available model signal.</p>
        </div>
        {selectedSymbol && (
          <div className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm">
            <span className="text-slate-500">Selected</span>
            <span className="ml-2 font-semibold">{selectedSymbol}</span>
          </div>
        )}
      </header>

      <section className="mb-6 rounded-2xl border border-slate-200 bg-white p-5">
        <form onSubmit={analyzeStock} className="flex flex-col gap-3 sm:flex-row">
          <label className="relative min-w-0 flex-1">
            <span className="sr-only">Search supported stocks</span>
            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              list="stock-symbols"
              value={query}
              onChange={(event) => setQuery(event.target.value.toUpperCase())}
              placeholder={catalogState === 'loading' ? 'Loading supported stocks…' : 'Search by symbol, e.g. RELIANCE'}
              disabled={catalogState === 'loading' || catalogState === 'error' || catalogState === 'empty'}
              className="w-full rounded-xl border border-slate-200 bg-white py-3 pl-10 pr-4 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-100 disabled:bg-slate-50"
            />
            <datalist id="stock-symbols">
              {filteredSymbols.map((symbol) => <option key={symbol} value={symbol} />)}
            </datalist>
          </label>
          <button
            type="submit"
            disabled={catalogState !== 'ready'}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <BarChart3 size={17} /> Analyze stock
          </button>
        </form>
        <AnalysisError message={searchError} />
        <p className="mt-3 text-xs text-slate-500">
          {catalogState === 'loading' && 'Loading the supported symbol list…'}
          {catalogState === 'ready' && `${symbols.length} symbols available from the StockPilot API.`}
          {catalogState === 'empty' && 'The API returned no supported symbols.'}
        </p>
        <AnalysisError message={catalogError} />
      </section>

      {!selectedSymbol && catalogState === 'ready' && (
        <div className="mb-6 rounded-2xl border border-dashed border-slate-300 bg-white p-8 text-center">
          <Search size={24} className="mx-auto text-slate-400" />
          <h3 className="mt-3 font-semibold">Choose a stock to begin</h3>
          <p className="mt-1 text-sm text-slate-500">Search for a supported symbol to load its details and prediction.</p>
        </div>
      )}

      {selectedSymbol && (
        <>
          <section className="mb-6 grid grid-cols-1 gap-4 xl:grid-cols-3">
            <article className="rounded-2xl border border-slate-200 bg-white p-5 xl:col-span-2">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="font-bold">{details?.name || selectedSymbol}</h3>
                  <p className="mt-1 text-sm text-slate-500">Historical price and technical indicators</p>
                  <p className="mt-1 text-xs text-slate-500" data-testid="market-data-date">
                    Actual market-data date: {displayMarketDate(prediction?.prediction_date || predictionB?.prediction_date || indicators.at(-1)?.date)}
                  </p>
                </div>
                <TrendingUp size={19} className="text-slate-400" />
              </div>
              <Suspense fallback={<div className="mt-4 grid h-64 place-items-center rounded-xl bg-slate-50 text-sm text-slate-500" role="status">Preparing chart…</div>}>
                <StockPriceChart
                  data={indicators}
                  range={historyRange}
                  onRangeChange={setHistoryRange}
                  status={indicatorState}
                  error={indicatorError}
                />
              </Suspense>
            </article>

            <article className="rounded-2xl border border-slate-200 bg-white p-5">
              <div className="flex items-center gap-2 text-sm text-slate-500"><Activity size={17} /> XGBoost prediction</div>
              <div className="mt-4 space-y-3">
                <PredictionCard model="A" prediction={prediction} state={predictionState} error={predictionError} recommendationStyle={recommendationStyle} />
                <PredictionCard model="B" prediction={predictionB} state={predictionBState} error={predictionBError} recommendationStyle={recommendationStyle} />
              </div>
            </article>
          </section>

          <section className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {indicatorState === 'error' && (
              <div className="sm:col-span-2 xl:col-span-4"><AnalysisError message={indicatorError} /></div>
            )}
            {indicatorGroups.map((group) => (
              <article key={group.title} className="rounded-2xl border border-slate-200 bg-white p-5">
                <h3 className="text-sm font-semibold text-slate-600">{group.title}</h3>
                {indicatorState === 'loading' && <p className="mt-3 text-sm text-slate-500" role="status">Loading indicators…</p>}
                {indicatorState === 'empty' && <PendingValue>No processed indicator rows are available for this range.</PendingValue>}
                {indicatorState === 'ready' && (
                  <dl className="mt-3 space-y-2">
                    {group.values.map((item) => {
                      const latest = latestIndicatorValue(indicators, item.key);
                      return (
                        <div key={item.key} className="flex items-baseline justify-between gap-2 text-sm">
                          <dt className="text-slate-500">{item.label}</dt>
                          <dd className="text-right font-semibold text-slate-900">
                            {latest ? latest.value.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : <span className="font-normal text-amber-600">Insufficient data</span>}
                          </dd>
                        </div>
                      );
                    })}
                  </dl>
                )}
              </article>
            ))}
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <h3 className="font-bold">Stock details</h3>
            {detailsState === 'loading' && <p className="mt-3 text-sm text-slate-500" role="status">Loading stock details…</p>}
            {detailsState === 'ready' && details && (
              <dl className="mt-4 grid grid-cols-1 gap-4 text-sm sm:grid-cols-2">
                <div><dt className="text-slate-500">Company</dt><dd className="mt-1 font-medium">{details.name}</dd></div>
                <div><dt className="text-slate-500">Symbol</dt><dd className="mt-1 font-medium">{details.symbol}</dd></div>
              </dl>
            )}
            {detailsState === 'idle' && <PendingValue>Choose a supported symbol to load stock details.</PendingValue>}
            <AnalysisError message={detailsError} />
          </section>
        </>
      )}
    </div>
  );
}
