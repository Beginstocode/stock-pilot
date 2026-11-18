import { useState } from 'react';
import { Activity } from 'lucide-react';
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const RANGE_OPTIONS = [
  { label: '1M', value: '1M' },
  { label: '3M', value: '3M' },
  { label: '6M', value: '6M' },
  { label: '1Y', value: '1Y' },
  { label: '3Y', value: '3Y' },
  { label: 'All', value: 'ALL' },
];
const CHART_MODES = [
  { label: 'Price & averages', value: 'price' },
  { label: 'RSI', value: 'rsi' },
  { label: 'MACD', value: 'macd' },
];

function formatDate(value) {
  return new Date(value).toLocaleDateString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: '2-digit',
  });
}

function formatPrice(value) {
  return `₹${Number(value).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
}

function formatIndicator(value) {
  return Number(value).toLocaleString('en-IN', { maximumFractionDigits: 2 });
}

export default function StockPriceChart({
  data,
  range,
  onRangeChange,
  status,
  error,
}) {
  const [chartMode, setChartMode] = useState('price');
  const isPriceMode = chartMode === 'price';
  const isRsiMode = chartMode === 'rsi';

  return (
    <>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-3">
          <div className="flex flex-wrap gap-1 rounded-xl bg-slate-100 p-1" aria-label="Historical chart time range">
            {RANGE_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                aria-pressed={range === option.value}
                onClick={() => onRangeChange(option.value)}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${range === option.value ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
              >
                {option.label}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap gap-1 rounded-xl bg-slate-100 p-1" aria-label="Chart indicator view">
            {CHART_MODES.map((mode) => (
              <button
                key={mode.value}
                type="button"
                aria-pressed={chartMode === mode.value}
                onClick={() => setChartMode(mode.value)}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${chartMode === mode.value ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}
              >
                {mode.label}
              </button>
            ))}
          </div>
        </div>
        {status === 'ready' && data.length > 0 && (
          <p className="text-xs text-slate-500">{data.length.toLocaleString('en-IN')} trading days</p>
        )}
      </div>

      <div className="mt-4 h-64 w-full">
        {status === 'loading' && (
          <div className="grid h-full place-items-center rounded-xl bg-slate-50 text-sm text-slate-500" role="status">
            Loading historical prices and indicators…
          </div>
        )}
        {status === 'error' && (
          <div className="grid h-full place-items-center rounded-xl border border-rose-200 bg-rose-50 px-5 text-center text-sm text-rose-700" role="alert">
            {error || 'Could not load historical stock data.'}
          </div>
        )}
        {status === 'empty' && (
          <div className="grid h-full place-items-center rounded-xl border border-dashed border-slate-200 bg-slate-50 px-5 text-center">
            <div>
              <Activity size={24} className="mx-auto text-slate-400" />
              <p className="mt-2 font-medium text-slate-700">No historical stock data for this range</p>
              <p className="mt-1 text-sm text-slate-500">Try a wider time range.</p>
            </div>
          </div>
        )}
        {status === 'ready' && data.length > 0 && (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#edf0f3" />
              <XAxis
                dataKey="date"
                axisLine={false}
                tickLine={false}
                tickFormatter={formatDate}
                minTickGap={32}
                tick={{ fill: '#94a3b8', fontSize: 11 }}
              />
              <YAxis
                yAxisId="left"
                axisLine={false}
                tickLine={false}
                width={isPriceMode ? 72 : 48}
                tickFormatter={isPriceMode ? formatPrice : formatIndicator}
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                domain={isRsiMode ? [0, 100] : ['auto', 'auto']}
              />
              {!isPriceMode && !isRsiMode && (
                <ReferenceLine yAxisId="left" y={0} stroke="#94a3b8" strokeDasharray="3 3" />
              )}
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleDateString('en-IN', {
                  day: 'numeric', month: 'long', year: 'numeric',
                })}
                formatter={(value, name) => [
                  value == null ? 'Insufficient data' : (isPriceMode && name !== 'RSI (14)' ? formatPrice(value) : formatIndicator(value)),
                  name,
                ]}
                contentStyle={{ borderRadius: 12, borderColor: '#e2e8f0', fontSize: 12 }}
              />
              {isPriceMode && <>
                <Line yAxisId="left" type="monotone" dataKey="close" name="Close" stroke="#0f766e" strokeWidth={2} dot={false} activeDot={{ r: 4, strokeWidth: 0 }} connectNulls={false} />
                <Line yAxisId="left" type="monotone" dataKey="sma_20" name="SMA 20" stroke="#2563eb" strokeWidth={1.5} dot={false} connectNulls={false} />
                <Line yAxisId="left" type="monotone" dataKey="sma_50" name="SMA 50" stroke="#d97706" strokeWidth={1.5} dot={false} connectNulls={false} />
                <Line yAxisId="left" type="monotone" dataKey="sma_200" name="SMA 200" stroke="#64748b" strokeWidth={1.5} dot={false} connectNulls={false} />
                <Line yAxisId="left" type="monotone" dataKey="ema_12" name="EMA 12" stroke="#059669" strokeWidth={1.5} dot={false} connectNulls={false} />
                <Line yAxisId="left" type="monotone" dataKey="ema_26" name="EMA 26" stroke="#9333ea" strokeWidth={1.5} dot={false} connectNulls={false} />
              </>}
              {isRsiMode && <>
                <ReferenceLine yAxisId="left" y={70} stroke="#f59e0b" strokeDasharray="4 4" />
                <ReferenceLine yAxisId="left" y={30} stroke="#f59e0b" strokeDasharray="4 4" />
                <Line yAxisId="left" type="monotone" dataKey="rsi_14" name="RSI (14)" stroke="#7c3aed" strokeWidth={2} dot={false} connectNulls={false} />
              </>}
              {!isPriceMode && !isRsiMode && <>
                <Bar yAxisId="left" dataKey="macd_histogram" name="MACD Histogram" fill="#94a3b8" />
                <Line yAxisId="left" type="monotone" dataKey="macd" name="MACD" stroke="#0f766e" strokeWidth={2} dot={false} connectNulls={false} />
                <Line yAxisId="left" type="monotone" dataKey="macd_signal" name="Signal (9)" stroke="#ea580c" strokeWidth={1.5} dot={false} connectNulls={false} />
              </>}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </>
  );
}
