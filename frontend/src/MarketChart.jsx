import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const chartData = [
  { day: "Mon", price: 24120 },
  { day: "Tue", price: 24380 },
  { day: "Wed", price: 24250 },
  { day: "Thu", price: 24600 },
  { day: "Fri", price: 24480 },
  { day: "Sat", price: 24820 },
  { day: "Sun", price: 25100 },
];

export default function MarketChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={chartData}>
        <defs>
          <linearGradient id="marketFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0f766e" stopOpacity={0.2} />
            <stop offset="95%" stopColor="#0f766e" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#edf0f3" />
        <XAxis dataKey="day" axisLine={false} tickLine={false} />
        <YAxis domain={["dataMin - 300", "dataMax + 300"]} hide />
        <Tooltip />
        <Area type="monotone" dataKey="price" stroke="#0f766e" strokeWidth={2.5} fill="url(#marketFill)" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
