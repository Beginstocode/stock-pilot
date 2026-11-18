export default function PlaceholderPage({ title }) {
  return (
    <section className="bg-white border border-slate-200 rounded-2xl p-6">
      <h2 className="text-2xl font-bold">{title}</h2>
      <div className="mt-8 rounded-xl border border-dashed border-slate-200 bg-slate-50 px-6 py-12 text-center">
        <p className="text-slate-500">Coming next</p>
      </div>
    </section>
  );
}
