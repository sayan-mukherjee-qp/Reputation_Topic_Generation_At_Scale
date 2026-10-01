import { useEffect } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../lib/api";
import { fmtDate, fmtGrowth, fmtInt, fmtNum, shortDay } from "../lib/format";
import { useFetch } from "../lib/hooks";
import type { RunOverview, Topic } from "../lib/types";
import { EMG, EST, isEmerging } from "./RunCharts";
import { AlertChip, AXIS, LifecycleChip, TooltipBox } from "./ui";

export function TopicDrawer({ dataset, run, topic, data, onClose, onViewRecords }: {
  dataset: string; run: string; topic: Topic; data: RunOverview; onClose: () => void; onViewRecords: () => void;
}) {
  const samples = useFetch(() => api.samples(dataset, run, topic.topic_id), [dataset, run, topic.topic_id]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const counts = data.series.counts[topic.topic_id] ?? [];
  const rows = data.series.dates.map((d, i) => ({ date: d, count: counts[i] ?? 0 }));
  const color = isEmerging(topic) ? EMG.color : EST.color;
  const keywords = (topic.keywords ?? "").split("/").map((k) => k.trim()).filter(Boolean);

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={topic.label}>
        <div className="row" style={{ alignItems: "flex-start" }}>
          <div className="grow">
            <div className="muted" style={{ fontSize: 12 }}>{topic.brand} · {topic.topic_id}</div>
            <h2 style={{ fontSize: 18, marginTop: 2 }}>{topic.label}</h2>
          </div>
          <button className="btn btn-sm" onClick={onViewRecords}>View all records</button>
          <button className="btn btn-sm" onClick={onClose} aria-label="Close">Close</button>
        </div>
        <div className="row wrap">
          <LifecycleChip lifecycle={topic.status_lifecycle} isNew={topic.is_new} />
          <AlertChip status={topic.status} />
          {topic.suppressed && <span className="chip">suppressed: {topic.suppress_reason?.replaceAll(";", ", ")}</span>}
          {topic.label_source && <span className="chip">label: {topic.label_source}</span>}
        </div>
        {topic.description && <p className="ink2">{topic.description}</p>}
        <div className="kv">
          <div><div className="k">Size (records)</div><div className="v">{fmtInt(topic.size)}</div></div>
          <div><div className="k">Recent 7-day volume</div><div className="v">{fmtInt(topic.recent_volume)}</div></div>
          <div><div className="k">Baseline volume</div><div className="v">{fmtNum(topic.baseline_volume, 1)}</div></div>
          <div><div className="k">Growth vs baseline</div><div className="v">{fmtGrowth(topic.growth_rate)}</div></div>
          <div><div className="k">Velocity</div><div className="v">{fmtNum(topic.velocity_ratio)}</div></div>
          <div><div className="k">Hot score</div><div className="v">{fmtNum(topic.hot_score)}</div></div>
          <div><div className="k">Coherence (NPMI)</div><div className="v">{fmtNum(topic.coherence, 3)}</div></div>
          <div><div className="k">First seen</div><div className="v">{fmtDate(topic.created_at, true)}</div></div>
          <div><div className="k">Last seen</div><div className="v">{fmtDate(topic.last_seen_at, true)}</div></div>
        </div>
        <section>
          <h3 style={{ marginBottom: 6 }}>Daily volume</h3>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={rows} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              <XAxis dataKey="date" {...AXIS} tickFormatter={shortDay} minTickGap={28} />
              <YAxis {...AXIS} axisLine={false} width={36} allowDecimals={false} />
              <Tooltip cursor={{ stroke: "var(--axis)" }}
                content={({ active, payload, label }) => active && payload?.length ? (
                  <TooltipBox title={shortDay(String(label))} rows={[{ name: "records", value: fmtInt(Number(payload[0].value)), color }]} />
                ) : null} />
              <Area dataKey="count" stroke={color} strokeWidth={2} fill={color} fillOpacity={0.1} isAnimationActive={false}
                activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2, fill: color }} />
            </AreaChart>
          </ResponsiveContainer>
        </section>
        {keywords.length > 0 && (
          <section>
            <h3 style={{ marginBottom: 6 }}>Keywords</h3>
            <div className="row wrap">{keywords.map((k) => <span className="chip" key={k}>{k}</span>)}</div>
          </section>
        )}
        <section>
          <h3 style={{ marginBottom: 6 }}>Example records</h3>
          {samples.loading && <p className="muted">Loading…</p>}
          {samples.error && <p className="muted">{samples.error}</p>}
          <ul className="quotes">
            {samples.data?.samples.map((s, i) => <li key={i}>{s}</li>)}
          </ul>
          {samples.data && !samples.data.samples.length && <p className="muted">No examples stored for this topic.</p>}
        </section>
      </aside>
    </>
  );
}
