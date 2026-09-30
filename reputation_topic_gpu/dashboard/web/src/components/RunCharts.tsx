import {
  Bar, BarChart, CartesianGrid, LabelList, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { fmtDuration, fmtGrowth, fmtInt, fmtNum, shortDay, timeTicks } from "../lib/format";
import type { RunOverview, Topic } from "../lib/types";
import { AXIS, ChartCard, TooltipBox } from "./ui";

// Color follows the entity everywhere on the page.
export const EST = { label: "Established", color: "var(--series-1)" };
export const EMG = { label: "Emerging", color: "var(--series-2)" };
export const isEmerging = (t: Topic) => t.status_lifecycle.endsWith("EMERGING");

export const TIMING_COLORS: Record<string, string> = {
  Embedding: "var(--series-1)",
  UMAP: "var(--series-2)",
  "Clustering & assignment": "var(--series-3)",
  "Labelling & scoring": "var(--series-4)",
  "I/O & metrics": "var(--series-5)",
};

// ------------------------------------------------------ daily volume ---

interface DayRow { date: string; established: number; emerging: number }

export function DailyVolume({ data, topics }: { data: RunOverview; topics: Topic[] }) {
  const ids = new Map(topics.map((t) => [t.topic_id, isEmerging(t)]));
  const rows: DayRow[] = data.series.dates.map((date) => ({ date, established: 0, emerging: 0 }));
  for (const [id, counts] of Object.entries(data.series.counts)) {
    const em = ids.get(id);
    if (em === undefined) continue;
    counts.forEach((c, i) => {
      if (em) rows[i].emerging += c;
      else rows[i].established += c;
    });
  }
  const last = rows[rows.length - 1];
  return (
    <ChartCard<DayRow>
      title="Daily topic volume"
      subtitle="Distinct records assigned per day; a record can sit in up to 2 topics"
      legend={[{ ...EST, shape: "line" }, { ...EMG, shape: "line" }]}
      table={{
        columns: [
          { key: "date", label: "Date" },
          { key: "established", label: "Established", numeric: true, render: (r) => fmtInt(r.established) },
          { key: "emerging", label: "Emerging", numeric: true, render: (r) => fmtInt(r.emerging) },
        ],
        rows,
      }}
    >
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={rows} margin={{ top: 8, right: 72, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="date" {...AXIS} tickFormatter={shortDay} minTickGap={28} />
          <YAxis {...AXIS} axisLine={false} width={48} tickFormatter={(v) => fmtInt(v)} />
          <Tooltip cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipBox title={shortDay(String(label))} rows={[
                  { name: EST.label, value: fmtInt((payload[0].payload as DayRow).established), color: EST.color },
                  { name: EMG.label, value: fmtInt((payload[0].payload as DayRow).emerging), color: EMG.color },
                ]} />
              ) : null} />
          {[{ key: "established", ...EST }, { key: "emerging", ...EMG }].map((s) => (
            <Line key={s.key} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} dot={false}
              activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2, fill: s.color }} isAnimationActive={false}>
              <LabelList dataKey={s.key} content={({ x, y, index }) =>
                index === rows.length - 1 && last ? (
                  <text x={Number(x) + 8} y={Number(y) + 4} fontSize={11.5} fill="var(--ink-2)">{s.label}</text>
                ) : null} />
            </Line>
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ---------------------------------------------------------- by brand ---

interface SegProps { x?: number; y?: number; width?: number; height?: number; fill?: string; payload?: BrandRow }

/** One segment of a horizontal stack. The segment that ends the bar gets the
 * 4px rounded data-end and the total label; every segment keeps the 2px
 * surface gap. (Recharts' `radius` can't know which segment is last.) */
function stackSegment(isEnd: (r: BrandRow) => boolean) {
  return function Segment(props: unknown) {
    const { x = 0, y = 0, width = 0, height = 0, fill, payload } = props as SegProps;
    if (width <= 0 || !payload) return <g />;
    const end = isEnd(payload);
    const r = end ? Math.min(4, width, height / 2) : 0;
    const d = `M${x},${y}H${x + width - r}Q${x + width},${y} ${x + width},${y + r}V${y + height - r}` +
      `Q${x + width},${y + height} ${x + width - r},${y + height}H${x}Z`;
    return (
      <g>
        <path d={d} fill={fill} stroke="var(--surface)" strokeWidth={2} />
        {end && (
          <text x={x + width + 6} y={y + height / 2 + 4} fontSize={11} fill="var(--ink-2)">{payload.total}</text>
        )}
      </g>
    );
  };
}

interface BrandRow { brand: string; established: number; emerging: number; total: number }

export function TopicsByBrand({ topics }: { topics: Topic[] }) {
  const by = new Map<string, BrandRow>();
  for (const t of topics) {
    const r = by.get(t.brand) ?? { brand: t.brand, established: 0, emerging: 0, total: 0 };
    if (isEmerging(t)) r.emerging++;
    else r.established++;
    r.total++;
    by.set(t.brand, r);
  }
  const rows = [...by.values()].sort((a, b) => b.total - a.total);
  return (
    <ChartCard<BrandRow>
      title="Topics by brand"
      subtitle="Registry topics per brand, split by lifecycle"
      legend={[EST, EMG]}
      table={{
        columns: [
          { key: "brand", label: "Brand" },
          { key: "established", label: "Established", numeric: true },
          { key: "emerging", label: "Emerging", numeric: true },
          { key: "total", label: "Total", numeric: true },
        ],
        rows,
      }}
    >
      <ResponsiveContainer width="100%" height={Math.max(140, rows.length * 28 + 36)}>
        <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 8 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" {...AXIS} allowDecimals={false} />
          <YAxis type="category" dataKey="brand" {...AXIS} axisLine={false} width={108} />
          <Tooltip cursor={{ fill: "var(--wash)" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const r = payload[0].payload as BrandRow;
              return <TooltipBox title={r.brand} rows={[
                { name: EST.label, value: fmtInt(r.established), color: EST.color },
                { name: EMG.label, value: fmtInt(r.emerging), color: EMG.color },
              ]} />;
            }} />
          <Bar dataKey="established" stackId="a" fill={EST.color} maxBarSize={20} isAnimationActive={false}
            shape={stackSegment((r: BrandRow) => r.emerging === 0)} />
          <Bar dataKey="emerging" stackId="a" fill={EMG.color} maxBarSize={20} isAnimationActive={false}
            shape={stackSegment(() => true)} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ------------------------------------------------------ alert status ---

const STATUS_ORDER = ["HOT", "TRENDING", "GROWING", "STABLE", "DECLINING", "LOW_EVIDENCE"];

interface StatusRow { status: string; label: string; count: number }

export function AlertStatus({ topics }: { topics: Topic[] }) {
  const counts = new Map<string, number>();
  topics.forEach((t) => counts.set(t.status, (counts.get(t.status) ?? 0) + 1));
  const keys = [...STATUS_ORDER.filter((s) => counts.has(s)), ...[...counts.keys()].filter((s) => !STATUS_ORDER.includes(s))];
  const rows: StatusRow[] = keys.map((s) => ({ status: s, label: s.replace("_", " ").toLowerCase(), count: counts.get(s) ?? 0 }));
  return (
    <ChartCard<StatusRow>
      title="Hot-score status"
      subtitle="HOT and TRENDING are alerts; the rest describe direction"
      table={{ columns: [{ key: "label", label: "Status" }, { key: "count", label: "Topics", numeric: true }], rows }}
    >
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={rows} margin={{ top: 18, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="label" {...AXIS} interval={0} />
          <YAxis {...AXIS} axisLine={false} width={40} allowDecimals={false} />
          <Tooltip cursor={{ fill: "var(--wash)" }}
            content={({ active, payload }) => active && payload?.length ? (
              <TooltipBox title={(payload[0].payload as StatusRow).label}
                rows={[{ name: "topics", value: fmtInt((payload[0].payload as StatusRow).count), color: "var(--series-1)" }]} />
            ) : null} />
          <Bar dataKey="count" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false}>
            <LabelList dataKey="count" position="top" style={{ fill: "var(--ink-2)", fontSize: 11 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ---------------------------------------------------- growth scatter ---

const LOG_TICKS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000];

interface Pt { x: number; y: number; t: Topic }

export function GrowthMap({ topics, onPick }: { topics: Topic[]; onPick: (t: Topic) => void }) {
  const pts = topics
    .filter((t) => (t.recent_volume ?? 0) > 0 && t.growth_rate != null)
    .map((t) => ({ x: t.recent_volume as number, y: t.growth_rate as number, t }));
  const sortedY = pts.map((p) => p.y).sort((a, b) => a - b);
  const cap = sortedY.length ? sortedY[Math.floor(sortedY.length * 0.98)] : 1;
  const clip = (p: Pt) => ({ ...p, y: Math.min(p.y, cap) });
  const est = pts.filter((p) => !isEmerging(p.t)).map(clip);
  const emg = pts.filter((p) => isEmerging(p.t)).map(clip);
  return (
    <ChartCard<Pt>
      title="Growth vs recent volume"
      subtitle="Each dot is a topic: last-7-day volume against growth over its baseline. Click a dot for detail"
      legend={[EST, EMG]}
      table={{
        columns: [
          { key: "label", label: "Topic", value: (p) => p.t.label, render: (p) => p.t.label },
          { key: "brand", label: "Brand", value: (p) => p.t.brand, render: (p) => p.t.brand },
          { key: "x", label: "Recent volume", numeric: true, render: (p) => fmtInt(p.x) },
          { key: "y", label: "Growth vs baseline", numeric: true, render: (p) => fmtGrowth(p.t.growth_rate) },
        ],
        rows: pts,
      }}
    >
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ top: 8, right: 12, bottom: 16, left: 0 }}>
          <CartesianGrid stroke="var(--grid)" />
          <XAxis type="number" dataKey="x" scale="log" domain={[1, "dataMax"]} {...AXIS} name="Recent volume"
            ticks={LOG_TICKS.filter((v) => v <= Math.max(1, ...pts.map((p) => p.x)) * 1.5)} tickFormatter={(v) => fmtInt(v)}
            label={{ value: "recent volume (7 days, log scale)", position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 11 }} />
          <YAxis type="number" dataKey="y" {...AXIS} axisLine={false} width={52} name="Growth"
            tickFormatter={(v) => fmtGrowth(v)} domain={[-1, "auto"]} />
          <ZAxis range={[64, 64]} />
          <Tooltip cursor={false}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as Pt;
              return <TooltipBox title={`${p.t.label} · ${p.t.brand}`} rows={[
                { name: "recent volume", value: fmtInt(p.x), color: isEmerging(p.t) ? EMG.color : EST.color },
                { name: "growth vs baseline", value: fmtGrowth(p.t.growth_rate) },
                { name: "hot score", value: fmtNum(p.t.hot_score) },
              ]} />;
            }} />
          <Scatter data={est} fill={EST.color} fillOpacity={0.55} stroke="var(--surface)" strokeWidth={1.5}
            isAnimationActive={false} onClick={(d) => onPick((d as unknown as Pt).t)} style={{ cursor: "pointer" }} />
          <Scatter data={emg} fill={EMG.color} stroke="var(--surface)" strokeWidth={2}
            isAnimationActive={false} onClick={(d) => onPick((d as unknown as Pt).t)} style={{ cursor: "pointer" }} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ------------------------------------------------------ stage timing ---

interface StageRow { stage: string; seconds: number }

export function StageTiming({ timing, total }: { timing: Record<string, number>; total: number | null }) {
  const rows: StageRow[] = Object.entries(timing)
    .map(([stage, seconds]) => ({ stage, seconds }))
    .filter((r) => r.seconds > 0.05)
    .sort((a, b) => b.seconds - a.seconds);
  const ticks = timeTicks(Math.max(1, ...rows.map((r) => r.seconds)));
  return (
    <ChartCard<StageRow>
      title="Where the time went"
      subtitle={`Per-stage wall clock · ${fmtDuration(total)} total`}
      table={{
        columns: [
          { key: "stage", label: "Stage" },
          { key: "seconds", label: "Time", numeric: true, render: (r) => fmtDuration(r.seconds) },
          { key: "share", label: "Share", numeric: true, value: (r) => r.seconds,
            render: (r) => (total ? `${((100 * r.seconds) / total).toFixed(1)}%` : "–") },
        ],
        rows,
      }}
    >
      <ResponsiveContainer width="100%" height={Math.max(140, rows.length * 26 + 36)}>
        <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 56, bottom: 0, left: 8 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" {...AXIS} ticks={ticks} domain={[0, ticks[ticks.length - 1]]} tickFormatter={(v) => fmtDuration(v)} />
          <YAxis type="category" dataKey="stage" {...AXIS} axisLine={false} width={140} />
          <Tooltip cursor={{ fill: "var(--wash)" }}
            content={({ active, payload }) => active && payload?.length ? (
              <TooltipBox title={(payload[0].payload as StageRow).stage}
                rows={[{ name: total ? `${((100 * (payload[0].payload as StageRow).seconds) / total).toFixed(1)}% of run` : "",
                  value: fmtDuration((payload[0].payload as StageRow).seconds), color: "var(--series-1)" }]} />
            ) : null} />
          <Bar dataKey="seconds" fill="var(--series-1)" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false}>
            <LabelList dataKey="seconds" position="right" formatter={(v) => fmtDuration(Number(v))}
              style={{ fill: "var(--ink-2)", fontSize: 11 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
