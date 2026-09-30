import {
  Bar, BarChart, CartesianGrid, LabelList, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { fmtDuration, fmtInt, fmtNum, timeTicks } from "../lib/format";
import type { RunCard } from "../lib/types";
import { TIMING_COLORS } from "./RunCharts";
import { AXIS, ChartCard, TooltipBox } from "./ui";

const short = (name: string) => (name.startsWith("base") ? "Base" : name.replace(/^.*_(\d+)$/, "Batch $1"));

type TimeRow = { run: string; total: number } & Record<string, number | string>;

export function RunTimeBreakdown({ runs }: { runs: RunCard[] }) {
  const groups = Object.keys(TIMING_COLORS);
  const rows: TimeRow[] = runs
    .filter((r) => r.timing_groups)
    .map((r) => ({ run: short(r.name), total: r.total_seconds ?? 0, ...(r.timing_groups as Record<string, number>) }));
  const all = rows.reduce((a, r) => a + r.total, 0);
  const ticks = timeTicks(Math.max(1, ...rows.map((r) => r.total)));
  if (!rows.length) return null;
  return (
    <ChartCard<TimeRow>
      title="Total time per run"
      subtitle={`Stage time by group for every finished run · ${fmtDuration(all)} across all ${rows.length} runs`}
      legend={groups.map((g) => ({ label: g, color: TIMING_COLORS[g] }))}
      table={{
        columns: [
          { key: "run", label: "Run" },
          ...groups.map((g) => ({ key: g, label: g, numeric: true, render: (r: TimeRow) => fmtDuration(Number(r[g])) })),
          { key: "total", label: "Total", numeric: true, render: (r: TimeRow) => fmtDuration(r.total) },
        ],
        rows,
      }}
    >
      <ResponsiveContainer width="100%" height={rows.length * 34 + 40}>
        <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 64, bottom: 0, left: 8 }} barCategoryGap={8}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" {...AXIS} ticks={ticks} domain={[0, ticks[ticks.length - 1]]} tickFormatter={(v) => fmtDuration(v)} />
          <YAxis type="category" dataKey="run" {...AXIS} axisLine={false} width={60} />
          <Tooltip cursor={{ fill: "var(--wash)" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const r = payload[0].payload as TimeRow;
              return <TooltipBox title={`${r.run} · ${fmtDuration(r.total)} total`}
                rows={groups.map((g) => ({ name: g, value: fmtDuration(Number(r[g])), color: TIMING_COLORS[g] }))} />;
            }} />
          {groups.map((g, i) => (
            <Bar key={g} dataKey={g} stackId="t" fill={TIMING_COLORS[g]} stroke="var(--surface)" strokeWidth={2}
              maxBarSize={22} radius={i === groups.length - 1 ? [0, 4, 4, 0] : 0} isAnimationActive={false}>
              {i === groups.length - 1 && (
                <LabelList dataKey="total" position="right" formatter={(v) => fmtDuration(Number(v))}
                  style={{ fill: "var(--ink-2)", fontSize: 11.5, fontWeight: 600 }} />
              )}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ------------------------------------------------------ small multiples ---

interface TrendRow { run: string; value: number | null }

function Trend({ title, subtitle, rows, kind, fmt }: {
  title: string; subtitle: string; rows: TrendRow[]; kind: "line" | "bar"; fmt: (v: number | null) => string;
}) {
  const tip = (active?: boolean, payload?: ReadonlyArray<{ payload?: unknown }>) => {
    const r = payload?.[0]?.payload as TrendRow | undefined;
    return active && r ? (
      <TooltipBox title={r.run} rows={[{ name: title.toLowerCase(), value: fmt(r.value), color: "var(--series-1)" }]} />
    ) : null;
  };
  const lastIdx = rows.length - 1;
  return (
    <ChartCard<TrendRow>
      title={title}
      subtitle={subtitle}
      table={{ columns: [{ key: "run", label: "Run" }, { key: "value", label: title, numeric: true, render: (r) => fmt(r.value) }], rows }}
    >
      <ResponsiveContainer width="100%" height={170}>
        {kind === "bar" ? (
          <BarChart data={rows} margin={{ top: 18, right: 4, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="run" {...AXIS} interval={0} tickFormatter={(v: string) => v.replace("Batch ", "B")} />
            <YAxis {...AXIS} axisLine={false} width={34} allowDecimals={false} />
            <Tooltip cursor={{ fill: "var(--wash)" }} content={({ active, payload }) => tip(active, payload)} />
            <Bar dataKey="value" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false}>
              <LabelList dataKey="value" position="top" style={{ fill: "var(--ink-2)", fontSize: 11 }} />
            </Bar>
          </BarChart>
        ) : (
          <LineChart data={rows} margin={{ top: 18, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="run" {...AXIS} interval={0} tickFormatter={(v: string) => v.replace("Batch ", "B")} />
            <YAxis {...AXIS} axisLine={false} width={40} domain={["auto", "auto"]} tickFormatter={(v) => fmt(v)} />
            <Tooltip cursor={{ stroke: "var(--axis)" }} content={({ active, payload }) => tip(active, payload)} />
            <Line dataKey="value" stroke="var(--series-1)" strokeWidth={2} isAnimationActive={false}
              dot={{ r: 4, fill: "var(--series-1)", stroke: "var(--surface)", strokeWidth: 2 }}
              activeDot={{ r: 5, stroke: "var(--surface)", strokeWidth: 2 }}>
              <LabelList dataKey="value" content={({ x, y, index, value }) =>
                index === lastIdx ? (
                  <text x={Number(x)} y={Number(y) - 10} textAnchor="end" fontSize={11} fill="var(--ink-2)">{fmt(Number(value))}</text>
                ) : null} />
            </Line>
          </LineChart>
        )}
      </ResponsiveContainer>
    </ChartCard>
  );
}

export function RunTrends({ runs }: { runs: RunCard[] }) {
  const ok = runs.filter((r) => !r.error);
  const rows = (f: (r: RunCard) => number | null) => ok.map((r) => ({ run: short(r.name), value: f(r) }));
  return (
    <div className="grid g4">
      <Trend title="New emerging topics" subtitle="Minted in each run's newest window" kind="bar"
        rows={rows((r) => r.new_emerging)} fmt={(v) => fmtInt(v)} />
      <Trend title="Live topics" subtitle="Registry topics with volume in the window" kind="line"
        rows={rows((r) => r.live_topics)} fmt={(v) => fmtInt(v)} />
      <Trend title="Alerts" subtitle="Topics scored HOT or TRENDING" kind="bar"
        rows={rows((r) => r.alerts)} fmt={(v) => fmtInt(v)} />
      <Trend title="UMAP drift" subtitle="Distance to the frozen manifold; 0.45 triggers a refit" kind="line"
        rows={rows((r) => r.drift)} fmt={(v) => fmtNum(v, 3)} />
    </div>
  );
}
