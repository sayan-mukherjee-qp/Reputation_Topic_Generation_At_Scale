import { useState, type ReactNode } from "react";

// ------------------------------------------------------------- icons ---

type IconKind = "done" | "running" | "failed" | "skipped" | "pending" | "warning" | "stopped";

/** State is never color alone: every status color comes with its own glyph. */
export function StatusIcon({ kind, label }: { kind: IconKind; label?: string }) {
  const title = label ?? kind;
  const common = { className: "status-icon", viewBox: "0 0 20 20", role: "img", "aria-label": title } as const;
  switch (kind) {
    case "done":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="9" fill="var(--good)" />
          <path d="M6 10.5l2.6 2.6L14 7.6" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "running":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="7.5" fill="none" stroke="var(--accent-wash)" strokeWidth="3" />
          <path className="spin" d="M10 2.5a7.5 7.5 0 0 1 7.5 7.5" fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" />
        </svg>
      );
    case "failed":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="9" fill="var(--critical)" />
          <path d="M7 7l6 6M13 7l-6 6" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
        </svg>
      );
    case "warning":
      return (
        <svg {...common}>
          <path d="M10 2l8.5 15h-17z" fill="var(--warning)" />
          <path d="M10 8v4" stroke="#0b0b0b" strokeWidth="2" strokeLinecap="round" />
          <circle cx="10" cy="14.6" r="1.1" fill="#0b0b0b" />
        </svg>
      );
    case "stopped":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="9" fill="var(--serious)" />
          <rect x="6.5" y="6.5" width="7" height="7" rx="1.5" fill="#fff" />
        </svg>
      );
    case "skipped":
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="8" fill="none" stroke="var(--axis)" strokeWidth="1.5" />
          <path d="M6.5 10h7" stroke="var(--muted)" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="7.5" fill="none" stroke="var(--axis)" strokeWidth="1.6" />
        </svg>
      );
  }
}

export function runIcon(status: string): IconKind {
  if (status === "succeeded" || status === "done") return "done";
  if (status === "running") return "running";
  if (status === "failed") return "failed";
  if (status === "stopped") return "stopped";
  if (status === "skipped") return "skipped";
  return "pending";
}

export const STATUS_WORD: Record<string, string> = {
  pending: "Waiting",
  running: "Running",
  succeeded: "Completed",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
  skipped: "Skipped",
};

// ------------------------------------------------------------- chips ---

const ALERT_DOT: Record<string, string> = {
  HOT: "var(--critical)",
  TRENDING: "var(--serious)",
};

/** Hot-score status. HOT / TRENDING are alerts and carry a status dot + word. */
export function AlertChip({ status }: { status: string }) {
  const dot = ALERT_DOT[status];
  return (
    <span className="chip" title={`Hot-score status: ${status}`}>
      {dot && <span className="dot" style={{ background: dot }} />}
      {status.replace("_", " ").toLowerCase()}
    </span>
  );
}

export function LifecycleChip({ lifecycle, isNew }: { lifecycle: string; isNew?: boolean }) {
  const emerging = lifecycle.endsWith("EMERGING");
  return (
    <span className="chip" title={`Lifecycle: ${lifecycle}`}>
      {emerging && <span className="dot" style={{ background: "var(--series-2)" }} />}
      {lifecycle === "MICRO_EMERGING" ? "micro-emerging" : lifecycle.toLowerCase()}
      {isNew && emerging ? " · new" : ""}
    </span>
  );
}

// ------------------------------------------------------------ charts ---

export interface LegendItem {
  label: string;
  color: string;
  shape?: "swatch" | "line";
}

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="chart-legend">
      {items.map((i) => (
        <span className="key" key={i.label}>
          <span className={i.shape === "line" ? "line" : "swatch"} style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export interface Column<R> {
  key: string;
  label: string;
  numeric?: boolean;
  render?: (row: R) => ReactNode;
  value?: (row: R) => number | string | null | undefined;
}

export function DataTable<R>({
  columns,
  rows,
  onRow,
  initialSort,
  maxHeight,
}: {
  columns: Column<R>[];
  rows: R[];
  onRow?: (row: R) => void;
  initialSort?: { key: string; desc: boolean };
  maxHeight?: number;
}) {
  const [sort, setSort] = useState(initialSort ?? null);
  const col = sort ? columns.find((c) => c.key === sort.key) : undefined;
  const sorted = col
    ? [...rows].sort((a, b) => {
        const get = col.value ?? ((r: R) => (r as Record<string, unknown>)[col.key] as number | string);
        const va = get(a);
        const vb = get(b);
        if (va == null) return 1;
        if (vb == null) return -1;
        const c = va < vb ? -1 : va > vb ? 1 : 0;
        return sort!.desc ? -c : c;
      })
    : rows;
  return (
    <div className="table-wrap" style={maxHeight ? { maxHeight } : undefined}>
      <table className="data">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.numeric ? "n" : undefined}
                aria-sort={sort?.key === c.key ? (sort.desc ? "descending" : "ascending") : undefined}>
                <button onClick={() => setSort({ key: c.key, desc: sort?.key === c.key ? !sort.desc : true })}>
                  {c.label}
                  {sort?.key === c.key ? (sort.desc ? " ↓" : " ↑") : ""}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i} className={onRow ? "clickable" : undefined} onClick={onRow ? () => onRow(r) : undefined}>
              {columns.map((c) => (
                <td key={c.key} className={c.numeric ? "n" : undefined}>
                  {c.render ? c.render(r) : String((r as Record<string, unknown>)[c.key] ?? "–")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A chart with its table-view twin: every value is reachable without hovering. */
export function ChartCard<R>({
  title,
  subtitle,
  legend,
  table,
  children,
  actions,
}: {
  title: string;
  subtitle?: ReactNode;
  legend?: LegendItem[];
  table?: { columns: Column<R>[]; rows: R[] };
  children: ReactNode;
  actions?: ReactNode;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <section className="card">
      <div className="card-head">
        <div className="grow">
          <h3>{title}</h3>
          {subtitle && <div className="sub">{subtitle}</div>}
        </div>
        {actions}
        {table && (
          <div className="seg" role="group" aria-label={`${title} view`}>
            <button aria-pressed={view === "chart"} onClick={() => setView("chart")}>Chart</button>
            <button aria-pressed={view === "table"} onClick={() => setView("table")}>Table</button>
          </div>
        )}
      </div>
      {view === "chart" ? (
        <>
          {legend && legend.length > 1 && <Legend items={legend} />}
          {children}
        </>
      ) : (
        table && <DataTable columns={table.columns} rows={table.rows} maxHeight={320} />
      )}
    </section>
  );
}

export interface TipRow {
  name: string;
  value: string;
  color?: string;
}

/** Values lead, series names follow; keyed with a short line, not a box. */
export function TooltipBox({ title, rows }: { title?: string; rows: TipRow[] }) {
  return (
    <div className="tooltip">
      {title && <div className="tt-title">{title}</div>}
      {rows.map((r) => (
        <div className="tt-row" key={r.name}>
          {r.color && <span className="tt-key" style={{ background: r.color }} />}
          <strong>{r.value}</strong>
          <span className="tt-name">{r.name}</span>
        </div>
      ))}
    </div>
  );
}

/** Tiny single-series trend; the card's numbers carry the values. */
export function Sparkline({ values, color = "var(--series-2)", height = 32 }: { values: number[]; color?: string; height?: number }) {
  const w = 120;
  if (!values.length) return <svg width="100%" height={height} />;
  const max = Math.max(1, ...values);
  const step = values.length > 1 ? w / (values.length - 1) : 0;
  const pts = values.map((v, i) => [i * step, height - 3 - (v / max) * (height - 6)] as const);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
  const area = `${line}L${w},${height}L0,${height}Z`;
  const [lx, ly] = pts[pts.length - 1];
  return (
    <svg viewBox={`0 0 ${w} ${height}`} width="100%" height={height} preserveAspectRatio="none" aria-hidden="true">
      <path d={area} fill={color} opacity={0.1} />
      <path d={line} fill="none" stroke={color} strokeWidth={2} vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={lx} cy={ly} r={3} fill={color} stroke="var(--surface)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function Tile({ label, value, foot }: { label: string; value: ReactNode; foot?: ReactNode }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {foot && <div className="foot">{foot}</div>}
    </div>
  );
}

export const AXIS = {
  stroke: "var(--axis)",
  tickLine: false,
  tick: { fill: "var(--muted)", fontSize: 11 },
} as const;
