import { useEffect, useRef, useState, type ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtClock, fmtDuration, fmtEpoch, fmtInt, timeTicks } from "../lib/format";
import { useTick } from "../lib/hooks";
import type { Batch, Job, Phase, Stage } from "../lib/types";
import { AXIS, ChartCard, StatusIcon, STATUS_WORD, TooltipBox, runIcon } from "./ui";

type Now = () => number;

const elapsed = (start: number | null, end: number | null, now: number) =>
  start == null ? null : (end ?? now) - start;

/** Seconds left on a stage list, from the previous run's per-stage timings. */
function remaining(stages: Stage[], now: number): number | null {
  let known = false;
  let left = 0;
  for (const s of stages) {
    if (s.expected_seconds == null) continue;
    if (s.status === "pending") {
      left += s.expected_seconds;
      known = true;
    } else if (s.status === "running") {
      left += Math.max(0, s.expected_seconds - (now - (s.started_at ?? now)));
      known = true;
    }
  }
  return known ? left : null;
}

function currentStage(job: Job): { where: string; stage: Stage } | null {
  for (const p of job.phases) {
    const s = p.stages?.find((x) => x.status === "running");
    if (s) return { where: p.title, stage: s };
    for (const b of p.batches ?? []) {
      const t = b.stages.find((x) => x.status === "running");
      if (t) return { where: `Stream batch ${b.index}`, stage: t };
    }
  }
  return null;
}

// --------------------------------------------------------------- stages ---

function StageProgress({ stage, now }: { stage: Stage; now: number }) {
  const p = stage.progress;
  if (p && p.total) {
    const pct = Math.min(100, (100 * p.done) / p.total);
    return (
      <>
        <div className="meter" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
          <div style={{ width: `${pct}%` }} />
        </div>
        <div className="detail">
          {fmtInt(p.done)} / {fmtInt(p.total)} {p.unit} · {pct.toFixed(0)}%{p.last ? ` · ${p.last}` : ""}
        </div>
      </>
    );
  }
  const exp = stage.expected_seconds;
  const run = now - (stage.started_at ?? now);
  if (exp && exp > 1) {
    const pct = Math.min(99, (100 * run) / exp);
    return (
      <>
        <div className="meter" title="Estimated from the previous run's timing"><div style={{ width: `${pct}%` }} /></div>
        <div className="detail">
          {p ? `${fmtInt(p.done)} ${p.unit}${p.last ? ` · ${p.last}` : ""} · ` : ""}
          last run took {fmtDuration(exp)}
        </div>
      </>
    );
  }
  return (
    <>
      <div className="meter indeterminate"><div /></div>
      {p && <div className="detail">{fmtInt(p.done)} {p.unit}{p.last ? ` · ${p.last}` : ""}</div>}
    </>
  );
}

export function StageList({ stages, now }: { stages: Stage[]; now: number }) {
  const visible = stages.filter((s) => s.status !== "skipped" || s.seconds != null);
  const longest = Math.max(1, ...visible.map((s) => (s.status === "running" ? now - (s.started_at ?? now) : s.seconds ?? 0)));
  return (
    <ol className="stages">
      {visible.map((s) => {
        const secs = s.status === "running" ? now - (s.started_at ?? now) : s.seconds;
        return (
          <li key={s.key} className={`stage ${s.status}`}>
            <StatusIcon kind={runIcon(s.status)} label={STATUS_WORD[s.status]} />
            <div style={{ minWidth: 0 }}>
              <div className="name">{s.label}</div>
              {s.status === "running" ? (
                <StageProgress stage={s} now={now} />
              ) : s.detail ? (
                <div className="detail" title={s.detail}>{s.detail}</div>
              ) : null}
              {s.status === "done" && secs != null && (
                <div className="dur-track"><div className="dur-bar" style={{ width: `${(100 * secs) / longest}%` }} /></div>
              )}
            </div>
            <div className="secs">{secs != null ? fmtDuration(secs) : s.status === "pending" && (s.expected_seconds ?? 0) >= 1 ? <span className="muted">~{fmtDuration(s.expected_seconds)}</span> : ""}</div>
          </li>
        );
      })}
    </ol>
  );
}

// --------------------------------------------------------------- phases ---

function PhaseHeader({ phase, now, extra }: { phase: Phase; now: number; extra?: ReactNode }) {
  const t = elapsed(phase.started_at, phase.finished_at, now);
  return (
    <div className="card-head">
      <StatusIcon kind={runIcon(phase.status)} label={STATUS_WORD[phase.status]} />
      <div className="grow">
        <h3>{phase.title}</h3>
        <div className="sub">{phase.subtitle} · {STATUS_WORD[phase.status]}</div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div className="num" style={{ fontWeight: 600, fontSize: 18 }}>{t == null ? "–" : fmtClock(t)}</div>
        <div className="sub">{extra}</div>
      </div>
    </div>
  );
}

function BasePhase({ phase, now }: { phase: Phase; now: number }) {
  const stages = phase.stages ?? [];
  const left = phase.status === "running" ? remaining(stages, now) : null;
  const info = phase.info ?? {};
  return (
    <section className="card">
      <PhaseHeader phase={phase} now={now}
        extra={left != null ? `≈ ${fmtDuration(left)} left` : phase.status === "succeeded" ? "total run time" : "elapsed"} />
      {(info.records || info.segments || info.device) && (
        <div className="row wrap" style={{ marginBottom: 10 }}>
          {info.records != null && <span className="chip">{fmtInt(info.records)} records</span>}
          {info.segments != null && <span className="chip">{fmtInt(info.segments)} segments</span>}
          {info.brands != null && <span className="chip">{info.brands} brands</span>}
          {info.device && <span className="chip">embedding on {info.device}</span>}
          {info.emerging_created != null && (
            <span className="chip"><span className="dot" style={{ background: "var(--series-2)" }} />
              candidate pass: {info.emerging_created} new + {info.micro_emerging_created ?? 0} micro</span>
          )}
        </div>
      )}
      <StageList stages={stages} now={now} />
      {phase.error && <p className="muted" style={{ marginTop: 8 }}>Error: {phase.error}</p>}
    </section>
  );
}

function StreamPhase({ phase, now }: { phase: Phase; now: number }) {
  const batches = phase.batches ?? [];
  const runningIdx = batches.find((b) => b.status === "running")?.index;
  const [picked, setPicked] = useState<number | null>(null);
  const shown: Batch | undefined =
    batches.find((b) => b.index === picked) ?? batches.find((b) => b.index === runningIdx) ??
    [...batches].reverse().find((b) => b.status !== "pending") ?? batches[0];
  const done = batches.filter((b) => b.status === "succeeded").length;
  let left: number | null = null;
  if (phase.status === "running") {
    const parts = batches.filter((b) => b.status === "pending" || b.status === "running").map((b) => remaining(b.stages, now));
    left = parts.every((x) => x != null) ? parts.reduce<number>((a, b) => a + (b ?? 0), 0) : null;
  }
  return (
    <section className="card">
      <PhaseHeader phase={phase} now={now}
        extra={left != null ? `≈ ${fmtDuration(left)} left` : `${done}/${batches.length} batches`} />
      <div className="batches" role="group" aria-label="Stream batches">
        {batches.map((b) => {
          const t = elapsed(b.started_at, b.finished_at, now);
          return (
            <button key={b.index} className={`batch ${b.status}`}
              aria-pressed={shown?.index === b.index && (b.status !== "pending" || picked === b.index)}
              onClick={() => setPicked(b.index)}>
              <div className="row">
                <StatusIcon kind={runIcon(b.status)} label={STATUS_WORD[b.status]} />
                <strong>Batch {b.index}</strong>
                <span className="spacer" />
                <span className="num t">{t == null ? "" : fmtDuration(t)}</span>
              </div>
              <div className="t">
                {b.summary
                  ? `${b.summary.topics} topics · ${b.summary.alerts} alerts`
                  : b.status === "running"
                    ? b.stages.find((s) => s.status === "running")?.label ?? "starting"
                    : b.chunk}
              </div>
            </button>
          );
        })}
      </div>
      {shown && shown.status !== "pending" && (
        <div style={{ marginTop: 12 }}>
          <div className="row" style={{ marginBottom: 6 }}>
            <h4 style={{ fontSize: 13 }}>Batch {shown.index} · {shown.chunk}</h4>
            <span className="spacer" />
            {shown.summary && (
              <span className="muted" style={{ fontSize: 12 }}>
                drift {shown.summary.drift ?? "–"} · id retention {shown.summary.id_retention ?? "–"} · events {shown.summary.events ?? "–"}
              </span>
            )}
          </div>
          <StageList stages={shown.stages} now={now} />
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------ run times ---

interface TimeRow {
  run: string;
  seconds: number;
  status: string;
}

function RunTimes({ job, now }: { job: Job; now: number }) {
  const rows: TimeRow[] = [];
  for (const p of job.phases) {
    if (p.key === "base") {
      const t = elapsed(p.started_at, p.finished_at, now);
      if (t != null) rows.push({ run: "Base 200k", seconds: t, status: p.status });
    }
    for (const b of p.batches ?? []) {
      const t = elapsed(b.started_at, b.finished_at, now);
      if (t != null) rows.push({ run: `Batch ${b.index}`, seconds: t, status: b.status });
    }
  }
  if (!rows.length) return null;
  const total = elapsed(job.started_at, job.finished_at, now) ?? 0;
  const ticks = timeTicks(Math.max(1, ...rows.map((r) => r.seconds)));
  return (
    <ChartCard<TimeRow>
      title="Time per run"
      subtitle={`Wall-clock per pipeline run in this job · ${fmtDuration(total)} in total`}
      table={{
        columns: [
          { key: "run", label: "Run" },
          { key: "status", label: "Status", render: (r) => STATUS_WORD[r.status] ?? r.status },
          { key: "seconds", label: "Time", numeric: true, render: (r) => fmtDuration(r.seconds) },
        ],
        rows,
      }}
    >
      <ResponsiveContainer width="100%" height={Math.max(96, rows.length * 34 + 36)}>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 64, bottom: 4, left: 8 }}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" {...AXIS} ticks={ticks} domain={[0, ticks[ticks.length - 1]]} tickFormatter={(v) => fmtDuration(v)} />
          <YAxis type="category" dataKey="run" width={76} {...AXIS} axisLine={false} />
          <Tooltip cursor={{ fill: "var(--wash)" }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox title={(payload[0].payload as TimeRow).run}
                  rows={[{ name: STATUS_WORD[(payload[0].payload as TimeRow).status] ?? "", value: fmtDuration((payload[0].payload as TimeRow).seconds), color: "var(--series-1)" }]} />
              ) : null} />
          <Bar dataKey="seconds" maxBarSize={20} radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {rows.map((r) => (
              <Cell key={r.run} fill="var(--series-1)" fillOpacity={r.status === "running" ? 0.55 : 1} />
            ))}
            <LabelList dataKey="seconds" position="right" formatter={(v) => fmtDuration(Number(v))}
              style={{ fill: "var(--ink-2)", fontSize: 11.5 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ------------------------------------------------------------------ log ---

function LogConsole({ lines }: { lines: string[] }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLPreElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines, open]);
  return (
    <section className="card">
      <div className="card-head" style={{ marginBottom: open ? 12 : 0 }}>
        <div className="grow">
          <h3>Pipeline log</h3>
          <div className="sub">Last {lines.length} lines of stdout/stderr</div>
        </div>
        <button className="btn btn-sm" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? "Hide" : "Show"}</button>
      </div>
      {open && (
        <pre className="log" ref={ref}
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}>
          {lines.join("\n")}
        </pre>
      )}
    </section>
  );
}

// ----------------------------------------------------------------- main ---

export function LivePipeline({ job, serverNow }: { job: Job; serverNow: Now }) {
  useTick(job.status === "running");
  const now = serverNow();
  const total = elapsed(job.started_at, job.finished_at, now);
  const cur = currentStage(job);
  const base = job.phases.find((p) => p.key === "base");
  const stream = job.phases.find((p) => p.key === "stream");
  const word = { running: "Running", succeeded: "Completed", failed: "Failed", stopped: "Stopped" }[job.status];

  return (
    <div className="section">
      <div className="section-head">
        <h2>Live pipeline</h2>
        <p>Started {fmtEpoch(job.started_at)}{job.finished_at ? ` · finished ${fmtEpoch(job.finished_at)}` : ""}</p>
      </div>
      <section className="card pipeline-head">
        <div>
          <div className="muted" style={{ fontSize: 12 }}>Total elapsed</div>
          <div className="hero-clock num">{fmtClock(total)}</div>
        </div>
        <div className="grow" style={{ minWidth: 220 }}>
          <div className="row" style={{ marginBottom: 4 }}>
            <StatusIcon kind={runIcon(job.status)} label={word} />
            <strong>{word}</strong>
          </div>
          {cur ? (
            <p className="ink2">
              Now: <strong>{cur.stage.label}</strong> <span className="muted">in {cur.where} · {fmtDuration(now - (cur.stage.started_at ?? now))}</span>
            </p>
          ) : job.status === "succeeded" ? (
            <p className="ink2">All runs finished. Results below are up to date.</p>
          ) : job.error ? (
            <p className="ink2">{job.error}</p>
          ) : null}
        </div>
        <div className="row wrap" style={{ gap: 20 }}>
          {job.phases.map((p) => (
            <div key={p.key}>
              <div className="muted" style={{ fontSize: 12 }}>{p.title}</div>
              <div className="num" style={{ fontWeight: 600 }}>{fmtDuration(elapsed(p.started_at, p.finished_at, now))}</div>
            </div>
          ))}
        </div>
      </section>
      {job.error && job.status === "failed" && (
        <div className="banner critical" role="alert" style={{ marginTop: 12 }}>
          <StatusIcon kind="failed" label="Failed" />
          <span>{job.error}. The full log is in the Pipeline log below.</span>
        </div>
      )}
      <div className="phase-grid">
        {base && <BasePhase phase={base} now={now} />}
        <div style={{ display: "grid", gap: 16, alignContent: "start" }}>
          {stream && <StreamPhase phase={stream} now={now} />}
          <RunTimes job={job} now={now} />
        </div>
      </div>
      <div style={{ marginTop: 16 }}>
        <LogConsole lines={job.log} />
      </div>
    </div>
  );
}
