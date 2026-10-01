import { useState } from "react";
import { api } from "../lib/api";
import { fmtCompact, fmtInt } from "../lib/format";
import { useFetch } from "../lib/hooks";
import type { DatasetInfo, Job, JobSnapshot } from "../lib/types";
import { StatusIcon } from "./ui";

export function ControlPanel({ job, onSnapshot, datasets, dataset, onDataset }: {
  job: Job | null;
  onSnapshot: (s: JobSnapshot) => void;
  datasets: DatasetInfo[];
  dataset: string;
  onDataset: (key: string) => void;
}) {
  const running = job?.status === "running";
  const [includeStream, setIncludeStream] = useState(true);
  const [fresh, setFresh] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pre = useFetch(() => api.preflight(dataset), [dataset, running]);
  const ds = datasets.find((d) => d.key === dataset);

  const blocking = pre.data?.checks.filter((c) => !c.ok && c.level === "error") ?? [];
  const streamMissing = pre.data?.checks.some((c) => c.key === "stream_input" && !c.ok);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      onSnapshot(await api.start({ dataset, include_stream: includeStream && !streamMissing, fresh }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (!window.confirm("Stop the running pipeline? The current stage's work is lost.")) return;
    setBusy(true);
    try {
      onSnapshot(await api.stop());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const baseN = ds?.base_rows ? fmtCompact(ds.base_rows) : "base";
  const streamN = ds?.stream_rows ? fmtCompact(ds.stream_rows) : "";

  return (
    <section className="card control">
      <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
        <div>
          <h2>Run the pipeline</h2>
          <p className="ink2" style={{ marginTop: 4 }}>
            Builds the topic registry from the base set, then replays the stream against it in batches,
            finding topics that are <strong>emerging</strong> in each new window.
          </p>
        </div>

        <div style={{ display: "grid", gap: 6 }}>
          <span className="muted" style={{ fontSize: 12 }}>Dataset</span>
          <div className="seg" role="group" aria-label="Dataset" style={{ width: "fit-content" }}>
            {datasets.map((d) => (
              <button key={d.key} aria-pressed={dataset === d.key} disabled={running}
                title={d.detail ?? undefined} onClick={() => onDataset(d.key)}
                style={{ padding: "6px 14px", fontSize: 13 }}>
                {d.label}
              </button>
            ))}
          </div>
          {ds && (
            <span className="muted" style={{ fontSize: 12 }}>
              {ds.available
                ? `${ds.base_csv} · ${fmtInt(ds.base_rows)} base records · ${fmtInt(ds.stream_rows)} stream records in ${ds.stream_chunks} chunks`
                : ds.detail}
              {ds.key === "slice" && ds.available ? " · a 1/10 sample, same dates and brand mix" : ""}
            </span>
          )}
        </div>

        <div style={{ display: "grid", gap: 8 }}>
          <label className="check">
            <input type="checkbox" checked={includeStream && !streamMissing} disabled={running || streamMissing}
              onChange={(e) => setIncludeStream(e.target.checked)} />
            <span>
              Continue into the {streamN} stream after the base run
              <span className="muted" style={{ display: "block", fontSize: 12 }}>
                {ds?.stream_chunks ?? 6} batches over a 3-chunk sliding window; each batch mints new EMERGING topics
              </span>
            </span>
          </label>
          <label className="check">
            <input type="checkbox" checked={fresh} disabled={running} onChange={(e) => setFresh(e.target.checked)} />
            <span>
              Fresh start
              <span className="muted" style={{ display: "block", fontSize: 12 }}>
                Deletes this dataset's previous rolling buffers and UMAP model so the run refits from scratch.
                The embedding cache is kept.
              </span>
            </span>
          </label>
        </div>
        <div className="row wrap">
          {running ? (
            <button className="btn btn-danger" onClick={stop} disabled={busy}>
              <StatusIcon kind="stopped" label="" /> Stop run
            </button>
          ) : (
            <button className="btn btn-primary" onClick={start}
              disabled={busy || blocking.length > 0 || !pre.data || !ds?.available}>
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                <path d="M4 2.5v11l9-5.5z" fill="currentColor" />
              </svg>
              Start {baseN} run
            </button>
          )}
          {blocking.length > 0 && <span className="muted">Fix the failed checks to start.</span>}
        </div>
        {error && (
          <div className="banner critical" role="alert">
            <StatusIcon kind="failed" label="Error" />
            <span>{error}</span>
          </div>
        )}
      </div>

      <div className="checks">
        <h3>Preflight</h3>
        {pre.error && <p className="muted">Could not reach the backend: {pre.error}</p>}
        <ul>
          {pre.data?.checks.map((c) => (
            <li key={c.key}>
              <StatusIcon kind={c.ok ? "done" : c.level === "error" ? "failed" : "warning"}
                label={c.ok ? "OK" : c.level === "error" ? "Failed" : "Warning"} />
              <div className="grow">
                <div>{c.label}</div>
                <div className="detail">{c.detail}</div>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
