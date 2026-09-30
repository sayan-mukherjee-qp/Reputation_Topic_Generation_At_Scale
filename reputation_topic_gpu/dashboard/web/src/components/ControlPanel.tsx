import { useState } from "react";
import { api } from "../lib/api";
import { useFetch } from "../lib/hooks";
import type { Job, JobSnapshot } from "../lib/types";
import { StatusIcon } from "./ui";

export function ControlPanel({ job, onSnapshot }: { job: Job | null; onSnapshot: (s: JobSnapshot) => void }) {
  const running = job?.status === "running";
  const [includeStream, setIncludeStream] = useState(true);
  const [fresh, setFresh] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pre = useFetch(() => api.preflight(), [running]);

  const blocking = pre.data?.checks.filter((c) => !c.ok && c.level === "error") ?? [];
  const streamMissing = pre.data?.checks.some((c) => c.key === "stream_input" && !c.ok);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      onSnapshot(await api.start({ include_stream: includeStream && !streamMissing, fresh }));
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

  return (
    <section className="card control">
      <div style={{ display: "grid", gap: 14, alignContent: "start" }}>
        <div>
          <h2>Run the pipeline</h2>
          <p className="ink2" style={{ marginTop: 4 }}>
            Builds the topic registry from the 200k base set, then replays the 300k stream against it
            in batches, finding topics that are <strong>emerging</strong> in each new window.
          </p>
        </div>
        <div style={{ display: "grid", gap: 8 }}>
          <label className="check">
            <input type="checkbox" checked={includeStream && !streamMissing} disabled={running || streamMissing}
              onChange={(e) => setIncludeStream(e.target.checked)} />
            <span>
              Continue into the 300k stream after the base run
              <span className="muted" style={{ display: "block", fontSize: 12 }}>
                6 batches over a 3-chunk sliding window; each batch mints new EMERGING topics
              </span>
            </span>
          </label>
          <label className="check">
            <input type="checkbox" checked={fresh} disabled={running} onChange={(e) => setFresh(e.target.checked)} />
            <span>
              Fresh start
              <span className="muted" style={{ display: "block", fontSize: 12 }}>
                Deletes the previous rolling buffers and UMAP model so the run refits from scratch. The embedding
                cache is kept.
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
            <button className="btn btn-primary" onClick={start} disabled={busy || blocking.length > 0 || !pre.data}>
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                <path d="M4 2.5v11l9-5.5z" fill="currentColor" />
              </svg>
              {job ? "Start a new 200k run" : "Start 200k run"}
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
