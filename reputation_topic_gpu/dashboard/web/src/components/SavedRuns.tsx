import { useEffect, useState } from "react";
import { fmtDate, fmtInt } from "../lib/format";
import type { DatasetInfo } from "../lib/types";
import { Results } from "./Results";
import { StatusIcon } from "./ui";

/** Result sets produced elsewhere (e.g. a GPU box), browsed exactly like a live run's. */
export function SavedRuns({ group, datasets }: { group: string; datasets: DatasetInfo[] }) {
  const storeKey = `rtm-saved-${group}`;
  const [picked, setPicked] = useState<string>(() => {
    try {
      return localStorage.getItem(storeKey) || "";
    } catch {
      return "";
    }
  });
  const ds = datasets.find((d) => d.key === picked) ?? datasets[0];
  useEffect(() => {
    try {
      if (ds) localStorage.setItem(storeKey, ds.key);
    } catch {
      /* storage unavailable */
    }
  }, [ds, storeKey]);
  if (!ds) return null;

  return (
    <>
      <section className="card" style={{ display: "grid", gap: 12 }}>
        <div className="row wrap" style={{ alignItems: "flex-start" }}>
          <div className="grow">
            <h2>{group}</h2>
            <p className="ink2" style={{ marginTop: 4 }}>
              Saved output from earlier pipeline runs, shown the same way as a live run. Read-only.
            </p>
          </div>
          {datasets.length > 1 && <div className="seg" role="group" aria-label="Result set">
            {datasets.map((d) => (
              <button key={d.key} aria-pressed={d.key === ds.key} onClick={() => setPicked(d.key)}
                style={{ padding: "6px 14px", fontSize: 13 }}>
                {d.label}
              </button>
            ))}
          </div>}
        </div>
        {ds.available ? (
          <div className="row wrap" style={{ gap: 6 }}>
            {ds.embedding_model && <span className="chip">{ds.embedding_model}{ds.embedding_dim ? ` · ${ds.embedding_dim}-dim` : ""}</span>}
            {ds.embedding_device && <span className="chip">embedded on {ds.embedding_device}</span>}
            {ds.labels && <span className="chip">labels: {ds.labels}{ds.llm_model ? ` · ${ds.llm_model}` : ""}</span>}
            <span className="chip">base + {ds.stream_batches?.length ?? 0} stream batches</span>
            {ds.completed_to && <span className="chip">finished {fmtDate(ds.completed_to, true)}</span>}
          </div>
        ) : (
          <div className="banner critical"><StatusIcon kind="failed" label="Unavailable" /><span>{ds.detail}</span></div>
        )}
        {ds.description && <p className="muted" style={{ fontSize: 13 }}>{ds.description}</p>}
        {!!ds.missing_batches?.length && (
          <div className="banner">
            <StatusIcon kind="warning" label="Note" />
            <span>
              Stream batch {ds.missing_batches.join(", ")} is not in this export, so batch{" "}
              {fmtInt(Math.max(...ds.missing_batches) + 1)} is compared with the base run when counting new topics.
            </span>
          </div>
        )}
        {ds.llm_error && (
          <div className="banner">
            <StatusIcon kind="warning" label="Note" />
            <span>No LLM labels in this set: <span className="mono">{ds.llm_error}</span></span>
          </div>
        )}
      </section>
      {ds.available && <Results key={ds.key} job={null} dataset={ds.key} />}
    </>
  );
}
