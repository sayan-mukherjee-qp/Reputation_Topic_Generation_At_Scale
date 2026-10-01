import { ControlPanel } from "./components/ControlPanel";
import { LivePipeline } from "./components/LivePipeline";
import { Results } from "./components/Results";
import { SavedRuns } from "./components/SavedRuns";
import { StatusIcon } from "./components/ui";
import { useEffect, useState } from "react";
import { api } from "./lib/api";
import { useFetch, useJob, useTheme, type ThemeChoice } from "./lib/hooks";

export function App() {
  const { snap, setSnap, connected, serverNow } = useJob();
  const [theme, setTheme] = useTheme();
  const job = snap?.job ?? null;
  const allDatasets = useFetch(() => api.datasets(), []).data?.datasets ?? [];
  const datasets = allDatasets.filter((d) => d.runnable);
  const groups = [...new Set(allDatasets.filter((d) => !d.runnable).map((d) => d.group))];
  const [tab, setTab] = useState<string>(() => {
    try {
      return localStorage.getItem("rtm-tab") || "pipeline";
    } catch {
      return "pipeline";
    }
  });
  const activeTab = tab === "pipeline" || groups.includes(tab) ? tab : "pipeline";
  useEffect(() => {
    try {
      localStorage.setItem("rtm-tab", activeTab);
    } catch {
      /* storage unavailable */
    }
  }, [activeTab]);
  const [dataset, setDataset] = useState<string>(() => {
    try {
      return localStorage.getItem("rtm-dataset") || "slice";
    } catch {
      return "slice";
    }
  });
  // A running job pins the view to its own dataset.
  useEffect(() => {
    if (job?.status === "running" && job.dataset !== dataset) setDataset(job.dataset);
  }, [job?.status, job?.dataset]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    try {
      localStorage.setItem("rtm-dataset", dataset);
    } catch {
      /* storage unavailable */
    }
  }, [dataset]);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">
          <svg width="16" height="16" viewBox="0 0 16 16">
            <rect x="1" y="9" width="3" height="6" rx="1" fill="#fff" />
            <rect x="6" y="5" width="3" height="10" rx="1" fill="#fff" />
            <rect x="11" y="1" width="3" height="14" rx="1" fill="#fff" opacity="0.7" />
          </svg>
        </div>
        <div className="grow">
          <h1>Reputation Topic Monitor</h1>
          <p className="muted" style={{ fontSize: 12 }}>Topic discovery and emerging-topic detection · GPU pipeline</p>
        </div>
        <span className="row muted" style={{ fontSize: 12 }} aria-live="polite">
          <StatusIcon kind={connected ? "done" : "warning"} label={connected ? "Connected" : "Reconnecting"} />
          {connected ? "Live" : "Reconnecting…"}
        </span>
        <div className="seg" role="group" aria-label="Theme">
          {(["auto", "light", "dark"] as ThemeChoice[]).map((t) => (
            <button key={t} aria-pressed={theme === t} onClick={() => setTheme(t)}>
              {t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </header>

      {groups.length > 0 && (
        <nav className="top-tabs" role="tablist" aria-label="Views">
          <button role="tab" aria-selected={activeTab === "pipeline"} onClick={() => setTab("pipeline")}>
            Pipeline
            {job?.status === "running" && <StatusIcon kind="running" label="Running" />}
          </button>
          {groups.map((g) => (
            <button key={g} role="tab" aria-selected={activeTab === g} onClick={() => setTab(g)}>{g}</button>
          ))}
        </nav>
      )}

      {activeTab === "pipeline" ? (
        <>
          <ControlPanel job={job} onSnapshot={setSnap} datasets={datasets} dataset={dataset} onDataset={setDataset} />
          {job && <LivePipeline job={job} serverNow={serverNow} />}
          <Results job={job} dataset={dataset} />
        </>
      ) : (
        <SavedRuns group={activeTab} datasets={allDatasets.filter((d) => !d.runnable && d.group === activeTab)} />
      )}
    </div>
  );
}
