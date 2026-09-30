import { ControlPanel } from "./components/ControlPanel";
import { LivePipeline } from "./components/LivePipeline";
import { Results } from "./components/Results";
import { StatusIcon } from "./components/ui";
import { useJob, useTheme, type ThemeChoice } from "./lib/hooks";

export function App() {
  const { snap, setSnap, connected, serverNow } = useJob();
  const [theme, setTheme] = useTheme();
  const job = snap?.job ?? null;

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

      <ControlPanel job={job} onSnapshot={setSnap} />
      {job && <LivePipeline job={job} serverNow={serverNow} />}
      <Results job={job} />
    </div>
  );
}
