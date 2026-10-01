import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import { fmtDate, fmtDuration, fmtGrowth, fmtInt, fmtNum, fmtPct } from "../lib/format";
import { useFetch } from "../lib/hooks";
import type { EventRecall, Job, RunCard, RunOverview, Topic } from "../lib/types";
import { RunTimeBreakdown, RunTrends } from "./AcrossRuns";
import { RecordsView } from "./RecordsView";
import { AlertStatus, DailyVolume, GrowthMap, isEmerging, StageTiming, TopicsByBrand } from "./RunCharts";
import { TopicDrawer } from "./TopicDrawer";
import { AlertChip, ChartCard, DataTable, LifecycleChip, Sparkline, StatusIcon, Tile } from "./ui";

const runTitle = (r: RunCard) => (r.kind === "base" ? "Base registry" : r.name.replace(/^.*_(\d+)$/, "Stream batch $1"));

// ------------------------------------------------------- emerging list ---

function EmergingCard({ t, series, onPick }: { t: Topic; series: number[]; onPick: () => void }) {
  const hot = Math.max(0, Math.min(1, t.hot_score ?? 0));
  return (
    <button className="topic-card" onClick={onPick}>
      <div className="row" style={{ alignItems: "flex-start" }}>
        <div className="grow">
          <div className="muted" style={{ fontSize: 12 }}>{t.brand} · first seen {fmtDate(t.created_at, true)}</div>
          <div className="title">{t.label}</div>
        </div>
      </div>
      <div className="row wrap">
        <LifecycleChip lifecycle={t.status_lifecycle} isNew={t.is_new} />
        <AlertChip status={t.status} />
        {t.suppressed && <span className="chip" title={t.suppress_reason ?? ""}>suppressed</span>}
      </div>
      <Sparkline values={series} />
      <div className="stats">
        <div><div className="v num">{fmtInt(t.size)}</div><div className="k">records</div></div>
        <div><div className="v num">{fmtInt(t.recent_volume)}</div><div className="k">last 7 days</div></div>
        <div><div className="v num">{fmtGrowth(t.growth_rate)}</div><div className="k">vs baseline</div></div>
        <div><div className="v num">{fmtNum(t.hot_score)}</div><div className="k">hot score</div></div>
      </div>
      <div className="hot-meter" aria-hidden="true"><div style={{ width: `${hot * 100}%` }} /></div>
    </button>
  );
}

function EmergingTopics({ topics, data, onPick }: { topics: Topic[]; data: RunOverview; onPick: (t: Topic) => void }) {
  const [showAll, setShowAll] = useState(false);
  const emerging = topics.filter(isEmerging).sort((a, b) => (b.hot_score ?? 0) - (a.hot_score ?? 0));
  const shown = showAll ? emerging : emerging.slice(0, 12);
  const window = 28;
  return (
    <section className="section" style={{ marginTop: 20 }}>
      <div className="section-head">
        <h2>Emerging topics</h2>
        <p>{emerging.length} topics not matched to the existing registry, ranked by hot score · sparkline: last {window} days</p>
      </div>
      {emerging.length === 0 ? (
        <div className="empty">No emerging topics under the current filters. Try including suppressed topics.</div>
      ) : (
        <>
          <div className="topic-grid">
            {shown.map((t) => (
              <EmergingCard key={t.topic_id} t={t} onPick={() => onPick(t)}
                series={(data.series.counts[t.topic_id] ?? []).slice(-window)} />
            ))}
          </div>
          {emerging.length > 12 && (
            <div style={{ marginTop: 10 }}>
              <button className="btn btn-sm" onClick={() => setShowAll(!showAll)}>
                {showAll ? "Show top 12" : `Show all ${emerging.length}`}
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}

// ------------------------------------------------------------ tables ---

function TopTopics({ topics, onPick }: { topics: Topic[]; onPick: (t: Topic) => void }) {
  return (
    <ChartCard title="Topic registry" subtitle={`${topics.length} topics · sortable; click a row for detail`}>
      <DataTable<Topic>
        rows={topics}
        onRow={onPick}
        initialSort={{ key: "hot_score", desc: true }}
        maxHeight={440}
        columns={[
          { key: "label", label: "Topic", render: (t) => <span title={t.description ?? ""}>{t.label}</span> },
          { key: "brand", label: "Brand" },
          { key: "status_lifecycle", label: "Lifecycle", render: (t) => <LifecycleChip lifecycle={t.status_lifecycle} isNew={t.is_new} /> },
          { key: "status", label: "Status", render: (t) => <AlertChip status={t.status} /> },
          { key: "size", label: "Size", numeric: true, render: (t) => fmtInt(t.size) },
          { key: "recent_volume", label: "Last 7d", numeric: true, render: (t) => fmtInt(t.recent_volume) },
          { key: "growth_rate", label: "Growth", numeric: true, render: (t) => fmtGrowth(t.growth_rate) },
          { key: "coherence", label: "Coherence", numeric: true, render: (t) => fmtNum(t.coherence, 3) },
          { key: "hot_score", label: "Hot score", numeric: true, render: (t) => fmtNum(t.hot_score) },
        ]}
      />
    </ChartCard>
  );
}

function Events({ events }: { events: EventRecall[] }) {
  if (!events.length) return null;
  const found = events.filter((e) => e.found).length;
  return (
    <ChartCard title="Known-event recall" subtitle={`${found} of ${events.length} labelled real-world events surfaced as a topic`}>
      <DataTable<EventRecall>
        rows={events}
        maxHeight={360}
        columns={[
          { key: "event", label: "Event" },
          { key: "brand", label: "Brand", render: (e) => e.brand ?? "any" },
          { key: "found", label: "Found", render: (e) => (
            <span className="row"><StatusIcon kind={e.found ? "done" : "failed"} label={e.found ? "Found" : "Missed"} />{e.found ? "yes" : "no"}</span>) },
          { key: "topic_label", label: "Best topic", render: (e) => e.topic_label ?? "–" },
          { key: "matched_records", label: "Records", numeric: true, render: (e) => fmtInt(e.matched_records) },
          { key: "recall_in_best_topic", label: "Recall", numeric: true, render: (e) => fmtPct(e.recall_in_best_topic) },
          { key: "topics_for_80pct", label: "Topics for 80%", numeric: true, render: (e) => fmtInt(e.topics_for_80pct) },
        ]}
      />
    </ChartCard>
  );
}

// -------------------------------------------------------------- main ---

export function Results({ job, dataset }: { job: Job | null; dataset: string }) {
  // The live job only feeds this view when it runs on the dataset shown.
  const liveJob = job?.dataset === dataset ? job : null;
  // Refetch the run list whenever the live job finishes another run.
  const done = liveJob?.completed_runs.length ?? 0;
  const runs = useFetch(() => api.runs(dataset), [dataset, done, liveJob?.status]);
  const list = (runs.data?.runs ?? []).filter((r) => !r.error);
  const [picked, setPicked] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);

  // Follow the newest run the live job produced, until the user picks a tab.
  useEffect(() => {
    if (!list.length) return;
    const latest = liveJob?.completed_runs[liveJob.completed_runs.length - 1];
    if (follow && latest && list.some((r) => r.name === latest)) setPicked(latest);
    else if (!picked || !list.some((r) => r.name === picked)) setPicked(list[0].name);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs.data, done]);
  useEffect(() => { setPicked(null); setFollow(true); }, [dataset]);

  const card = list.find((r) => r.name === picked) ?? null;
  const overview = useFetch(() => (picked && card ? api.run(dataset, picked) : Promise.resolve(null)),
    [dataset, picked, card?.completed_at]);
  const data = overview.data;

  const [brand, setBrand] = useState("all");
  const [includeSuppressed, setIncludeSuppressed] = useState(false);
  const [topic, setTopic] = useState<Topic | null>(null);
  const [view, setView] = useState<"overview" | "records">("overview");
  const [recordTopic, setRecordTopic] = useState("");
  useEffect(() => setRecordTopic(""), [dataset, picked]);

  const brands = useMemo(() => [...new Set((data?.topics ?? []).map((t) => t.brand))].sort(), [data]);
  const topics = useMemo(
    () => (data?.topics ?? []).filter((t) => (brand === "all" || t.brand === brand) && (includeSuppressed || !t.suppressed)),
    [data, brand, includeSuppressed],
  );

  const regenerating = liveJob?.status === "running" && card && !liveJob.completed_runs.includes(card.name);

  if (runs.error) return <div className="section"><div className="empty">Could not load results: {runs.error}</div></div>;
  if (!list.length) {
    return (
      <div className="section">
        <div className="section-head"><h2>Results</h2></div>
        <div className="empty">
          {runs.loading ? "Loading…" : `No finished runs in ${runs.data?.out_dir ?? "the output directory"} yet. Results appear here as soon as the base run finishes.`}
        </div>
      </div>
    );
  }

  const emerging = topics.filter(isEmerging);
  const alerts = topics.filter((t) => t.status === "HOT" || t.status === "TRENDING");

  return (
    <div className="section">
      <div className="section-head">
        <h2>Results</h2>
        <p>{runs.data?.out_dir}</p>
      </div>

      <div className="tabs" role="tablist" aria-label="Runs">
        {list.map((r) => (
          <button key={r.name} role="tab" className="tab" aria-selected={r.name === picked}
            onClick={() => { setPicked(r.name); setFollow(false); }}>
            <span className="t1">{runTitle(r)}</span>
            <span className="t2" title={`${r.emerging} emerging incl. suppressed`}>
              {r.emerging_kept} emerging{r.emerging > r.emerging_kept ? ` (+${r.emerging - r.emerging_kept})` : ""} · {fmtDuration(r.total_seconds)}
            </span>
          </button>
        ))}
      </div>

      {regenerating && (
        <div className="banner info" style={{ marginBottom: 12 }}>
          <StatusIcon kind="running" label="Running" />
          <span>Showing the previous output of <strong>{card?.name}</strong> (finished {fmtDate(card?.completed_at, true)}). It will be replaced when the live run reaches it.</span>
        </div>
      )}

      <div className="filters" role="group" aria-label="Filters">
        <div className="seg" role="group" aria-label="View">
          <button aria-pressed={view === "overview"} onClick={() => setView("overview")}>Overview</button>
          <button aria-pressed={view === "records"} onClick={() => setView("records")}>Records</button>
        </div>
        <label className="lbl" htmlFor="brand">Brand</label>
        <select id="brand" className="select" value={brand} onChange={(e) => setBrand(e.target.value)}>
          <option value="all">All brands</option>
          {brands.map((b) => <option key={b} value={b}>{b}</option>)}
        </select>
        <label className="check" style={{ alignItems: "center", display: view === "records" ? "none" : undefined }}>
          <input type="checkbox" checked={includeSuppressed} onChange={(e) => setIncludeSuppressed(e.target.checked)} style={{ marginTop: 0 }} />
          Include suppressed topics <span className="muted">(junk, residual, campaign)</span>
        </label>
        <span className="spacer" />
        {card && (
          <span className="muted" style={{ fontSize: 12 }}>
            {card.input} · window {fmtDate(card.window_start)} – {fmtDate(card.window_end)}
          </span>
        )}
      </div>

      {data?.meta.llm?.enabled && (data.meta.llm.failures ?? 0) > 0 && (
        <div className="banner critical" role="alert" style={{ marginBottom: 12 }}>
          <StatusIcon kind="warning" label="Warning" />
          <span>
            LLM labelling failed for {fmtInt(data.meta.llm.failed)} of {fmtInt(data.meta.llm.requested)} topics
            ({fmtInt(data.meta.llm.failures)} failed calls), so those kept c-TF-IDF keyword labels.
            {data.meta.llm.last_error && <> Last error: <span className="mono">{data.meta.llm.last_error}</span>.</>}
            {data.meta.llm.base_url && <> Endpoint: <span className="mono">{data.meta.llm.base_url}</span>.</>}
          </span>
        </div>
      )}

      {!data ? (
        <div className="empty">{overview.error ?? "Loading run…"}</div>
      ) : view === "records" && picked ? (
        <RecordsView dataset={dataset} run={picked} brand={brand} topics={data.topics}
          topic={recordTopic} onTopic={setRecordTopic} onOpenTopic={setTopic} />
      ) : (
        <div className={overview.loading ? "dim" : undefined}>
          <div className="tiles">
            <Tile label="Records" value={fmtInt(data.run.records)} foot={`${fmtInt(data.run.segments)} segments`} />
            <Tile label="Topics" value={fmtInt(topics.length)}
              foot={`${fmtInt(data.run.live_topics)} live · ${fmtInt(data.run.kept_topics)} published`} />
            <Tile label="Emerging" value={fmtInt(emerging.length)}
              foot={`${emerging.filter((t) => t.is_new).length} new this run`} />
            <Tile label="Alerts" value={fmtInt(alerts.length)} foot="HOT or TRENDING" />
            <Tile label="Mean coherence" value={fmtNum(data.run.mean_coherence, 3)} foot="NPMI, all topics" />
            <Tile label="UMAP drift" value={fmtNum(data.run.drift, 3)} foot="vs frozen manifold" />
            <Tile label="Run time" value={fmtDuration(data.run.total_seconds)}
              foot={`embedding on ${data.meta.embedding_device ?? "?"}`} />
          </div>

          <EmergingTopics topics={topics} data={data} onPick={setTopic} />

          <div className="section-head" style={{ marginTop: 28 }}>
            <h2>Analytics</h2>
            <p>{runTitle(data.run)} · all charts follow the filters above</p>
          </div>
          <div className="grid g2">
            <DailyVolume data={data} topics={topics} />
            <GrowthMap topics={topics} onPick={setTopic} />
          </div>
          <div className="grid g2" style={{ marginTop: 16, alignItems: "start" }}>
            <TopicsByBrand topics={topics} />
            <StageTiming timing={data.run.timing} total={data.run.total_seconds} />
            <AlertStatus topics={topics} />
            <Events events={data.events} />
          </div>
          <div style={{ marginTop: 16 }}>
            <TopTopics topics={topics} onPick={setTopic} />
          </div>
        </div>
      )}

      {list.length > 1 && (
        <>
          <div className="section-head" style={{ marginTop: 28 }}>
            <h2>Across runs</h2>
            <p>Base registry, then each stream batch as it lands</p>
          </div>
          <div style={{ display: "grid", gap: 16 }}>
            <RunTimeBreakdown runs={list} />
            <RunTrends runs={list} />
          </div>
        </>
      )}

      {topic && data && picked && (
        <TopicDrawer dataset={dataset} run={picked} topic={topic} data={data} onClose={() => setTopic(null)}
          onViewRecords={() => { setRecordTopic(topic.topic_id); setView("records"); setTopic(null); }} />
      )}
    </div>
  );
}
