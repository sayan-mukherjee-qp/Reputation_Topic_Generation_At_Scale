import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { fmtDate, fmtInt, fmtNum } from "../lib/format";
import { useFetch } from "../lib/hooks";
import type { RecordsQuery, Topic } from "../lib/types";

type Status = RecordsQuery["status"];

const STATUSES: { key: Status; label: string }[] = [
  { key: "all", label: "All" },
  { key: "assigned", label: "With a topic" },
  { key: "unassigned", label: "Unassigned" },
  { key: "emerging", label: "In emerging topics" },
];

/** 1 … 4 5 [6] 7 8 … 20 */
function pageWindow(page: number, pages: number): (number | "…")[] {
  const out: (number | "…")[] = [];
  const add = (n: number) => out.push(n);
  const lo = Math.max(2, page - 2);
  const hi = Math.min(pages - 1, page + 2);
  add(1);
  if (lo > 2) out.push("…");
  for (let n = lo; n <= hi; n++) add(n);
  if (hi < pages - 1) out.push("…");
  if (pages > 1) add(pages);
  return out;
}

export function RecordsView({ dataset, run, brand, topics, topic, onTopic, onOpenTopic }: {
  dataset: string;
  run: string;
  brand: string;                         // from the Results filter row; "all" for none
  topics: Topic[];                       // this run's registry, for the topic picker
  topic: string;                         // "" for none
  onTopic: (id: string) => void;
  onOpenTopic: (t: Topic) => void;
}) {
  const [qInput, setQInput] = useState("");
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<Status>("all");
  const [order, setOrder] = useState<RecordsQuery["order"]>("newest");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // Search as you type, without a request per keystroke.
  useEffect(() => {
    const id = setTimeout(() => setQ(qInput.trim()), 300);
    return () => clearTimeout(id);
  }, [qInput]);
  // Any change of filter starts again from page 1.
  useEffect(() => setPage(1), [dataset, run, brand, topic, q, status, order, pageSize]);

  const query: RecordsQuery = {
    page, page_size: pageSize, status, order,
    brand: brand === "all" ? undefined : brand, topic: topic || undefined, q: q || undefined,
  };
  const res = useFetch(() => api.records(dataset, run, query), [dataset, run, JSON.stringify(query)]);
  const data = res.data;
  const byId = new Map(topics.map((t) => [t.topic_id, t]));
  const options = [...topics]
    .filter((t) => brand === "all" || t.brand === brand)
    .sort((a, b) => a.brand.localeCompare(b.brand) || a.label.localeCompare(b.label));
  const first = data && data.total ? (data.page - 1) * data.page_size + 1 : 0;
  const last = data ? Math.min(data.total, data.page * data.page_size) : 0;

  return (
    <section className="section" style={{ marginTop: 20 }}>
      <div className="section-head">
        <h2>Records</h2>
        <p>Every record in this run with the topics it was assigned to, best match first. Click a tag to filter by it.</p>
      </div>

      <div className="filters" role="group" aria-label="Record filters">
        <input className="search" type="search" placeholder="Search text…" value={qInput}
          onChange={(e) => setQInput(e.target.value)} aria-label="Search record text" style={{ minWidth: 220 }} />
        <select className="select" value={topic} onChange={(e) => onTopic(e.target.value)} aria-label="Topic"
          style={{ maxWidth: 320 }}>
          <option value="">All topics</option>
          {options.map((t) => (
            <option key={t.topic_id} value={t.topic_id}>
              {t.brand} · {t.label}{t.status_lifecycle.endsWith("EMERGING") ? " (emerging)" : ""}
            </option>
          ))}
        </select>
        <div className="seg" role="group" aria-label="Assignment">
          {STATUSES.map((s) => (
            <button key={s.key} aria-pressed={status === s.key} onClick={() => setStatus(s.key)}>{s.label}</button>
          ))}
        </div>
        <span className="spacer" />
        <select className="select" value={order} onChange={(e) => setOrder(e.target.value as RecordsQuery["order"])}
          aria-label="Order">
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
        </select>
      </div>

      <div className="row wrap" style={{ marginBottom: 10, fontSize: 13 }}>
        <span className="ink2">
          {data
            ? data.total
              ? <>Showing <strong>{fmtInt(first)}–{fmtInt(last)}</strong> of <strong>{fmtInt(data.total)}</strong> records</>
              : "No records match these filters"
            : res.error ?? "Loading records…"}
        </span>
        {data && (
          <span className="muted">· {fmtInt(data.all_records)} in the run, {fmtInt(data.unassigned)} with no topic</span>
        )}
        {topic && (
          <button className="chip chip-btn" onClick={() => onTopic("")} title="Clear the topic filter">
            topic: {byId.get(topic)?.label ?? topic} ✕
          </button>
        )}
      </div>

      <div className={`card records ${res.loading ? "dim" : ""}`} style={{ padding: 0 }}>
        {data?.records.map((r) => (
          <article key={r.record_id} className="record">
            <div className="record-meta">
              <span className="num">{fmtDate(r.event_time, true)}</span>
              <span>{r.brand}</span>
              <span className="muted mono">#{r.record_id}</span>
            </div>
            <p className="record-text">{r.text}</p>
            <div className="row wrap" style={{ gap: 6 }}>
              {r.topics.length === 0 && <span className="chip muted">no topic</span>}
              {r.topics.map((tag, i) => {
                const meta = data.topics[tag.topic_id];
                const emerging = meta?.lifecycle.endsWith("EMERGING");
                return (
                  <span key={tag.topic_id} className="tag-group">
                    <button className={`chip chip-btn tag ${meta?.suppressed ? "tag-suppressed" : ""} ${i === 0 ? "tag-best" : ""}`}
                      onClick={() => onTopic(tag.topic_id)}
                      title={`${tag.topic_id}${tag.similarity != null ? ` · similarity ${fmtNum(tag.similarity, 3)}` : " · recovered"}${meta ? ` · ${meta.lifecycle.toLowerCase()} · ${meta.status.toLowerCase()}` : ""}${meta?.suppressed ? " · suppressed" : ""} — click to filter`}>
                      {emerging && <span className="dot" style={{ background: "var(--series-2)" }} />}
                      {meta?.label ?? tag.topic_id}
                      {tag.similarity != null && <span className="muted num" style={{ fontSize: 10.5 }}>{fmtNum(tag.similarity, 2)}</span>}
                    </button>
                    {byId.get(tag.topic_id) && (
                      <button className="tag-open" aria-label={`Open topic ${meta?.label ?? tag.topic_id}`}
                        title="Open topic detail" onClick={() => onOpenTopic(byId.get(tag.topic_id)!)}>↗</button>
                    )}
                  </span>
                );
              })}
            </div>
          </article>
        ))}
        {data && data.total === 0 && <div className="empty" style={{ border: 0 }}>No records match these filters.</div>}
      </div>

      {data && data.pages > 1 && (
        <nav className="pager" aria-label="Pages">
          <button className="btn btn-sm" disabled={data.page <= 1} onClick={() => setPage(data.page - 1)}>‹ Prev</button>
          {pageWindow(data.page, data.pages).map((n, i) =>
            n === "…" ? (
              <span key={`e${i}`} className="muted">…</span>
            ) : (
              <button key={n} className="btn btn-sm" aria-current={n === data.page ? "page" : undefined}
                onClick={() => setPage(n)}>{fmtInt(n)}</button>
            ),
          )}
          <button className="btn btn-sm" disabled={data.page >= data.pages} onClick={() => setPage(data.page + 1)}>Next ›</button>
          <span className="spacer" />
          <label className="muted" style={{ fontSize: 12 }}>
            Per page{" "}
            <select className="select" value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))}>
              {[25, 50, 100].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
        </nav>
      )}
    </section>
  );
}
