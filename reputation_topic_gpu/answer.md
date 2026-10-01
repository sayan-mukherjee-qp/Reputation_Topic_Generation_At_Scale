I checked all five points in the pipeline code, and the first turned up a real bug. Topic IDs were jumping between brands from one stream batch to the next. It's fixed and tested.

## 1. Does a topic belong to one brand only?

It does now. Before the fix, almost everything was already per brand:
- Topics are discovered separately for each brand.
- A record can only join a topic of its own brand.
- New emerging topics are only compared against their own brand's topics.
- All merging happens between topics of the same brand.

So Delta and American Airlines can each have their own "flight delays" topic, which is what you want.

**The gap was between stream batches.** Each batch re-matches its topics to the previous batch's topics to keep IDs stable, and that step ignored brand. So a Delta topic could take over an American Airlines topic's ID, start date and LLM label. On your 300k stream this happened to 12–34 topics per batch. For example, `T46` "Flight delays and baggage wait times" bounced between Delta and American Airlines.

**Fix:** a topic can now only inherit an ID from the same brand's previous topics. I replayed the 6 small stream batches: brand switches went from 1–23 per batch to 0, and ID stability and event detection stayed the same.

This applies to new runs only, so run the stream again to get clean IDs. One edge case remains by design: brands with fewer than 80 records share one pooled group. All 12 of your brands are well above that.

## 2. Why at most 2 topics per record?

It isn't hardcoded. It's a setting, `--max-topics-per-segment`, and 2 is the default.
- **The limit is per piece of text, not per record.** Long records are split into up to 6 pieces, each with up to 2 topics. Tweets are short, though, so in practice every record has at most 2.
- **Why 2:** every extra topic counts the same record again, which inflates topic volumes and hot scores.
- **What the data shows:** 65% of tweets already take 2 topics, and the second is usually a weaker runner-up, typically 0.075 less similar than the first.
- **A better rule than a bigger number:** "take every topic nearly as good as the best one." The code supports this (`--secondary-margin`), but it was switched off because it made known events harder to detect (recall dropped from 0.51 to 0.43).

I can run a test with 3 topics plus that rule and compare the numbers.

## 3. How are topics merged, and are LLM labels used?

There are four merge steps, all within one brand:

| Merge | When | Uses LLM label? |
|---|---|---|
| Duplicates | Two topics are nearly identical (similarity ≥ 0.92) | No |
| Same label | The LLM gave both the same label, and they're reasonably similar (≥ 0.70) | **Yes** |
| Too many topics | Over 600 live topics: merge the most similar pairs (≥ 0.75), newest emerging topics protected | No |
| New vs existing | A new cluster that clearly matches an existing topic joins it instead of becoming new | No |

LLM labels are used in two other ways:
- A topic whose meaning barely changed keeps its label across batches.
- A topic the LLM says has no real subject (thank-yous, small talk) is hidden from alerts.

One limitation: the same-label merge needs an exact match, so "Flight delays" and "Flight delay issues" don't merge.

## 4. How is a topic's status decided?

Each topic has two separate statuses.

**Lifecycle (is it new?)**
- **Emerging:** a new topic that didn't match any existing one.
- **Micro-emerging:** a tiny but very tight new group, at least 3 records from at least 3 different people.
- **Active:** an existing topic.
- **Dormant:** kept from the previous batch but no records yet. It's removed if it stays empty while its brand is still active.

**Trend (is it heating up?)** This compares the last 7 days with the 21 days before them:

| Status | Rule |
|---|---|
| Low evidence | 2 or fewer records in the last 7 days |
| **Hot** | Hot score ≥ 0.75 and more than 50% growth |
| **Trending** | Hot score ≥ 0.55 and more than 20% growth |
| Growing | More than 5% growth |
| Declining | More than 20% drop |
| Stable | Anything else |

Hot and Trending are alerts. An alert is dropped if the topic is incoherent, has fewer than 250 records, or is mostly small talk.

## 5. What is the hot score?

It's a number from 0 to 1 for how hot a topic is right now, compared with the other topics in the same run. Each ingredient is scaled against the other topics, then blended:

- 25%: volume, meaning records in the last 7 days
- 25%: growth, the last 7 days against normal
- 20%: speed, the last 2 days against the 5 before
- 15%: spike, how unusual this week is
- 10%: "recency"
- 5%: steadiness, how many of the last 7 days had activity

A score of 0.8 means "near the top of this run", not an absolute level. Compare scores within one run only.

One thing I noticed: "recency" is currently calculated exactly like volume, so volume really counts for 35%. That looks unintended. I didn't change it, because fixing it would change every score; I can fix it if you want.