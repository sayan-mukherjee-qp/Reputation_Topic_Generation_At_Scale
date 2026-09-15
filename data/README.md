# Data

Two files in this directory are **not tracked in git**. Both exceed GitHub's
100 MB per-file limit, and both are regenerable from a public dataset:

| File | Rows | Size | Status |
|---|---:|---:|---|
| `twcs.csv` | 3,002,523 | 493 MB | untracked — download (see below) |
| `twcs_prepared.csv` | 1,585,950 | 224 MB | untracked — rebuild (see below) |
| `twcs_subset.csv` | 151,755 | 22 MB | tracked |
| `twcs_subset_150k.csv` | 160,588 | 23 MB | tracked |
| `twcs_subset_200k.csv` | 214,340 | 31 MB | tracked |
| `stream/stream_200k.csv` | 218,732 | 32 MB | tracked |
| `stream/chunk_1..4.csv` | ~54k each | 7-9 MB | tracked |

The tracked subsets are what the pipeline is normally run against, so a fresh
clone can run everything without rebuilding anything.

## Rebuilding the two untracked files

`twcs.csv` is the **Customer Support on Twitter** dataset by Stuart Axelbrooke,
published on Kaggle under CC BY-NC-SA 4.0:

    https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Download it and put `twcs.csv` in this directory. Schema:

    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

Then derive `twcs_prepared.csv` — inbound customer messages only, with the
brand each one was addressed to resolved and timestamps normalised:

    cd ../reputation_topic_gpu
    python prepare_twcs.py            # ../data/twcs.csv -> ../data/twcs_prepared.csv

Resulting schema: `tweet_id, event_time, author_id, text, brand`.

## Rebuilding the subsets

Only needed if you want to change the sampling. Each takes a contiguous 61-day
window (2017-10-01 to 2017-12-01) and applies a time-stratified per-brand cap,
so each brand's daily volume shape survives the downsampling:

    cd ../reputation_topic_gpu
    python make_subset.py             # -> ../data/twcs_subset_150k.csv

`twcs_subset_200k.csv` is the same procedure with a larger per-brand cap.

The streaming chunks are 200k *different* records for the same brands from the
same window, ordered by event time and cut into four equal arrivals:

    cd ../reputation_topic_gpu
    python build_stream_batch.py      # -> ../data/stream/

Note what this can and cannot test: the corpus ends 2017-12-03, so these are
disjoint records from the same window delivered in time order, not a genuine
forward-in-time extrapolation.
