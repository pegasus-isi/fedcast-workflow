# Author feedback — 2026-10-06

Reply from the paper's first author (Zhe Xu) to our 2026-08-31 question list.
It covers only what is already public in the eScience paper and its appendix.
Non-public materials (split manifests, the full benchmark table with geographic
footprints, trained checkpoints, per-metric result tables, MCT TOPSIS inputs, MCT
source) stay unavailable until the author is back in the US; a follow-up was
promised then.

This file records what the reply settles, how it maps onto this workflow, and
what changes as a result. SPEC.md Sec. 6 is updated to point here.

## 1. What the reply settles

### Data construction

- **Study interval:** 2020-11 through 2024-10 (48 months). The retained-sequence
  counts (542, 478, 489, 885, 839, 715, 831) are what survived construction,
  QC, and filtering inside that interval.
- **Crops:** 16 × 256 × 256 (T × H × W); samples are 4 input + 12 target frames
  at the nominal 2-min MRMS cadence.
- **Values:** valid precipitation capped at 128 mm/h and quantized in steps of
  1/32.
- **Precipitation-content filter**, over the whole 16-frame crop:

  ```
  R_sat = 1 - exp(-R / s),               s = 1.0
  q     = min(1, q_min + m * mean(R_sat)),  q_min = 2e-4, m = 0.1
  keep if q >= 8e-3
  ```

  (This is the DGMR importance-sampling statistic used as a hard threshold.)
- **Gaps / corrupt files:** a sequence that cannot be built from 16 complete
  frames is dropped; no interpolation.
- **PAHG:** Alaska-domain PrecipRate from `noaa-mrms-pds`, same processing as
  CONUS.
- **Split:** first three available days of each month → test; validation sampled
  from the remainder to about the test size with **seed 2025**; the rest is
  train. Splits frozen and shared by centralized and federated runs.

### Event benchmark

12 events, one predefined forecast initialization each (so ~12 forecast cases per
model). Chosen for diversity over intensity, coverage/density, and spatial
complexity: two high-end and two median-ranked per characteristic, moving down
the ranking on duplicates. Not balanced by site, season, or source catalog.

| Event ID | Window (UTC) | Centroid (lat, lon) | Selection |
|---|---|---|---|
| 20220930_1746 | 2022-09-30 17:46–18:06 | 37.0000, −76.5300 | Max-1 intensity |
| 20240110_0048 | 2024-01-10 00:48–01:08 | 36.9000, −77.5000 | Max-2 intensity |
| 20241122_1252 | 2024-11-22 12:52–13:12 | 42.5100, −73.9900 | Median-1 intensity |
| 20220307_2120 | 2022-03-07 21:20–21:40 | 42.3100, −74.0700 | Median-2 intensity |
| 20220103_1310 | 2022-01-03 13:10–13:30 | 37.1900, −77.0900 | Max-1 density |
| 20250111_0714 | 2025-01-11 07:14–07:34 | 37.0800, −77.0800 | Max-2 density |
| 20240809_1648 | 2024-08-09 16:48–17:08 | 42.5000, −74.1700 | Median-1 density |
| 20220203_0024 | 2022-02-03 00:24–00:44 | 34.7100, −106.1600 | Median-2 density |
| 20220103_1206 | 2022-01-03 12:06–12:26 | 37.1900, −77.0900 | Max-1 complexity |
| 20220103_1030 | 2022-01-03 10:30–10:50 | 37.1900, −77.0900 | Max-2 complexity |
| 20250720_0048 | 2025-07-20 00:48–01:08 | 37.2700, −77.2700 | Median-1 complexity |
| 20240404_1916 | 2024-04-04 19:16–19:36 | 42.4200, −74.0700 | Median-2 complexity |

### Paired instances and training windows

- n = (2, 16, 8, 4, 2, 1) for L = (1, 3, 6, 12, 24, 48) months counts **paired
  date-tagged trained models** (one centralized + one federated per date tag),
  not events or stochastic repeats.
- Each window ends at its date tag and spans the preceding L calendar months
  (rolling, not all anchored at one end).
- Software: DGMR from Open Climate Fix `skillful_nowcasting` (maintained inside
  the authors' SPRITE codebase), Flower, PyTorch Lightning.
- No global training seed; training is not bitwise deterministic.
- Hardware: one A100 per job. 48-month, 100 epochs/rounds: centralized
  4 d 12:14:09 (~108.2 GPU-h), federated 5 d 11:16:52 (~131.3 GPU-h), including
  validation, aggregation, logging, checkpointing.
- Federated: all 7 clients every round, one local epoch each, same batch size
  for every client.

### Evaluation and MCT

- **TOPSIS pools:** E1 = all centralized and federated DGMR candidates across all
  L and all date tags, plus STEPS, in **one** pool. E2.1 = centralized reference
  + E2.1 federated ablation + STEPS. E2.2 = **one pool per SAM configuration**
  (that ρ's centralized-SAM models + unchanged federated models + STEPS). Equal
  metric weights; ideal best/worst by benefit vs. cost metric.
- **Categorical metrics:** per event and per lead, averaged over the 12 leads and
  the events. Not pooled into one global contingency table. Threshold 0.1 mm/h.
- **Ensembles:** deterministic verification on the pixel-wise ensemble mean (in
  rain-rate space, before thresholding); probabilistic verification on the full
  ensemble.
- **Executing_Time:** per forecast instance, all methods in one environment — a
  relative measure.
- **STEPS:** Lucas-Kanade motion, 12 steps, 20 members, 6 cascade levels, AR
  order 2, semilagrangian extrapolation, FFT decomposition, Gaussian bandpass,
  nonparametric noise, BPS velocity perturbation, incremental mask, CDF
  probability matching, threshold −10 dB, 2 km/pixel, 5-min time step, seed 24.

## 2. Gap analysis against this workflow (as of e8ff341)

### Already consistent

| Item | Where |
|---|---|
| Per-event, per-lead categorical scores, then averaged | `bin/mct_verify.py` |
| Ensemble mean before the 0.1 mm/h threshold; CRPS on the full ensemble | `bin/mct_verify.py` |
| Test = first 3 days/month; val ≈ test size | `bin/preprocess_sequences.py` |
| Gapped sequences dropped, never interpolated | `bin/preprocess_sequences.py` (`scan_buffer`) |
| One local epoch per client per round, all clients each round | `bin/fl_train_client.py`, `fl_round.py` |
| E2.1 has its own pool | `workflow_generator.py` |
| PAHG from the Alaska product hierarchy | resolved earlier, SPEC Q3 |

### A — parameter and default changes (small, unambiguous)

| Item | Before | After |
|---|---|---|
| Study window | `--start-month 2021-01` | `2020-11` |
| Split seed | `--split-seed 1337` | `2025` |
| Precipitation filter | ≥5 % wet pixels (>0.1 mm/h) in any frame | `q ≥ 8e-3` on the whole crop, formula above |
| Value encoding | float16, uncapped | capped at 128, quantized to 1/32 |
| Model grid | center-crop 300 → 288 | 256 × 256 |
| STEPS | `kmperpixel=1`, `timestep=2`, no seed | `kmperpixel=2`, `timestep=5`, `seed=24`, other settings explicit |
| E2.2 pools | both ρ in one pool | one pool per ρ |

### B — benchmark rebuild (structural)

Our benchmark was compiled from WPC MPD / LSR / Storm Events and matched against
each client's **test split**. The author's benchmark is a fixed list of 12
events, and:

- 9 of 12 centroids (Virginia, New Mexico) are outside every client's window,
  so they cannot come from any client's test split;
- 3 events (2024-11, 2025-01, 2025-07) postdate the study window;
- one forecast initialization per event, not every overlapping sequence.

So the benchmark becomes the frozen table above, with its own MRMS fetch: a
16-frame, 256 × 256 crop centered on each event centroid. The three event-source
fetch jobs and `build_benchmark` leave the DAG.

### C — date-tagged rolling windows (structural)

Before this item we trained one model per (method, L) on the last L months. The paper trains
2 + 16 + 8 + 4 + 2 + 1 = 33 date-tagged models per paradigm and pools all of them
in E1's TOPSIS. Estimated E1 cost from the author's timings: about
5 × (108 + 131) ≈ 1,200 A100-h (each fully tiled L costs about one 48-month run;
L = 1 adds ~10 GPU-h). The second reply (Sec. 5) confirms E2.1 and both E2.2
settings cover all 33 date tags, reusing the E1 centralized models (E2.1) and
federated models (E2.2) as references — so the new training is 33 E2.1
federated models (~660 A100-h) and 2 × 33 centralized-SAM models (~1,100 A100-h
before SAM's roughly 2× step cost): about 3,000–4,000 A100-h in total. The
L = 1 tags are 2022-02 and 2022-10. Implemented in the generator (see Status);
running it at full scale waits on the cost review. Unity (MGHPCC) is where it
would run.

### Status

- **A** and **B** implemented 2026-10-06 (with the TOPSIS candidate change and a
  CRPS normalization fix found while testing B: the spread term was K times too
  large, so CRPS went negative and rewarded noisier ensembles).
- **C** implemented 2026-10-06: `--date-tags rolling` (default) trains one
  model per (method, L, date tag) on the L months ending at the tag; `last`
  keeps the old layout. Models, checkpoints and metrics are named
  `{method}_L{L}_{YYYYMM}`. TOPSIS keys candidates by date tag too. The
  report's gates use the median over each L's models and list the count
  per L against the paper's n. At full scale E1 is 66 models and
  4,630 parent jobs plus 3,300 FL-round sub-workflows (was 12 models and
  1,837 jobs).
- Second reply (Sec. 5) implemented 2026-10-06: client crop jitter
  (`--crop-jitter`, default 31) and `--steps-grid {paper,mrms}`.

## 3. Open questions for the author

1. Event windows are 20 min; a 4 + 12 sample at 2-min cadence spans 30 min
   (32 min to the last target's valid time). How does the window map onto the
   initialization time?
2. Is each event's 256 × 256 crop centered on its centroid? For client data, is
   the 256 crop the center of the 300 × 300 site window, or are several crops
   taken per window?
3. Which two months are the L = 1 date tags?
4. Did E2.1 and E2.2 train every date tag, or a subset?
5. Were the 2 km / 5 min STEPS parameters intended for 1 km / 2-min MRMS input?
   (We match them as stated.)
6. Event 20220307_2120 (21:20–21:40): the public MRMS archive has no
   2022-03-07 21:24 PrecipRate file, and every 16-frame, 2-min-cadence sample
   anchored on that window contains 21:24. Was its initialization outside the
   window, or did benchmark construction tolerate a missing frame? (Found on
   the first Unity pilot, 2026-10-06.)

## 4. Our working rules where the reply is silent

These are documented choices, revisable when the author answers.

- **Event initialization (Q1):** the event window's start is the first target
  frame. With `ws` the window start, the 16 frames are `ws − 8 min .. ws + 22 min`
  at 2-min steps: inputs `ws − 8 .. ws − 2`, targets `ws .. ws + 22`. All 12
  window starts are on the even-minute grid already.
- **Archive gaps in an event sample (Q6):** if the default sample has a gap,
  the initialization slides in 2-min steps, nearest first, to the closest
  complete sample whose targets still overlap the event window. The shift is
  recorded per event (`init_shift_s` in `benchmark_sequences.npz`); for
  20220307_2120 it is +14 min (frames 21:26–21:56).
- **Rolling windows (item C):** for L > 1 the n = 48 / L date tags tile the
  archive with consecutive, non-overlapping L-month windows anchored at its
  last month (2024-10): L = 3 tags 2021-01, 2021-04, …, 2024-10. The reply gives
  the counts and says each window ends at its tag, which fits tiling, but does
  not state the tags. L = 1 uses the two published tags. A sequence belongs to
  a window by its first frame's time.
- **Event crop (Q2):** 256 × 256 on the 0.01° grid, centered on the centroid
  — confirmed by the second reply. (Client crops are no longer our rule; see
  Sec. 5 item 2.)

## 5. Second reply (2026-10-06) — answers to Sec. 3

1. **Event timing — partly answered.** Frame layout confirmed: with t0 the last
   input's valid time, inputs t0 − 6 … t0 and targets t0 + 2 … t0 + 24 (30 min
   first to last). That is our layout. Where t0 sits relative to the 20-min
   event window is still not stated, so our rule in Sec. 4 stands. **Ask again**,
   with Q6.
2. **Crops — answered.**
   - Benchmark: the appendix centroid is the midpoint of the final crop's
     bounding box, so the crop is centered on it. Matches ours.
   - Clients: one 256 × 256 crop per 300 × 300 block, not tiles. Train and
     validation origins are offset independently by 0–31 px per axis (the same
     for every frame of the sequence); test crops sit at (0, 0) with no jitter.
     We used a fixed center crop (origin 22, 22). **Changed:**
     `preprocess_sequences --crop-jitter 31` draws each sequence's origin from
     an RNG keyed on (split seed, site, start time), so shards stay
     reproducible and independent of chunking. The origins are stored in the
     shard (`crop_origin`). The filter runs on the jittered crop, as theirs did.
3. **L = 1 tags — answered:** 2022-02 and 2022-10 (training-period end
   months). Feeds item C.
4. **E2.1 / E2.2 coverage — answered:** all 33 date/window combinations; E2.1
   reuses the centralized reference models, both E2.2 settings reuse the
   unchanged federated ones. Feeds item C (cost in Sec. 2 C).
5. **STEPS — answered.** 2 km / 5 min came from the official PySTEPS STEPS
   example; they affect velocity perturbation and incremental masking only, not
   DGMR. The author will re-run STEPS with MRMS-matched parameters and
   recompute the joint TOPSIS. **Changed:** `--steps-grid` chooses `paper`
   (2 km / 5 min, default — reproduces what was published) or `mrms` (1 km /
   2 min). Running both lets us check their sensitivity result independently.
6. **MRMS gap in 20220307_2120 — not answered.** Ask again.
