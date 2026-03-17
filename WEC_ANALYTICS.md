# WEC Analytics — Project Guide

## Project Overview

This is a Python project called **wec-analytics** — an open source library and analysis toolkit for FIA World Endurance Championship (WEC) and European Le Mans Series (ELMS) racing data.

The project has two intertwined goals:
1. **Educational** — deepen the developer's understanding of Data Mining, AI/ML, and Computer Vision as taught in their university modules
2. **Portfolio** — produce a real, reusable open source library that demonstrates motorsport domain knowledge to potential employers (specifically Hyundai Motorsport GmbH)

The project is built in **three phases**. We are currently working on **Phase 1: Data Ingestion & Lap Analysis**.

---

## Developer Context

- Third-year Computer Science student (TU Dublin), currently on exchange at Kyungpook National University, South Korea
- Current modules: **Data Mining Theory & Applications**, **Computer Vision**, and Korean Language
- Python level: **Intermediate** — comfortable with OOP, some experience with libraries
- Goal: build understanding of data mining concepts through hands-on motorsport application
- Career target: IT/software internship at a WEC/endurance motorsport team (e.g. Hyundai Motorsport GmbH, Alzenau, Germany) in 2027

---

## Teaching & Explanation Style

When explaining concepts or making implementation decisions, Claude should:

- **Lead with intuition, then formalism** — explain what something *does* and *why it matters* before showing the code or formula
- **Use motorsport analogies** — relate data concepts to racing (e.g. "clustering is like grouping cars by tyre strategy")
- **Flag module connections explicitly** — when a concept maps to something from Data Mining or Computer Vision lectures, say so
- **Ask Socratic questions** when the developer seems to be going in the wrong direction
- **Proactively flag misconceptions** — if a common mistake is likely, mention it before it happens
- **Explain library choices** — don't just use pandas/numpy/sklearn; briefly explain *why* that tool is right for this task

---

## Project Structure

```
wec-analytics/
├── wec_analytics/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── alkamelsystems.py      # Al Kamel Systems CSV fetcher & parser
│   │   └── models.py              # Data models (Session, Lap, Stint, Driver)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── laps.py                # Lap time analysis, outlier detection
│   │   ├── stints.py              # Stint detection, driver change identification
│   │   ├── strategy.py            # Pit stop analysis, undercut/overcut detection
│   │   └── multiclass.py          # Multi-class race analysis (Hypercar vs LMP2 vs LMGT3)
│   ├── ml/                        # Phase 2 — leave empty for now
│   │   └── __init__.py
│   └── vision/                    # Phase 3 — leave empty for now
│       └── __init__.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lap_analysis.ipynb
│   └── 03_stint_strategy.ipynb
├── tests/
│   ├── test_ingestion.py
│   └── test_analysis.py
├── data/
│   └── samples/                   # Small cached CSV samples for offline dev
├── docs/
│   └── concepts.md                # Running glossary of data mining concepts learned
├── README.md
├── WEC_ANALYTICS.md               # This file
├── requirements.txt
└── setup.py
```

---

## Phase 1 Scope — Data Ingestion & Lap Analysis

### What We Are Building

A clean Python library that:
1. Fetches WEC session data from Al Kamel Systems (the official WEC timing provider)
2. Parses it into structured pandas DataFrames
3. Provides analysis functions for lap times, stints, and pit stop strategy
4. Exposes a clean, well-documented public API

### Data Source

**Al Kamel Systems** is the official timing provider for FIA WEC (since 2012) and ELMS. Historical session CSV files are publicly accessible at:

```
http://fiawec.alkamelsystems.com/Results/
```

URL pattern example:
```
http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV
```

Each CSV contains per-lap records with columns including: `LAP_NUMBER`, `LAP_TIME`, `DRIVER_NUMBER`, `CROSSING_FINISH_LINE_IN_PIT`, `PIT_TIME`, `CAR_DRIVER`, `CLASS`, `TEAM` and others.

**Important note on data rights:** Al Kamel Systems owns this data. The library should clearly document this in its README and should only access publicly available historical result files (not live timing). This is consistent with how `fastf1` handles the F1 data feed.

### Phase 1 Milestones

#### Milestone 1 — Raw Data Ingestion
- `alkamelsystems.py`: function to download a CSV by URL and load into a raw DataFrame
- Handle encoding issues (files sometimes use non-UTF-8 encoding)
- Handle `%20`-encoded spaces in URLs
- Cache downloaded files locally to avoid repeated network requests
- Write unit tests for the fetcher

**Data Mining concept to learn:** *Data acquisition and the raw-to-structured pipeline. Understanding what "dirty data" looks like before cleaning.*

#### Milestone 2 — Data Cleaning & Parsing
- Parse `LAP_TIME` from `MM:SS.mmm` string format to float seconds
- Detect and flag in-laps and out-laps (pit entry/exit laps)
- Handle missing values and corrupted rows
- Normalise column names (strip whitespace, lowercase, snake_case)
- Create a `Session` dataclass that holds cleaned lap data with metadata

**Data Mining concept to learn:** *Data preprocessing — one of the most important and time-consuming steps in any real data pipeline. Covers: type coercion, missing value handling, outlier identification.*

#### Milestone 3 — Stint Detection
- Identify driver stints (continuous blocks of laps by the same driver)
- Detect driver changes within a car
- Calculate per-stint statistics: avg lap time, fastest lap, degradation slope
- Flag safety car / slow laps using statistical outlier detection (Z-score or IQR method)

**Data Mining concept to learn:** *Segmentation of time-series data. Outlier detection methods (Z-score, IQR). These are core Data Mining techniques directly applicable to module coursework.*

#### Milestone 4 — Pit Stop & Strategy Analysis
- Calculate pit stop durations
- Identify undercut/overcut strategy patterns
- Compare pit windows across car classes
- Visualise strategy as a horizontal stint chart (similar to F1 TV strategy graphics)

**Data Mining concept to learn:** *Feature engineering — deriving new meaningful variables from raw data. Pattern detection in sequential event data.*

#### Milestone 5 — Multi-Class Analysis
- WEC has 3 simultaneous classes: Hypercar, LMP2, LMGT3
- Build functions to filter, compare, and analyse data by class
- Analyse traffic impact: identify laps where a car was likely held up by a slower class

**Data Mining concept to learn:** *Multi-group analysis, stratified statistics. Particularly relevant to clustering concepts in the Data Mining module.*

#### Milestone 6 — Public API & Documentation
- Clean up public-facing functions with type hints and docstrings
- Write a `README.md` with installation, usage examples, and data source attribution
- Publish to GitHub with a meaningful commit history
- Optionally publish to PyPI as `wec-analytics`

---

## Coding Standards

- **Python 3.10+**
- Type hints on all public functions
- Docstrings in NumPy style (consistent with scientific Python libraries)
- `pandas` for all tabular data — do not use raw lists/dicts where a DataFrame is more appropriate
- `requests` for HTTP fetching
- `pytest` for all tests — aim for >80% coverage on ingestion and analysis modules
- Follow PEP8; use `black` for formatting if available
- No hardcoded URLs in business logic — use constants or config

---

## Key Libraries (Phase 1)

| Library | Purpose | Why this one |
|---|---|---|
| `pandas` | Core data structure | Industry standard for tabular data; used in real motorsport teams |
| `requests` | HTTP fetching | Simple, reliable, widely used |
| `numpy` | Numerical operations | Required for statistical calculations |
| `matplotlib` / `seaborn` | Visualisation | Matplotlib for control, seaborn for quick statistical plots |
| `pytest` | Testing | Standard Python testing framework |
| `dataclasses` | Data models | Clean, lightweight alternative to full ORM for session metadata |

---

## Module Connections Tracker

Use this section to note explicit connections between project work and university module content. Update it as you build.

### Data Mining Theory & Applications
- [ ] **Data preprocessing** → Milestone 2 (cleaning lap time strings, handling nulls)
- [ ] **Outlier detection (Z-score, IQR)** → Milestone 3 (safety car lap detection)
- [ ] **Feature engineering** → Milestone 4 (pit stop duration, tyre age features)
- [ ] **Clustering** → future: grouping cars by strategy type
- [ ] **Time series analysis** → stint degradation modelling
- [ ] **Association rules** → future: tyre compound vs lap time patterns

### Computer Vision (Phase 3 prep — note concepts as you encounter them)
- [ ] Object detection basics — will apply to car detection in broadcast footage
- [ ] Colour-based classification — livery detection by HSV colour space
- [ ] Frame extraction from video — OpenCV basics

### AI / ML (Phase 2 prep)
- [ ] Supervised learning setup — train/test split on historical WEC sessions
- [ ] Regression — lap time prediction from features
- [ ] Classification — predict likely pit window from stint data

---

## WEC Domain Knowledge Glossary

| Term | Meaning |
|---|---|
| **Stint** | A continuous block of laps by a single driver on one set of tyres |
| **In-lap** | The lap on which a car enters the pits |
| **Out-lap** | The lap immediately after exiting the pits (typically slower) |
| **Undercut** | Pitting earlier than a rival to gain track position via fresh tyres |
| **Overcut** | Staying out longer than a rival to gain time while they are stationary |
| **Safety Car (SC)** | Deployed after an incident; all cars must follow at reduced speed |
| **Full Course Yellow (FCY)** | WEC-specific; similar to SC but cars maintain position |
| **Hypercar** | Top class in WEC (e.g. Toyota GR010, Ferrari 499P, Porsche 963, Genesis GMR-001) |
| **LMP2** | Le Mans Prototype 2 — spec chassis, spec engine, customer teams |
| **LMGT3** | GT class using production-based GT3 cars (Porsche 911, BMW M4, etc.) |
| **Al Kamel Systems** | Official timing provider for FIA WEC and ELMS since 2012 |
| **Balance of Performance (BoP)** | Technical regulations used to equalise performance between manufacturers |
| **Driver change** | Mandatory in endurance racing — cars must have multiple drivers per race |

---

## Phase 2 Preview — AI/ML (do not build yet)

When Phase 1 is complete, Phase 2 will add:
- Lap time prediction model (regression) trained on features extracted in Phase 1
- Pit stop window classifier — given current stint age and lap delta, predict optimal pit lap
- Tyre degradation curve fitting using historical stint data
- Model explainability — understanding *why* the model makes predictions

---

## Phase 3 Preview — Computer Vision (do not build yet)

Phase 3 will add:
- Extract frames from publicly available WEC/ELMS race footage (YouTube)
- Safety Car period detection from video (flags, pace car appearance)
- Car class identification by livery colour using HSV colour space analysis
- Overlay timing data onto video frames (telemetry-style graphics)

---

## Questions to Guide Development Sessions

- *"Let's implement Milestone [N]. Explain the concept behind what we're building before we write any code."*
- *"Review my implementation of [function]. Does it follow good data engineering practice? What would a professional data engineer change?"*
- *"I'm seeing [error/unexpected output]. Walk me through debugging this step by step."*
- *"Connect what we just built to what I'd be learning in my Data Mining module — what's the theoretical concept here?"*
- *"How would a real WEC team use what we just built? What would they do differently with access to full telemetry?"*
