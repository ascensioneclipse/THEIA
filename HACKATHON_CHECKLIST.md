# NASA Space Apps 2026: submission checklist

**Do these yourself before submitting.** Rules and deadlines change every year. This list says what
to check, not what the rules say.

## 1. Official documents: read them, don't trust memory

- [ ] **2026 Judging and Awards Guide** (spaceappschallenge.org → Resources / Judging). Check:
  - the five criteria (Impact, Creativity, Validity, Relevance, Presentation) and how each is scored
  - what global nominees must submit (video length, slides, links)
- [ ] **Participant rules / terms**, especially:
  - the rules on **AI tool use**
  - **originality**
  - **eligibility** of your local event for global nomination
- [ ] **Challenge page:** "Identify Earth Locations that Analog the Permanent Moon Base Locations and
  Mars" (spaceappschallenge.org/2026/challenges/permanent-moon-base-locations-and-mars). Re-read the
  wording and the **Resources** tab, then mention the NASA resources you used.
- [ ] **Deadlines:** today is after the October hackathon weekend. Check whether your project is in
  local judging, global judging or a post-event phase, and what can still be changed.

## 2. AI disclosure

- [ ] State clearly in the project description, README and slides that an **AI coding assistant**
  helped write the software and documentation.
- [ ] Make sure you can explain every part yourself (see `DEFENSE_GUIDE.md`).
- [ ] Do not present AI-written text as your own personal writing where the rules forbid it.

## 3. Data and licences

- [ ] Every dataset is listed with its licence and citation in `streamlit_app.py` §3, the app's
  *Data & method* tab and `README.md`. Copy the citations into your submission.
- [ ] **Mars Climate Database:** free for research, citation required. Keep the citation.
- [ ] **NASA POWER:** CC BY 4.0, attribution required (done in sources).
- [ ] **MODIS via Microsoft Planetary Computer:** NASA open data. Cite LP DAAC DOIs plus "accessed via
  Microsoft Planetary Computer".
- [ ] **Software licences:** CesiumJS (Apache-2.0), Plotly.js (MIT), Streamlit (Apache-2.0), PyTorch (BSD). Keep their licence files in the distribution
  (`static/globe/cesium/LICENSE.md`).
- [ ] Choose and state the **project licence** (the repository has a `LICENSE` file; check it is the
  one you want, e.g. MIT or Apache-2.0).

## 4. Presentation (counts as much as the other four criteria)

- [ ] 30-second demo path, rehearsed:
  1. pick Shackleton (Moon)
  2. "Find Earth analogs"
  3. the analogs fly in
  4. click #1, show "why"
  5. open *Watch the AI think*
  6. replay training
  7. open *Validation*, show the headline numbers
- [ ] Record the demo video at 1080p. Keep the globe moving and the narration short.
- [ ] One slide each: problem → idea → how it learns → results (validation numbers) → limitations →
  next steps.
- [ ] Put **real numbers** from `RESULTS.md` on the validation slide. Never round up or cherry-pick.

## 5. Before any public claim or paper

- [ ] Re-run the literature search (`LITERATURE.md`). Do not call it "the first" anything without checking.
- [ ] Double-check the coordinates and references of every known analog site in `streamlit_app.py` §2.
- [ ] Re-run `python streamlit_app.py build evaluate` and `python streamlit_app.py build results` after any change, and quote
  only those numbers.
