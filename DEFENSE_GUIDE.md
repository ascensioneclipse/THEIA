# Defence guide: explain every part of THEIA in your own words

*THEIA is the app; AnalogNet is the neural matching model inside it.*

Read this aloud a few times, then close it and explain each section to a friend. If you can't explain
a part without the page, open the file listed next to it and read the comments.

---

## The 30-second pitch

> "Space agencies test rovers and astronauts in places like the Atacama Desert because they look like
> Mars. But those places were picked by experts one at a time. We built THEIA: a neural network that
> learned one shared 'terrain fingerprint' for Earth, the Moon and Mars, using only open NASA data and
> no human labels. You pick a future base site, for example the lunar south pole, and it searches all of
> Earth's land and shows the closest analogs on a 3-D globe. Each match has an error bar and an
> explanation. We checked it honestly: without ever being shown the list, it ranks the analog sites that
> experts already use far above random places: 65 % of known Mars analogs and 83 % of known Moon
> analogs land in Earth's top 10 %, against 19 % for random places."

---

## 1. Why "same ruler everywhere"? (`streamlit_app.py` §4, `streamlit_app.py` §1)

- The data come as latitude/longitude grids. One degree on the Moon is 30 km, on Earth 111 km. If we
  compared grids by pixel, a lunar patch would cover 3.7 times less ground, and we would be comparing
  different things.
- So we lay a flat sheet of "graph paper" with **3.7 km squares** on the sphere at each patch centre
  (a tangent plane) and read the elevation under every square. 32 x 32 squares = **118 km**.
- **Why 3.7 km?** It is the native resolution of the Mars map (MOLA, 16 pixels per degree). The Moon
  and Earth maps are finer, so we never invent detail. **Why 118 km?** It is regional context: big
  enough for craters, valleys and dune fields, small enough to be "a place".
- Patch centres come from a **Fibonacci lattice**: points spread evenly over a sphere (like sunflower
  seeds). A normal lat/lon grid crowds points at the poles, and our most important target (the lunar
  south pole) is at a pole.

## 2. What the network is (`streamlit_app.py` §9)

- A **convolutional neural network (CNN)**. A 3x3 filter slides over the height map and responds to
  small patterns (edges, bowls). Stacked layers combine them into bigger patterns (a ring of edges is a
  crater rim).
- 3 blocks, then "how much of each pattern is present" (average and max pooling), then 32 output
  numbers. The output is scaled to length 1, so comparing two fingerprints is a **cosine similarity**.
- **Input = 2 channels:** height relative to the patch mean (in km, keeps "how tall"), and the same
  divided by its own spread (keeps "what shape", even on very flat plains). Absolute height above sea
  level is removed, because Earth's sea level, the Moon's reference sphere and Mars's areoid are not
  comparable.
- About 0.18 million weights: tiny by modern standards. It trains on a laptop CPU, and a small model is
  less likely to memorise instrument quirks.

## 3. How it learns without labels (`streamlit_app.py` §10)

- Nobody has labelled "this Earth valley = that Mars valley". So we use **self-supervised contrastive
  learning (SimCLR)**:
  1. Take a 178 km tile. Cut **two random 118 km windows** from it (they may only partly overlap),
     then rotate, mirror, make it a bit taller or flatter, blur, add noise.
  2. Do this for 256 tiles: 512 views.
  3. The network must find, for each view, its twin among the other 511.
- To win, it has to describe the **landscape** (craters, ridges, dunes) and ignore what we randomised
  (orientation, exact position, sensor noise). Blur and noise imitate the differences between the three
  instruments.
- **Every batch has equal numbers of Earth, Moon and Mars tiles.** Otherwise Earth and Mars (~58k
  patches each) would drown out the Moon (15k).
- The loss is the **NT-Xent** loss: cross-entropy of "picking the right twin". The "twin found"
  accuracy rises from under 1% (chance) to about 39%.

## 4. The four axes and the score (`streamlit_app.py` §11)

| Axis | Measures | Why |
|---|---|---|
| Terrain (neural) | 1 - cosine of fingerprints | landscape shape, learned |
| Mobility | slope + roughness statistics | what a rover's wheels meet |
| Thermal | mean, day-night swing, max, min surface temperature | thermal design, ice stability |
| Bare ground | share of the Earth patch that is not vegetated or under ice | Moon/Mars have no plants |

- Each distance is divided by its **typical value** (the median distance between random Earth and
  Moon/Mars patches), so the axes have the same units. A mission preset weights them.
- **Score = 100 x exp(-D).** 100 = identical. About 37 = as different as a random pair (D = 1 by
  construction). It is an absolute similarity. "Top x %" is the rank among Earth land.
- **Why an ensemble?** We trained 5 networks with different random starts and different 80 % data
  subsets. Where they agree, the answer is stable. Where they disagree, the model is guessing. Spread =
  error bar. **Votes** = how many networks rank a site in their own top 100.
- **Is the uncertainty honest?** Leave-one-network-out test: if 3 of the other 4 networks vote for a
  site, the 5th should agree about 75 % of the time. The gap between expected and observed is the
  **expected calibration error (ECE)**.

## 5. Explanations

- **Feature bars:** each hand-made feature (relief, slope, roughness, temperatures...), Earth minus
  target, in standard deviations. Green = alike, red = different.
- **Occlusion saliency:** hide one 30 km square of the patch (flatten it), measure how much the
  similarity drops, and repeat for every position. Big drop = that landform mattered. This is model-
  agnostic and easy to explain. We chose it over gradient saliency because gradients on tiny CNNs are
  noisy.

## 6. Validation, the most important slide (`python streamlit_app.py build evaluate`, `RESULTS.md`)

- **Answer key:** 32 Earth sites already used as Moon/Mars analogs (`streamlit_app.py` §2). **Never used
  in training or design.**
- For every Earth patch we compute its "Mars-likeness" (mean distance to its 20 most similar Mars
  patches) and turn it into a percentile.
- **Site score** = best percentile within 120 km of the known site.
- **Compared against:**
  1. *Random land sites* (2,000 random sets of the same size), which gives the p-value.
  2. A *non-neural baseline*: cosine similarity on the raw hand-made features.
  3. *Ablations*: remove one axis at a time, or use one network instead of five.
- **Why this is fair:** the random sites get exactly the same "best within 120 km" treatment.
- **Honest caveats:**
  - Experts chose those sites for many reasons (access, geology, history), so a perfect score is
    neither expected nor wanted.
  - Most known analogs are deserts, and the bare-ground axis rewards deserts. That is why we report the
    ablation without it.

## 7. The app (`streamlit_app.py`, `streamlit_app.py` §13, `static/globe/`)

- **Streamlit** turns a Python script into a website: every widget (site picker, mission sliders)
  re-runs the script, and results are cached so a repeat search is instant.
- The 3-D globe is **CesiumJS** (open source, used in NASA tools), embedded as a **Streamlit custom
  component**: `static/globe/index.html` talks to Python by messages, so clicking a pin selects a result
  and clicking the Moon or Mars picks a new site.
- **Why it is light on hardware:** every globe image is cut into small tiles (a *tile pyramid*); the
  browser only downloads the tiles on screen at the current zoom. Cesium runs in *render-on-demand*
  mode, so it only draws a new frame when something changes.
- Globe styles:
  - **Hologram** (default): dot-matrix continents and glowing coastlines, drawn from the elevation data.
  - **Satellite:** our true-colour globe built from MODIS MCD43A4 via Microsoft Planetary Computer,
    sharpened by a live Sentinel-2 mosaic when zoomed in.
  - Moon/Mars: LOLA/MOLA shaded relief, with NASA Trek mosaics when online.
- Charts are **Plotly**. Reports are built with **matplotlib** (Markdown + PDF download).

---

## Likely judge questions and good answers

**"Isn't this just nearest-neighbour search?"**
The search is nearest-neighbour, yes. The contribution is the space it searches in: a fingerprint
learned jointly on three worlds at the same physical scale. We show what it adds by comparing against
plain nearest-neighbour on raw features (the baseline).

**"How do you know it works?"**
Rediscovery: 65 % of known Mars analogs and 83 % of known Moon analogs fall in Earth's top 10 %, vs
19 % for random land (p ≤ 0.0005). The raw-feature baseline is near chance (AUC 0.52 Mars, 0.44
Moon). Ablations show which axis does the work: for the Moon, the neural landform fingerprint; for
Mars, bare ground.

**"So the neural network doesn't help for Mars?"**
Correct, at this 118 km scale, and we report it. Mars analogs were chosen mainly for their environment
(dry, bare, cold), which the bare-ground and thermal axes capture. Lunar analogs were chosen for their
landforms (impact craters, volcanic fields), and there the fingerprint is the best method by far. That
is exactly why AnalogNet is multi-axis with mission presets.

**"Your ECE is 0.07. Is the uncertainty useful?"**
Yes: the more networks vote for a site, the more likely an unseen network agrees (20 % → 44 % → 66 %
→ 91 %). It is slightly over-confident, so read a 5/5 vote as "about 90 % stable", not "certain".

**"Why not use satellite images instead of elevation?"**
Images depend on lighting, atmosphere, dust and vegetation colour, which differ completely between
worlds. Elevation is measured the same way (laser altimetry: LOLA and MOLA) and describes the shape
that rovers and landers actually face.

**"Your resolution is 3.7 km. Rovers care about 10 cm rocks."**
Correct, and we say so. AnalogNet is a regional screening tool: it tells you where to look. Fine-scale
checks need high-resolution data (HiRISE, LROC NAC, drone surveys) at the shortlisted sites. That is
our next step.

**"Earth has water, plants and erosion. How can Earth match the Moon?"**
It mostly can't, and our scores show it: lunar matches have lower absolute similarity than Mars
matches. We report that honestly. The body-identity probe measures how much the fingerprint still
knows about which world a patch came from.

**"Why five networks?"**
To measure uncertainty cheaply and honestly (deep ensembles are a standard, well-calibrated method).
We tested the calibration instead of just claiming it.

**"What is new compared to Meng et al. 2025?"**
They learned an Earth-Mars image similarity for one region (Qaidam Basin). We use elevation at a common
physical scale across three bodies, search the whole Earth, and add multi-axis matching, calibrated
uncertainty and a rediscovery benchmark. (See `LITERATURE.md`.)

**"Did you use AI to build this?"**
Yes. An AI coding assistant helped write the code; we disclose this as the rules require. We
understand, ran and checked every part, and the validation numbers come from our own runs.

**"What would you do with more time?"**
- Add a mineralogy layer (Mars TES/CRISM, Moon M3, Earth ASTER) for geology matching.
- Add finer-scale terrain at the shortlisted sites.
- Train with a domain-adversarial term to make the fingerprint even less body-specific.
- Run field validation with an analog-mission team.
