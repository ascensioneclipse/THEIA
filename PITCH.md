# THEIA pitch outline (7 slides + live demo, about 5 minutes)

Open and close on the THEIA logo (`static/brand/theia_logo.png`).

Numbers below are from `RESULTS.md`. Re-check them after any re-run. Never round them up.

1. **The problem**: one photo of Atacama next to a Mars rover photo.
   "Analog sites let us test before we fly, but they were chosen site by site by experts, and many are
   poorly characterised. What if we could ask the whole planet?"

2. **The idea**: three globes, with an arrow to one shared "fingerprint space".
   "One neural network learns what terrain looks like on Earth, the Moon and Mars at the same 3.7 km
   scale. No labels."

3. **How it learns**: the "spot the twin" picture (two crops of one landscape, rotated and noisy).
   "If it can recognise the same landscape after we scramble it, it has learned the landscape." Show the
   training curve: twin found rises from under 1 % to about 40 %.

4. **Live demo (90 s)**:
   1. Shackleton (Moon), then *Find Earth analogs*: the scan and the fly-in.
   2. Click #1 and show "why this matches" with the saliency outline.
   3. Switch the preset to *Rover mobility*: the ranking changes, so "a great analog for X, not Y".
   4. Jezero (Mars), same flow.
   5. *Watch the AI think*: replay training, the fingerprint map with known analogs as stars.

5. **Does it work? (Validity)**: the validation bar chart.
   "Without ever seeing the list, AnalogNet puts 65 % of the 28 known Mars analogs and 83 % of the 13
   known Moon analogs in Earth's top 10 %. Random places get 19 % (p ≤ 0.0005). A non-neural baseline
   is no better than chance."
   Then the twist: "For the Moon, the neural landform fingerprint does the work. For Mars, bare ground
   does. So analog selection needs several axes, not one similarity."

6. **What's new + new candidates (Creativity, Impact)**:
   - a shared 3-body embedding
   - multi-axis matching for different mission purposes
   - tested uncertainty (ECE 0.07, slightly over-confident, and we say so)
   - new candidate regions for field checks (e.g. Altyn-Tagh / Kunlun, Ustyurt Plateau, Baffin Island)

7. **Limits and next steps (Relevance, honesty)**:
   - regional scale only, no mineralogy yet
   - next: CRISM/M3/ASTER mineral layer, high-resolution zoom at shortlisted sites, field validation
     with an analog-mission team
   - all open data, open source, runs offline on a laptop

Closing line: *"Before we build a base on the Moon, THEIA tells you where on Earth to rehearse it,
and how sure it is."*

**Video tips:** 1080p screen recording of the app; keep the globe moving; caption key numbers; under 3 minutes if the rules ask for that (check the 2026 guide).
