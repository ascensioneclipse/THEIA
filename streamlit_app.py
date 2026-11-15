"""
THEIA - Terrestrial Habitat Evaluation and Interplanetary Analogue
NASA Space Apps Challenge 2026: Identify Earth Locations that Analog the Permanent Moon Base
Locations and Mars.  The neural matching model inside THEIA is called AnalogNet.

Everything is in this one file.

    streamlit run streamlit_app.py            the website
    python streamlit_app.py build <step>      rebuild the data / model (see the BUILD PIPELINE section)

Folders next to this file hold data, not code:
    assets/      trained 5-network ensemble + processed open data
    static/      globe tiles, fonts, logo, and the 3-D globe page with CesiumJS (static/globe)
    .streamlit/  Streamlit settings

Sections:
     1. CONFIG - every important number, with the reason for it
     2. SITES - Moon/Mars targets and the known Earth analogs (the answer key)
     3. DATA SOURCES - exact products, licences, citations
     4. SAMPLING - Fibonacci lattice and tangent-plane patches
     5. TERRAIN FEATURES - hand-made descriptors (baseline, mobility, explanations)
     6. GLOBAL GRIDS - raw files to elevation and temperature grids
     7. MODIS - Earth temperature and vegetation from Microsoft Planetary Computer
     8. EARTH TEXTURE - true-colour globe from MODIS MCD43A4
     9. NEURAL NETWORK - the AnalogNet terrain encoder
    10. TRAINING - self-supervised contrastive learning
    11. MATCHING ENGINE - scores, uncertainty, votes, explanations
    12. REPORT - Markdown + PDF export
    13. 3-D GLOBE - CesiumJS as a Streamlit component
    14. THE WEBSITE
    15. BUILD PIPELINE
    16. ENTRY POINT
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Callable, Iterator
import argparse
import datetime as dt
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import zipfile

# Streamlit also runs this file as "__main__", but it has already imported itself by then.
_COMMAND_LINE = __name__ == "__main__" and "streamlit" not in sys.modules

# "python streamlit_app.py" with no arguments starts the website (same as "streamlit run streamlit_app.py").
if _COMMAND_LINE and len(sys.argv) == 1:
    import subprocess
    print("Starting THEIA ... your browser will open at http://localhost:8501  (Ctrl+C to stop)", flush=True)
    try:
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)]))
    except KeyboardInterrupt:
        sys.exit(0)

from PIL import Image, ImageFilter
from scipy import ndimage as ndi
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
if _COMMAND_LINE:   # build commands: hide Streamlit's harmless "bare mode" warnings
    import logging
    logging.disable(logging.WARNING)
import torch
import torch.nn as nn
import torch.nn.functional as TF

__version__ = "0.1.0"


def _http():
    """The requests library, imported only when a build step downloads data."""
    import requests
    return requests


# ==================================================================================================
# 1. CONFIG - every important number, with the reason for it
# ==================================================================================================
# Central configuration for AnalogNet.
#
# Everything that is a "magic number" in the project lives here, with the reason
# for the choice written next to it, so that every number can be defended.
# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent                      # the app folder

RAW_DIR = ROOT / "data" / "raw"          # downloaded originals (NOT shipped, large)
ASSET_DIR = Path(os.environ.get("ANALOGNET_ASSET_DIR", ROOT / "assets"))   # processed data + trained model (shipped)
UI_DIR = ROOT / "static" / "globe"       # the globe page + CesiumJS (served by Streamlit as a component)
# Globe textures, fonts and logo, served by Streamlit as /app/static/...
STATIC_DIR = ROOT / "static"

# Reports are written to a user-writable folder (the install dir may be read-only).
REPORT_DIR = Path(os.environ.get("ANALOGNET_REPORT_DIR", Path.home() / "AnalogNet_reports"))

# --------------------------------------------------------------------------
# The three bodies.  Mean radii in metres (IAU values used by the map products).
# --------------------------------------------------------------------------
BODIES = ("earth", "moon", "mars")
RADIUS_M = {"earth": 6_371_000.0, "moon": 1_737_400.0, "mars": 3_396_190.0}
BODY_ID = {b: i for i, b in enumerate(BODIES)}

# --------------------------------------------------------------------------
# Patch geometry -- THE key design decision.
#
# A "patch" is a square piece of terrain sampled on a local flat (tangent-plane)
# grid with the SAME PHYSICAL pixel size on every body.  If we compared pixels
# in degrees instead, a lunar pixel would cover 3.7x less ground than an Earth
# pixel and the network would be comparing different things.
#
# 3.7 km/pixel is the native resolution of the MOLA 16 pixel/degree Mars grid
# (2*pi*3396.19 km / 5760 = 3.705 km).  The Moon (LOLA) and Earth (ETOPO 2022)
# grids are finer than this, so nothing is up-sampled / invented.
# 32 x 32 pixels -> a footprint of ~118 km: "regional context" scale.
# --------------------------------------------------------------------------
PIXEL_M = 3700.0
PATCH = 32
FOOTPRINT_KM = PIXEL_M * PATCH / 1000.0

# Spacing between patch centres (km).  ~Half a footprint so neighbouring
# patches overlap by about 50 % and no place falls between two patches.
CENTRE_SPACING_KM = 50.0

# Earth patches need at least this fraction of land to be used as candidates.
MIN_LAND_FRACTION = 0.5

# --------------------------------------------------------------------------
# Neural network / training
# --------------------------------------------------------------------------
EMBED_DIM = 32          # size of the "terrain fingerprint" vector
CONTEXT = 48            # training tiles are 48x48 px (178 km): the 32x32 patch plus a margin
CROP = 28               # the two "views" are random 28x28 windows inside the 48x48 tile, so
                        # they can be up to 74 km apart: "same landscape", not "same pixels"
ENSEMBLE = 5            # number of independently trained encoders (for uncertainty)
TRAIN_STEPS = 2000      # optimisation steps per ensemble member (~6 min each on 4 CPU cores)
BATCH = 256
TEMPERATURE = 0.2       # softness of the contrastive loss
LR = 2e-3
SEED = 2026

# Fixed scale (metres) used to turn elevation into network input.  It is a
# constant - NOT computed per patch - so that "how tall" survives in channel 0.
ELEV_SCALE_M = 1000.0

# --------------------------------------------------------------------------
# Names of the hand-crafted terrain descriptors (see section 5)
# and of the thermal descriptors.  These are used by the baseline, by the
# "mobility" axis and by the explanation panel.
# --------------------------------------------------------------------------
MORPH_FEATURES = [
    "relief_m",          # 95th - 5th percentile of elevation in the patch
    "elev_std_m",        # standard deviation of elevation
    "slope_mean_deg",    # mean slope at 3.7 km baseline
    "slope_p90_deg",     # 90th percentile slope (the steep parts)
    "rough_small_m",     # RMS height left after removing an 11 km (3x3 px) mean
    "rough_mid_m",       # RMS height between 11 km and 33 km scales
    "curvature",         # mean |Laplacian|: how sharply the surface bends
    "hypsometric",       # (mean-min)/(max-min): plateau-like (high) vs basin-like (low)
    "elev_skew",         # skewness of elevation: peaks (+) or pits (-)
    "pit_density",       # fraction of pixels that are local minima (crater proxy)
    "pit_depth_m",       # mean depth of closed depressions (crater / sinkhole proxy)
    "anisotropy",        # 0 = no preferred direction, 1 = strongly aligned ridges/valleys
]
# Features that are heavily right-skewed are compared in log space.
MORPH_LOG = {"relief_m", "elev_std_m", "slope_mean_deg", "slope_p90_deg",
             "rough_small_m", "rough_mid_m", "curvature", "pit_depth_m"}
# Subset that matters for driving a rover.
MOBILITY_FEATURES = ["slope_mean_deg", "slope_p90_deg", "rough_small_m", "rough_mid_m", "relief_m"]

THERMAL_FEATURES = [
    "t_mean_K",          # long-term mean surface temperature
    "t_diurnal_K",       # typical day-night swing
    "t_max_K",           # typical warmest surface temperature
    "t_min_K",           # typical coldest surface temperature
]

# --------------------------------------------------------------------------
# Matching axes and mission-purpose presets (weights sum to 1).
#   terrain  = learned neural embedding (shape/texture of the landscape)
#   mobility = slope + roughness statistics (what a rover "feels")
#   thermal  = surface-temperature environment
#   exposure = is the ground bare?  Moon and Mars have no vegetation, so an
#              Earth analog should be bare rock/regolith/sand (MODIS NDVI) and
#              not under an ice sheet.  Earth-side only: Moon/Mars are 100 % bare.
# --------------------------------------------------------------------------
AXES = ("terrain", "mobility", "thermal", "exposure")
PRESETS = {
    "balanced": {
        "label": "Balanced (all axes)",
        "weights": {"terrain": 0.35, "mobility": 0.25, "thermal": 0.25, "exposure": 0.15},
        "why": "General-purpose analog: landscape shape, trafficability, thermal environment and bare ground.",
    },
    "rover": {
        "label": "Rover mobility testing",
        "weights": {"terrain": 0.3, "mobility": 0.5, "thermal": 0.05, "exposure": 0.15},
        "why": "Slopes and roughness dominate: what matters is whether wheels meet the same ground.",
    },
    "ice": {
        "label": "Ice / resource extraction",
        "weights": {"terrain": 0.2, "mobility": 0.1, "thermal": 0.6, "exposure": 0.1},
        "why": "Cold-environment operations: temperature extremes dominate, terrain is secondary.",
    },
    "landform": {
        "label": "Landform science (shape only)",
        "weights": {"terrain": 1.0, "mobility": 0.0, "thermal": 0.0, "exposure": 0.0},
        "why": "Pure neural terrain fingerprint: geomorphology field studies.",
    },
}
# The lava-tube / cave-habitat purpose needs a volcanic-geology layer on all
# three bodies, which version 1 does not have.  We list it honestly as planned.
PLANNED_PRESETS = {
    "cave": "Lava-tube / cave habitat - needs a lithology (volcanic unit) layer; planned for v2.",
}

PROC_DIR = ROOT / "data" / "processed"   # training tiles (built by "build dataset", not shipped)


# ==================================================================================================
# 2. SITES - Moon/Mars targets and the known Earth analogs (the answer key)
# ==================================================================================================
# Reference places.
#
# TARGETS       - Moon and Mars sites a user can pick (landing sites, candidate bases).
# KNOWN_ANALOGS - Earth field sites that the planetary-science community already
#                 uses as Moon/Mars analogs.  They are our "answer key": the model
#                 never sees this list during training; it is used ONLY to check
#                 whether the model rediscovers them (validation dashboard).
#
# Coordinates are approximate site centres (degrees, east-positive longitude),
# good to roughly 0.1 deg, which is far finer than our ~118 km footprint.
# IMPORTANT (for the paper): each row's "ref" is a pointer to where the site is
# described; verify every reference yourself before citing.
# id, name, body, lat, lon, note
TARGETS = [
    # ----- Moon: south-polar candidate regions for Artemis / a permanent base -----
    ("shackleton", "Shackleton rim / Connecting Ridge", "moon", -89.5, -137.0,
     "Artemis III candidate region; near-permanent sunlight next to permanently shadowed craters."),
    ("malapert", "Malapert Massif", "moon", -86.0, 0.0,
     "Artemis III candidate region; high massif with Earth line-of-sight."),
    ("mouton", "Mons Mouton", "moon", -84.6, 31.0,
     "Artemis candidate region and planned VIPER rover site; flat-topped mountain."),
    ("nobile", "Nobile rim", "moon", -85.2, 53.5, "Artemis III candidate region near Nobile crater."),
    ("amundsen", "Amundsen rim", "moon", -84.5, 82.8, "Artemis III candidate region; crater floor access."),
    ("schrodinger", "Schrodinger basin", "moon", -75.0, 132.4,
     "Far-side peak-ring basin with volcanic vent; CLPS science target."),
    ("chandrayaan3", "Chandrayaan-3 site (Shiv Shakti)", "moon", -69.37, 32.32, "ISRO landing, 2023."),
    # ----- Moon: other reference sites -----
    ("apollo11", "Apollo 11 - Mare Tranquillitatis", "moon", 0.67, 23.47, "Flat basaltic mare."),
    ("apollo15", "Apollo 15 - Hadley Rille", "moon", 26.13, 3.63, "Mare edge, sinuous rille, Apennine front."),
    ("apollo16", "Apollo 16 - Descartes highlands", "moon", -8.97, 15.50, "Cratered highlands."),
    ("apollo17", "Apollo 17 - Taurus-Littrow", "moon", 20.19, 30.77, "Valley between massifs."),
    ("change4", "Chang'e-4 - Von Karman crater", "moon", -45.46, 177.59, "CNSA far-side landing, SPA basin."),
    ("change5", "Chang'e-5 - Mons Rumker region", "moon", 43.06, -51.92, "CNSA sample return, young mare."),
    ("marius", "Marius Hills pit (lava-tube skylight)", "moon", 14.09, -56.77,
     "Volcanic dome field with a known pit; candidate sheltered habitat."),
    ("tranq_pit", "Mare Tranquillitatis pit", "moon", 8.34, 33.22, "Deep pit with possible sublunarean void."),
    ("aristarchus", "Aristarchus plateau", "moon", 23.7, -47.4, "Pyroclastic deposits; resource interest."),
    ("tycho", "Tycho crater", "moon", -43.31, -11.36, "Young complex crater with central peak."),
    # ----- Mars: landed missions -----
    ("jezero", "Jezero crater (Perseverance)", "mars", 18.44, 77.45, "Ancient delta in a crater lake."),
    ("gale", "Gale crater (Curiosity)", "mars", -4.59, 137.44, "Crater with 5 km central mound."),
    ("gusev", "Gusev crater (Spirit)", "mars", -14.57, 175.47, "Volcanic plains inside a crater."),
    ("meridiani", "Meridiani Planum (Opportunity)", "mars", -1.95, -5.53, "Very flat hematite plains."),
    ("chryse", "Chryse Planitia (Viking 1)", "mars", 22.27, -47.95, "Outflow-channel plains."),
    ("utopia_vl2", "Utopia Planitia (Viking 2)", "mars", 47.64, 134.29, "Northern lowland plains."),
    ("ares", "Ares Vallis (Pathfinder)", "mars", 19.13, -33.22, "Flood-channel mouth."),
    ("phoenix", "Vastitas Borealis (Phoenix)", "mars", 68.22, -125.75, "Polar plains with shallow ground ice."),
    ("insight", "Elysium Planitia (InSight)", "mars", 4.50, 135.62, "Smooth volcanic plains."),
    ("zhurong", "Utopia Planitia (Zhurong)", "mars", 25.07, 109.93, "CNSA rover site."),
    ("oxia", "Oxia Planum (Rosalind Franklin)", "mars", 18.2, -24.3, "ESA rover site; clay-bearing plains."),
    # ----- Mars: candidate human-base / resource regions -----
    ("arcadia", "Arcadia Planitia", "mars", 40.0, -160.0, "Mid-latitude plains with shallow water ice."),
    ("deuteronilus", "Deuteronilus Mensae", "mars", 43.0, 23.0, "Debris-covered glaciers; ice resource."),
    ("korolev", "Korolev crater", "mars", 72.8, 164.5, "Crater filled with water ice."),
    ("melas", "Melas Chasma (Valles Marineris)", "mars", -10.4, -72.7, "Deep canyon floor; high air pressure."),
    ("arsia", "Arsia Mons (cave skylights)", "mars", -8.35, -120.09, "Shield volcano with pit-crater chains."),
    ("hellas", "Hellas basin floor", "mars", -42.0, 70.0, "Lowest region of Mars."),
    ("mawrth", "Mawrth Vallis", "mars", 22.6, -16.5, "Clay-rich ancient channel."),
    ("medusae", "Medusae Fossae", "mars", -3.0, -163.0, "Wind-sculpted soft deposits (yardangs)."),
]

# name, lat, lon, analog-of (bodies), what it is used for, pointer to a source
KNOWN_ANALOGS = [
    ("Atacama Desert (Yungay), Chile", -24.07, -69.90, ("mars",),
     "Hyper-arid soils, life-detection rover tests", "Navarro-Gonzalez et al. 2003, Science 302"),
    ("Haughton impact structure, Devon Island, Canada", 75.38, -89.67, ("mars", "moon"),
     "Impact crater in polar desert; Haughton-Mars Project", "Lee & Osinski 2005, Meteorit. Planet. Sci. 40"),
    ("Death Valley, USA", 36.24, -116.82, ("mars",),
     "Alluvial fans, playas, dunes; rover/instrument tests", "NASA/JPL field tests; Golombek & Rapp 1997"),
    ("McMurdo Dry Valleys, Antarctica", -77.50, 162.00, ("mars",),
     "Cold hyper-arid desert, permafrost", "Marchant & Head 2007, Icarus 192"),
    ("Hanksville / MDRS, Utah, USA", 38.41, -110.79, ("mars",),
     "Mars Desert Research Station; clay-rich badlands", "Clarke & Stoker 2011, Int. J. Astrobiol. 10"),
    ("Rio Tinto, Spain", 37.72, -6.55, ("mars",),
     "Acidic iron-sulfate geochemistry (Meridiani analog)", "Fernandez-Remolar et al. 2005, EPSL 240"),
    ("Dallol / Danakil Depression, Ethiopia", 14.24, 40.30, ("mars",),
     "Hydrothermal salts, extreme chemistry", "Cavalazzi et al. 2019, Astrobiology 19"),
    ("Askja / Holuhraun, Iceland", 65.03, -16.75, ("mars", "moon"),
     "Basaltic volcanism; Apollo astronaut training", "Apollo geology training 1965/67; Voigt et al. 2021"),
    ("Mauna Kea / Mauna Loa (HI-SEAS), Hawaii, USA", 19.60, -155.49, ("mars", "moon"),
     "Basaltic shield volcano; habitat and rover analog", "HI-SEAS; ten Kate et al. 2013 (PISCES)"),
    ("Svalbard (AMASE), Norway", 79.40, 13.40, ("mars",),
     "Arctic volcanics and carbonates; instrument tests", "Steele et al. 2007 (AMASE), LPSC"),
    ("Pilbara, Western Australia", -21.20, 119.30, ("mars",),
     "Ancient stromatolites; Mars 2020 team field training", "Allwood et al. 2006; NASA/ESA field trip 2019"),
    ("Qaidam Basin, China", 38.00, 92.50, ("mars",),
     "High, cold, hyper-arid basin with yardangs and evaporites", "Xiao et al. 2017, Earth-Sci. Rev. 164"),
    ("Namib Desert, Namibia", -24.70, 15.30, ("mars",),
     "Large linear dunes (Bagnold dunes analog)", "Fenton et al. / dune analog literature"),
    ("Dhofar, Oman (AMADEE-18)", 18.17, 55.27, ("mars",),
     "Analog astronaut mission site", "Groemer et al. 2020, Astrobiology 20"),
    ("Makhtesh Ramon, Negev, Israel", 30.60, 34.80, ("mars",),
     "Erosional crater; D-MARS / AMADEE-20", "D-MARS; Austrian Space Forum AMADEE-20"),
    ("Noerdlinger Ries crater, Germany", 48.88, 10.57, ("moon",),
     "Impact crater; Apollo 14/17 crew training", "Apollo training 1970; Stoeffler et al. 2013"),
    ("San Francisco Volcanic Field / Black Point, Arizona, USA", 35.67, -111.47, ("moon", "mars"),
     "Cinder cones, lava flows; Apollo training, Desert RATS", "Ross et al. 2013, Acta Astronaut. 90"),
    ("Craters of the Moon, Idaho, USA", 43.42, -113.52, ("moon", "mars"),
     "Young basalt flows, lava tubes; FINESSE / BASALT", "Heldmann et al. 2019; Lim et al. 2019, Astrobiology 19"),
    ("Tolbachik, Kamchatka, Russia", 55.83, 160.33, ("moon",),
     "Volcanic cinder fields; Lunokhod rover testing", "Soviet Lunokhod test programme"),
    ("Sudbury impact structure, Canada", 46.60, -81.20, ("moon",),
     "Large impact basin; Apollo 16/17 crew training", "Apollo training 1971-72; Lunar and Planetary Institute"),
    ("Mistastin crater, Labrador, Canada", 55.88, -63.30, ("moon",),
     "Impact crater in anorthosite (lunar highland rock)", "Mader & Osinski 2018; CSA analog missions"),
    ("Licancabur / Altiplano-Puna, Chile-Bolivia", -22.83, -67.88, ("mars",),
     "High-altitude lakes, intense UV, thin air", "Cabrol et al. 2009, J. Geophys. Res. 114"),
    ("Ladakh (Tso Kar), India", 33.30, 78.00, ("mars",),
     "High cold desert; ISRO analog site", "Pandey et al. 2020, Int. J. Astrobiol. 19"),
    ("Erfoud (Ibn Battuta Centre), Morocco", 31.40, -4.20, ("mars",),
     "Desert rover/lander test area", "Ibn Battuta Centre; Ori et al., ExoMars tests"),
    ("Chott el Jerid, Tunisia", 33.70, 8.40, ("mars",),
     "Salt flat (playa) analog for martian evaporites", "Barbieri & Stivaletta 2011"),
    ("Kilbourne Hole / Potrillo volcanic field, New Mexico, USA", 31.97, -106.96, ("moon", "mars"),
     "Maar crater; Apollo training, RIS4E", "Apollo training; RIS4E SSERVI team"),
    ("Lonar crater, India", 19.98, 76.51, ("moon", "mars"),
     "Impact crater in basalt", "Fredriksson et al. 1973, Science 180; Maloof et al. 2010"),
    ("Arkaroola, South Australia", -30.30, 139.30, ("mars",),
     "Mars Society Australia analog region", "Clarke et al. 2004"),
    ("Axel Heiberg Island, Canada", 79.40, -90.80, ("mars",),
     "Polar desert, cold saline springs, permafrost", "Pollard et al. 2009, Planet. Space Sci. 57"),
    ("Mount Etna, Italy", 37.75, 15.00, ("moon", "mars"),
     "Volcanic terrain; ROBEX / ARCHES robotic tests", "Wedler et al. 2017 (ROBEX)"),
    ("Lanzarote, Canary Islands, Spain", 29.00, -13.70, ("moon", "mars"),
     "Volcanic island, lava tubes; ESA PANGAEA training", "Sauro et al. 2019 (PANGAEA)"),
    ("Teide, Tenerife, Spain", 28.27, -16.64, ("mars",),
     "Volcanic caldera; rover and instrument tests", "ESA / Teide field campaigns"),
]


def targets_as_dicts() -> list[dict]:
    return [dict(id=t[0], name=t[1], body=t[2], lat=t[3], lon=t[4], note=t[5]) for t in TARGETS]


def analogs_as_dicts() -> list[dict]:
    return [dict(name=a[0], lat=a[1], lon=a[2], bodies=list(a[3]), use=a[4], ref=a[5]) for a in KNOWN_ANALOGS]


# ==================================================================================================
# 3. DATA SOURCES - exact products, licences, citations
# ==================================================================================================
# Exact data sources used by AnalogNet (checked on 2026-11-15).
#
# Every entry: what it is, where it comes from, how it is licensed and how to
# cite it.  `python streamlit_app.py build download` downloads exactly these files.
# Nothing else is used to train or evaluate the model.
PDS = "https://pds-geosciences.wustl.edu"
DIVINER_BANDS = ["90s80s", "80s70s", "70s60s", "60s50s", "50s40s", "40s30s", "30s20s", "20s10s",
                 "10s00s", "00n10n", "10n20n", "20n30n", "30n40n", "40n50n", "50n60n", "60n70n",
                 "70n80n", "80n90n"]
MCD_SEASONS = [0, 90, 180, 270]                 # solar longitude Ls (deg): the 4 martian seasons
MCD_LOCAL_TIMES = list(range(0, 24, 2))         # local true solar time (martian hours)
POWER_ZARR = ("https://nasa-power.s3.us-west-2.amazonaws.com/merra2/temporal/"
              "power_merra2_climatology_temporal_lst.zarr")

SOURCES = [
    {
        "key": "moon_dem", "body": "Moon", "variable": "Elevation",
        "product": "LRO LOLA Gridded Data Record LDEM_16 (16 pixel/degree, ~1.9 km)",
        "agency": "NASA (LRO / LOLA team, GSFC) via PDS Geosciences Node",
        "files": {"moon/ldem_16.img":
                  f"{PDS}/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/ldem_16.img"},
        "license": "Public domain (NASA PDS; no restrictions, citation requested)",
        "cite": "Smith, D.E. et al. (2010) The Lunar Orbiter Laser Altimeter investigation on the "
                "Lunar Reconnaissance Orbiter mission. Space Sci. Rev. 150, 209-241. "
                "Data set LRO-L-LOLA-4-GDR-V1.0.",
    },
    {
        "key": "mars_dem", "body": "Mars", "variable": "Elevation",
        "product": "MGS MOLA Mission Experiment Gridded Data Record MEGDR megt90n000eb "
                   "(16 pixel/degree, ~3.7 km, topography above the areoid)",
        "agency": "NASA (MGS / MOLA team) via PDS Geosciences Node",
        "files": {"mars/megt90n000eb.img":
                  f"{PDS}/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/megt90n000eb.img"},
        "license": "Public domain (NASA PDS; citation requested)",
        "cite": "Smith, D.E. et al. (2001) Mars Orbiter Laser Altimeter: experiment summary after "
                "the first year of global mapping of Mars. J. Geophys. Res. 106, 23689-23722. "
                "Data set MGS-M-MOLA-5-MEGDR-L3-V1.0.",
    },
    {
        "key": "earth_dem", "body": "Earth", "variable": "Elevation",
        "product": "ETOPO 2022 Global Relief Model, 60 arc-second, ice-surface version (~1.9 km)",
        "agency": "NOAA National Centers for Environmental Information (NCEI)",
        "files": {"earth/ETOPO_2022_v1_60s_N90W180_surface.tif":
                  "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/"
                  "60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif"},
        "license": "Public domain (U.S. Government work; citation requested)",
        "cite": "NOAA National Centers for Environmental Information (2022) ETOPO 2022 15 Arc-Second "
                "Global Relief Model. doi:10.25921/fd45-gt74.",
    },
    {
        "key": "moon_temp", "body": "Moon", "variable": "Surface temperature",
        "product": "LRO Diviner Global Cumulative Products (GCP): bolometric brightness temperature "
                   "binned at 0.5 deg and 0.25 h of local time, 2009-07 to 2015-04",
        "agency": "NASA (LRO / Diviner team, UCLA) via PDS Geosciences Node",
        "files": {f"diviner/global_cumul_avg_cyl_{b}_002.tab":
                  f"{PDS}/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gcp/"
                  f"global_cumul_avg_cyl_{b}_002.tab" for b in DIVINER_BANDS},
        "license": "Public domain (NASA PDS; citation requested)",
        "cite": "Williams, J.-P. et al. (2017) The global surface temperatures of the Moon as measured "
                "by the Diviner Lunar Radiometer Experiment. Icarus 283, 300-325. "
                "Data set LRO-L-DLRE-5-GCP-V1.0.",
    },
    {
        "key": "mars_temp", "body": "Mars", "variable": "Surface temperature",
        "product": "Mars Climate Database v6.2, surface temperature, climatology scenario, "
                   "4 seasons x 12 local times, native 5.625 x 3.83 deg grid. "
                   "NOTE: this is a validated climate MODEL, not a direct observation.",
        "agency": "LMD / OU / IAA, funded by ESA and CNES (web interface)",
        "files": {},   # fetched through the MCD web interface, see build_download()
        "license": "Free for research use; citation required (see MCD 'How to cite')",
        "cite": "Millour, E. et al. (2022) The Mars Climate Database (version 6.1). Europlanet Science "
                "Congress 2022, EPSC2022-786; Forget, F. et al. (1999) J. Geophys. Res. 104, 24155-24175.",
    },
    {
        "key": "earth_temp", "body": "Earth", "variable": "Surface (skin) temperature",
        "product": "NASA POWER climatology (MERRA-2 reanalysis, 0.5 x 0.625 deg): TS, TS_MAX_AVG, "
                   "TS_MIN_AVG monthly climatologies. NOTE: reanalysis = model constrained by observations.",
        "agency": "NASA Langley Research Center POWER project / NASA GMAO (AWS Open Data)",
        "files": {},   # read from the public Zarr store POWER_ZARR
        "license": "CC BY 4.0",
        "cite": "NASA POWER Project, NASA Langley Research Center, https://power.larc.nasa.gov ; "
                "Gelaro, R. et al. (2017) MERRA-2. J. Climate 30, 5419-5454.",
    },
]

SOURCES.append({
    "key": "earth_water_ice", "body": "Earth", "variable": "Lake / glacier masks and place names",
    "product": "Natural Earth 1:10m physical vectors: lakes, glaciated areas, Antarctic ice shelves",
    "agency": "Natural Earth (naturalearthdata.com)",
    "files": {f"naturalearth/{n}.zip": f"https://naturalearth.s3.amazonaws.com/10m_physical/{n}.zip"
              for n in ("ne_10m_lakes", "ne_10m_glaciated_areas", "ne_10m_antarctic_ice_shelves_polys")}
    | {"naturalearth/ne_10m_admin_0_countries.zip": "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip",
       "naturalearth/ne_10m_geography_regions_polys.zip": "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_geography_regions_polys.zip"},
    "license": "Public domain",
    "cite": "Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com.",
})

SOURCES.append({
    "key": "earth_lst_modis", "body": "Earth", "variable": "Land surface temperature (observed)",
    "product": "MODIS Aqua MYD11A2 v6.1 8-day LST, day (~13:30) and night (~01:30), 12 monthly "
               "composites of 2023, read at ~7 km from Cloud-Optimised GeoTIFFs",
    "agency": "NASA LP DAAC (producer); hosted by Microsoft Planetary Computer (collection modis-11A2-061)",
    "files": {},   # see build_modis()
    "license": "NASA EOSDIS open data, no restrictions (citation requested)",
    "cite": "Wan, Z., Hook, S., Hulley, G. (2021) MODIS/Aqua Land Surface Temperature/Emissivity 8-Day L3 "
            "Global 1km SIN Grid V061. NASA LP DAAC. doi:10.5067/MODIS/MYD11A2.061. "
            "Accessed via Microsoft Planetary Computer.",
})
SOURCES.append({
    "key": "earth_ndvi_modis", "body": "Earth", "variable": "Vegetation (bare-ground fraction)",
    "product": "MODIS Aqua MYD13A1 v6.1 16-day NDVI, 4 seasonal composites of 2023 (max taken), "
               "read at ~3.7 km; bare = max NDVI < 0.2",
    "agency": "NASA LP DAAC (producer); hosted by Microsoft Planetary Computer (collection modis-13A1-061)",
    "files": {},
    "license": "NASA EOSDIS open data, no restrictions (citation requested)",
    "cite": "Didan, K. (2021) MODIS/Aqua Vegetation Indices 16-Day L3 Global 500m SIN Grid V061. "
            "NASA LP DAAC. doi:10.5067/MODIS/MYD13A1.061. Accessed via Microsoft Planetary Computer.",
})
for _s in SOURCES:
    if _s["key"] == "earth_temp":
        _s["variable"] = "Surface temperature (gap filler only)"

IMAGERY = [
    {"body": "Earth", "what": "AnalogNet true-colour globe built from MODIS MCD43A4 v6.1 BRDF-adjusted reflectance "
                              "(NASA LP DAAC, doi:10.5067/MODIS/MCD43A4.061) via Microsoft Planetary Computer, summer "
                              "composites of 2023, with NOAA ETOPO 2022 relief shading and ocean depth (offline, 16384 x 8192 px)",
     "license": "NASA open data / public domain"},
    {"body": "Earth", "what": "Sentinel-2 L2A cloud-free mosaic (ESA Copernicus) rendered live by the Microsoft Planetary "
                              "Computer data API when zoomed in (online, not stored)",
     "license": "Copernicus Sentinel data terms (free, attribution)"},
    {"body": "Moon", "what": "Shaded relief rendered from LRO LOLA 16 px/deg by AnalogNet (offline); LRO WAC mosaic via "
                             "NASA Solar System Treks (online)", "license": "Public domain (NASA/GSFC/ASU)"},
    {"body": "Mars", "what": "Shaded relief rendered from MGS MOLA 16 px/deg by AnalogNet (offline); Viking MDIM 2.1 colour "
                             "mosaic via NASA Solar System Treks (online)", "license": "Public domain (NASA/JPL/USGS)"},
]


# ==================================================================================================
# 4. SAMPLING - Fibonacci lattice and tangent-plane patches
# ==================================================================================================
# Geometry: where patches are, and how a patch is cut out of a global map.
#
# Two ideas, both simple:
#
# 1. FIBONACCI LATTICE - a way to scatter points almost perfectly evenly over a
#    sphere (like seeds in a sunflower head).  A regular latitude/longitude grid
#    would crowd points near the poles; the lunar south pole is our most important
#    target, so we cannot afford that distortion.
#
# 2. TANGENT-PLANE SAMPLING - to cut a patch we imagine laying a flat square
#    sheet of graph paper on the globe at the patch centre.  Every node of the
#    sheet is a fixed number of METRES east/north of the centre.  We convert each
#    node back to latitude/longitude and read the elevation there.  The result
#    has the same physical pixel size everywhere, on every body, including poles.
def fibonacci_lattice(radius_m: float, spacing_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spread points on a sphere. Returns (lat_deg, lon_deg)."""
    area_km2 = 4.0 * np.pi * (radius_m / 1000.0) ** 2
    n = int(round(area_km2 / spacing_km ** 2))        # one point per spacing^2 of area
    i = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * i / n))     # equal-area in latitude
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    lon = (360.0 * i / golden) % 360.0 - 180.0         # golden-angle spiral in longitude
    return lat.astype(np.float32), lon.astype(np.float32)


def patch_latlon(lat0, lon0, radius_m: float, n: int = PATCH, pixel_m: float = PIXEL_M):
    """
    Latitude/longitude of every pixel of tangent-plane patches.

    lat0, lon0 : arrays of shape (B,) in degrees - the patch centres.
    Returns lat, lon arrays of shape (B, n, n) in degrees.
    Row 0 is the northern edge, column 0 the western edge.

    Maths: inverse azimuthal-equidistant projection.  A node at (x east, y north)
    metres is at angular distance c = sqrt(x^2+y^2)/R from the centre, in compass
    direction theta = atan2(x, y).  Standard great-circle formulas then give its
    latitude and longitude.
    """
    lat0 = np.radians(np.atleast_1d(np.asarray(lat0, dtype=np.float64)))[:, None, None]
    lon0 = np.radians(np.atleast_1d(np.asarray(lon0, dtype=np.float64)))[:, None, None]
    half = (n - 1) / 2.0
    offs = (np.arange(n) - half) * pixel_m
    x = offs[None, None, :]                    # east, metres
    y = -offs[None, :, None]                   # north (row 0 = north)
    c = np.sqrt(x * x + y * y) / radius_m      # angular distance (radians)
    theta = np.arctan2(x, y)                   # bearing from north, clockwise
    sin_lat = np.sin(lat0) * np.cos(c) + np.cos(lat0) * np.sin(c) * np.cos(theta)
    lat = np.arcsin(np.clip(sin_lat, -1.0, 1.0))
    lon = lon0 + np.arctan2(np.sin(theta) * np.sin(c) * np.cos(lat0),
                            np.cos(c) - np.sin(lat0) * sin_lat)
    return np.degrees(lat), (np.degrees(lon) + 180.0) % 360.0 - 180.0


def bilinear(grid: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Read a global equirectangular grid at arbitrary lat/lon by bilinear interpolation.

    Grid convention used everywhere in this project:
      row 0 = +90 (north) ... last row = -90; col 0 = -180 ... last col = +180,
      pixel-registered (values sit at pixel centres).
    Longitude wraps around; latitude is clamped at the poles.
    """
    h, w = grid.shape
    r = (90.0 - lat) / 180.0 * h - 0.5
    c = (lon + 180.0) / 360.0 * w - 0.5
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    fr = (r - r0).astype(np.float32)
    fc = (c - c0).astype(np.float32)
    r0c = np.clip(r0, 0, h - 1)
    r1c = np.clip(r0 + 1, 0, h - 1)
    c0w = c0 % w
    c1w = (c0 + 1) % w
    top = grid[r0c, c0w] * (1 - fc) + grid[r0c, c1w] * fc
    bot = grid[r1c, c0w] * (1 - fc) + grid[r1c, c1w] * fc
    return (top * (1 - fr) + bot * fr).astype(np.float32)


def sample_patches(grid: np.ndarray, lat0, lon0, radius_m: float,
                   n: int = PATCH, pixel_m: float = PIXEL_M, chunk: int = 2048) -> np.ndarray:
    """Cut (B, n, n) tangent-plane patches out of a global grid (chunked to bound memory)."""
    lat0 = np.atleast_1d(lat0)
    lon0 = np.atleast_1d(lon0)
    out = np.empty((len(lat0), n, n), dtype=np.float32)
    for s in range(0, len(lat0), chunk):
        la, lo = patch_latlon(lat0[s:s + chunk], lon0[s:s + chunk], radius_m, n, pixel_m)
        out[s:s + chunk] = bilinear(grid, la, lo)
    return out


def great_circle_km(lat1, lon1, lat2, lon2, radius_m: float) -> np.ndarray:
    """Distance along the surface between points (haversine formula)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * radius_m * np.arcsin(np.sqrt(np.clip(a, 0, 1))) / 1000.0


# ==================================================================================================
# 5. TERRAIN FEATURES - hand-made descriptors (baseline, mobility, explanations)
# ==================================================================================================
# Hand-crafted terrain descriptors ("morphometry").
#
# Why keep hand-made features when we have a neural network?
#   * They are the BASELINE the network must beat (validation dashboard).
#   * They are human-readable, so they power the "why this matches" panel.
#   * A subset (slope, roughness) forms the rover-mobility axis.
#
# Every function works on a batch of patches: array (B, 32, 32) of elevation in
# metres, pixel size config.PIXEL_M.  All are pure numpy/scipy - no learning.
def morphometry(patches: np.ndarray, pixel_m: float = PIXEL_M) -> np.ndarray:
    """Return (B, len(MORPH_FEATURES)) float32 array, columns in MORPH_FEATURES order."""
    z = patches.astype(np.float32)
    b = z.shape[0]
    flat = z.reshape(b, -1)

    # --- amplitude ---------------------------------------------------------
    p5, p95 = np.percentile(flat, [5, 95], axis=1)
    relief = p95 - p5
    std = flat.std(axis=1)

    # --- slope: steepness between neighbouring pixels (central differences) -
    gy, gx = np.gradient(z, pixel_m, axis=(1, 2))
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).reshape(b, -1)
    slope_mean = slope.mean(axis=1)
    slope_p90 = np.percentile(slope, 90, axis=1)

    # --- roughness at two scales: what is left after smoothing --------------
    m3 = ndi.uniform_filter(z, size=(1, 3, 3), mode="nearest")     # ~11 km mean
    m9 = ndi.uniform_filter(z, size=(1, 9, 9), mode="nearest")     # ~33 km mean
    rough_small = np.sqrt(((z - m3) ** 2).reshape(b, -1).mean(axis=1))
    rough_mid = np.sqrt(((m3 - m9) ** 2).reshape(b, -1).mean(axis=1))

    # --- curvature: mean absolute Laplacian (metres per pixel^2) ------------
    # Laplacian = (sum of the 4 neighbours) - 4 * centre.  np.roll wraps around
    # the patch edge, so the outermost ring of pixels is discarded afterwards.
    lap = (np.roll(z, 1, 1) + np.roll(z, -1, 1) + np.roll(z, 1, 2) + np.roll(z, -1, 2) - 4 * z)
    curvature = np.abs(lap[:, 1:-1, 1:-1]).reshape(b, -1).mean(axis=1)

    # --- distribution shape --------------------------------------------------
    zmin, zmax, zmean = flat.min(axis=1), flat.max(axis=1), flat.mean(axis=1)
    hyps = (zmean - zmin) / np.maximum(zmax - zmin, 1e-3)
    skew = ((flat - zmean[:, None]) ** 3).mean(axis=1) / np.maximum(std, 1e-3) ** 3

    # --- closed depressions (impact craters, calderas, sinkholes) ------------
    # A pixel is a "pit" if it is lower than all 8 neighbours.
    nb_min = ndi.minimum_filter(z, footprint=_ring8(), mode="nearest")
    pit_density = (z < nb_min).reshape(b, -1).mean(axis=1)
    # Morphological closing fills holes narrower than 5 px (~18 km); the amount
    # of "fill" is the depth of those holes.
    closed = ndi.grey_closing(z, size=(1, 5, 5), mode="nearest")
    pit_depth = (closed - z).reshape(b, -1).mean(axis=1)

    # --- anisotropy: do slopes share a direction (dunes, ridges, valleys)? ---
    # Structure tensor of the gradient; compare its two eigenvalues.
    jxx = (gx * gx).reshape(b, -1).mean(axis=1)
    jyy = (gy * gy).reshape(b, -1).mean(axis=1)
    jxy = (gx * gy).reshape(b, -1).mean(axis=1)
    tr = jxx + jyy
    det_term = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0))
    aniso = det_term / np.maximum(tr, 1e-12)      # (l1-l2)/(l1+l2) in [0, 1]

    cols = {
        "relief_m": relief, "elev_std_m": std, "slope_mean_deg": slope_mean,
        "slope_p90_deg": slope_p90, "rough_small_m": rough_small, "rough_mid_m": rough_mid,
        "curvature": curvature, "hypsometric": hyps, "elev_skew": skew,
        "pit_density": pit_density, "pit_depth_m": pit_depth, "anisotropy": aniso,
    }
    return np.stack([cols[k] for k in MORPH_FEATURES], axis=1).astype(np.float32)


def _ring8() -> np.ndarray:
    """3x3 neighbourhood without the centre, as a (1,3,3) footprint for batched filters."""
    fp = np.ones((1, 3, 3), dtype=bool)
    fp[0, 1, 1] = False
    return fp


def to_comparable(feats: np.ndarray) -> np.ndarray:
    """
    Put features on comparable footing before standardising: skewed, strictly
    positive ones (relief, slope, roughness ...) are log-transformed, so that
    "10 m vs 20 m" counts the same as "1000 m vs 2000 m".
    """
    out = feats.astype(np.float32).copy()
    for j, name in enumerate(MORPH_FEATURES):
        if name in MORPH_LOG:
            out[:, j] = np.log10(np.maximum(out[:, j], 1e-3))
    return out


def network_input(patches: np.ndarray) -> np.ndarray:
    """
    Turn raw elevation patches (B, n, n) [metres] into the 2-channel network input.

      channel 0 - "how tall": (elevation - patch mean) / 1000 m, a FIXED scale so
                  that a 3 km mountain and a 30 m dune field stay different;
      channel 1 - "what shape": the same patch divided by its own standard
                  deviation, so that shape is visible even in very flat terrain.

    Absolute elevation above a datum is deliberately removed: "sea level",
    the lunar reference sphere and the martian areoid are not comparable.
    """
    z = patches.astype(np.float32)
    z = z - z.mean(axis=(1, 2), keepdims=True)
    ch0 = z / ELEV_SCALE_M
    ch1 = z / np.maximum(z.std(axis=(1, 2), keepdims=True), 1.0)   # floor: 1 m
    return np.stack([ch0, ch1], axis=1)


# ==================================================================================================
# 6. GLOBAL GRIDS - raw files to elevation and temperature grids
# ==================================================================================================
# Turn the raw downloads into small, uniform global grids ("assets").
#
# After this step every body has:
#   * dem   : int16 elevation in metres, row 0 = north pole, col 0 = 180 W
#   * therm : four temperature descriptors on that body's native thermal grid
# and Earth additionally has a land mask.
#
# These assets are what the app ships with: the website needs no internet for its data.
RAW = RAW_DIR


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------
def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Average f x f blocks (reduces resolution without aliasing)."""
    h, w = a.shape
    return a.reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def moon_dem() -> np.ndarray:
    """LOLA LDEM_16: little-endian int16, value*0.5 = metres above a 1737.4 km sphere,
    longitudes 0..360 E.  We average 2x2 -> 8 px/deg (3.79 km) and roll to -180..180."""
    raw = np.fromfile(RAW / "moon" / "ldem_16.img", dtype="<i2").reshape(2880, 5760)
    z = _block_mean(raw.astype(np.float32) * 0.5, 2)
    return np.roll(z, z.shape[1] // 2, axis=1).round().astype(np.int16)


def mars_dem() -> np.ndarray:
    """MOLA MEGDR 16 px/deg: big-endian int16 metres above the areoid, longitudes 0..360 E."""
    raw = np.fromfile(RAW / "mars" / "megt90n000eb.img", dtype=">i2").reshape(2880, 5760)
    return np.roll(raw, raw.shape[1] // 2, axis=1).astype(np.int16)


def _rasterize(name: str, shape: tuple[int, int]) -> np.ndarray:
    """Burn Natural Earth polygons (a zipped shapefile) into a boolean global grid."""
    import io
    import zipfile
    import shapefile                       # pyshp, pure Python
    from PIL import Image, ImageDraw
    h, w = shape
    zf = zipfile.ZipFile(RAW / "naturalearth" / f"{name}.zip")
    rd = shapefile.Reader(shp=io.BytesIO(zf.read(f"{name}.shp")), shx=io.BytesIO(zf.read(f"{name}.shx")),
                          dbf=io.BytesIO(zf.read(f"{name}.dbf")))
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for shp in rd.shapes():
        pts = np.asarray(shp.points, dtype=np.float64)
        parts = list(shp.parts) + [len(pts)]
        rings = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        # shapefile convention: outer rings are clockwise (negative signed area), holes are not
        def area(r):
            return 0.5 * np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1])
        for fill, want_outer in ((1, True), (0, False)):
            for r in rings:
                if len(r) >= 3 and ((area(r) < 0) == want_outer):
                    xy = np.column_stack([(r[:, 0] + 180.0) / 360.0 * w, (90.0 - r[:, 1]) / 180.0 * h])
                    draw.polygon([tuple(p) for p in xy], fill=fill)
    return np.asarray(img, dtype=bool)


def earth_dem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ETOPO 2022 60 arc-second: averaged 2x2 -> 2 arc-minute (3.7 km).
    Returns (elevation with water flattened, land mask, ice mask)."""
    import tifffile
    z = tifffile.imread(RAW / "earth" / "ETOPO_2022_v1_60s_N90W180_surface.tif").astype(np.float32)
    z = _block_mean(z, 2)
    # Sea = big connected regions at or below sea level (oceans, Black Sea, Caspian),
    # plus anything deeper than -450 m (no dry land on Earth is lower than the
    # Dead Sea shore at about -430 m).  Small shallow below-sea-level regions
    # (Death Valley, Danakil, Turpan, Qattara) are real dry land and must be
    # kept - they are classic Mars analogs.
    lab, n = ndi.label(z <= 0)
    idx = np.arange(1, n + 1)
    big = np.bincount(lab.ravel(), minlength=n + 1)[1:] > 20_000     # ~270,000 km2 at the equator
    deep = ndi.minimum(z, lab, idx) < -450.0
    sea = np.concatenate([[False], big | deep])[lab]
    z[sea] = 0.0                                                      # flat water surface
    # Lakes: ETOPO stores the lake FLOOR for some lakes (Baikal, Great Lakes),
    # which would look like a canyon.  Flatten every lake to its surface level.
    lakes = _rasterize("ne_10m_lakes", z.shape) & ~sea
    llab, ln = ndi.label(lakes)
    if ln:
        level = ndi.maximum(z, llab, np.arange(1, ln + 1))
        z[lakes] = np.concatenate([[0.0], level])[llab][lakes]
    ice = (_rasterize("ne_10m_glaciated_areas", z.shape)
           | _rasterize("ne_10m_antarctic_ice_shelves_polys", z.shape))
    return z.round().astype(np.int16), ~(sea | lakes), ice


# ---------------------------------------------------------------------------
# Temperature.  Output for every body: dict(lat=(H,), lon=(W,), data=(4,H,W))
# with the 4 layers in config.THERMAL_FEATURES order, all in kelvin.
# ---------------------------------------------------------------------------
def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    """Replace NaNs by the nearest valid value (used for the few unobserved cells)."""
    bad = np.isnan(a)
    if not bad.any():
        return a
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def moon_thermal() -> dict:
    """Diviner GCP tables: one row per (0.5 deg cell, 0.25 h local-time bin) with the
    bolometric temperature 'tbol'.  For every cell we build the full day-night
    curve (96 bins), fill gaps by interpolating around the clock, and summarise it."""
    cube = np.full((360, 720, 96), np.nan, dtype=np.float32)       # lat, lon, local time
    for band in DIVINER_BANDS:
        t = np.loadtxt(RAW / "diviner" / f"global_cumul_avg_cyl_{band}_002.tab",
                       delimiter=",", skiprows=1, usecols=(0, 1, 2, 10), dtype=np.float32)
        ok = t[:, 3] > 0                                           # -9999 = no observation
        t = t[ok]
        li = np.round((89.75 - t[:, 1]) / 0.5).astype(int)
        lo = np.round((t[:, 0] + 179.75) / 0.5).astype(int)
        ti = np.clip(np.round((t[:, 2] - 0.125) / 0.25).astype(int), 0, 95)
        cube[li, lo, ti] = t[:, 3]
    # Circular interpolation over local time for missing bins.
    hours = np.arange(96)
    flat = cube.reshape(-1, 96)
    coverage = np.isfinite(flat).mean(axis=1)
    for i in np.where((coverage < 1.0) & (coverage > 0.25))[0]:
        v = np.isfinite(flat[i])
        flat[i, ~v] = np.interp(hours[~v], hours[v], flat[i, v], period=96)
    flat[coverage <= 0.25] = np.nan
    cube = flat.reshape(360, 720, 96)
    hi = np.nanpercentile(cube, 97, axis=2)        # robust "maximum" (ignores 1-2 noisy bins)
    lo_ = np.nanpercentile(cube, 3, axis=2)
    layers = np.stack([np.nanmean(cube, axis=2), hi - lo_, hi, lo_])
    layers = np.stack([_fill_nan_nearest(x) for x in layers]).astype(np.float32)
    return {"lat": np.arange(89.75, -90, -0.5, dtype=np.float32),
            "lon": np.arange(-179.75, 180, 0.5, dtype=np.float32), "data": layers,
            "coverage": float(coverage.mean())}


def _read_mcd(path: Path):
    """Parse one MCD web-interface ASCII map: header lines, a latitude row, then
    one row per longitude."""
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    lat = np.array(lines[0].split("||")[1].split(), dtype=np.float32)
    rows = [ln.split("||") for ln in lines[2:]]
    lon = np.array([r[0] for r in rows], dtype=np.float32)
    data = np.array([r[1].split() for r in rows], dtype=np.float32)   # (lon, lat)
    return lat, lon, data.T                                            # -> (lat, lon)


def mars_thermal() -> dict:
    """Mars Climate Database: surface temperature at 12 local times for 4 seasons."""
    cube = []
    for ls in MCD_SEASONS:
        per_lt = []
        for lt in MCD_LOCAL_TIMES:
            lat, lon, d = _read_mcd(RAW / "mcd" / f"tsurf_ls{ls}_lt{lt}.txt")
            per_lt.append(d)
        cube.append(per_lt)
    cube = np.array(cube)                                   # (season, local time, lat, lon)
    day_max, day_min = cube.max(axis=1), cube.min(axis=1)   # per season
    layers = np.stack([cube.mean(axis=(0, 1)),              # mean
                       (day_max - day_min).mean(axis=0),    # typical day-night swing
                       day_max.max(axis=0),                 # warmest season's daily max
                       day_min.min(axis=0)])                # coldest season's daily min
    order = np.argsort(-lat)                                # make latitude run north -> south
    return {"lat": lat[order], "lon": lon, "data": layers[:, order].astype(np.float32)}


def earth_thermal() -> dict:
    """Earth surface temperature on a 0.1 deg grid.

    Primary source: MODIS Aqua land-surface temperature (satellite observation,
    ~13:30 and ~01:30 local time, 12 monthly 8-day composites of 2023, read from
    Microsoft Planetary Computer -> data/raw/modis/modis_2023.npz).
    Gap filler: NASA POWER / MERRA-2 climatology (cloudy months, polar night,
    oceans).  A statistic uses MODIS only where at least 6 of 12 months are valid.
    """
    merra = earth_thermal_merra()
    mpath = RAW / "modis" / "modis_2023.npz"
    if not mpath.exists():
        return merra
    m = np.load(mpath)
    day, night = m["lst_day"].astype(np.float32), m["lst_night"].astype(np.float32)   # (12, 1800, 3600)
    H, W = day.shape[1:]
    lats = (90 - (np.arange(H) + 0.5) * 180 / H).astype(np.float32)
    lons = (-180 + (np.arange(W) + 0.5) * 360 / W).astype(np.float32)
    both = np.isfinite(day) & np.isfinite(night)
    with np.errstate(all="ignore"):
        modis_layers = np.stack([
            np.nanmean(np.where(both, (day + night) / 2, np.nan), axis=0),     # mean
            np.nanmean(np.where(both, day - night, np.nan), axis=0),           # day-night swing
            np.nanmax(day, axis=0),                                            # warmest afternoon
            np.nanmin(night, axis=0)])                                         # coldest night
    enough = np.stack([both.sum(0) >= 6, both.sum(0) >= 6,
                       np.isfinite(day).sum(0) >= 6, np.isfinite(night).sum(0) >= 6])
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    out = np.empty_like(modis_layers)
    for i in range(4):
        fill = interp_nodes(merra["data"][i], merra["lat"], merra["lon"], LA, LO)
        out[i] = np.where(enough[i] & np.isfinite(modis_layers[i]), modis_layers[i], fill)
    return {"lat": lats, "lon": lons, "data": out.astype(np.float32),
            "modis_fraction": enough.mean(axis=(1, 2)).astype(np.float32),
            "source": "MODIS MYD11A2 v6.1 (2023) + NASA POWER MERRA-2 gap fill"}


def earth_bare_grid() -> np.ndarray | None:
    """0.05 deg grid: True where the ground is bare (MODIS NDVI never exceeds 0.2
    in any season of 2023).  Returns None if the MODIS file is missing."""
    mpath = RAW / "modis" / "modis_2023.npz"
    if not mpath.exists():
        return None
    nd = np.load(mpath)["ndvi_max"].astype(np.float32)
    return np.nan_to_num(nd, nan=1.0) < 0.2        # missing (ocean) counts as not bare


def earth_thermal_merra() -> dict:
    """NASA POWER (MERRA-2) skin-temperature climatology read from the public Zarr store.
    TS = monthly mean, TS_MAX_AVG / TS_MIN_AVG = monthly mean of the daily max / min."""
    import zarr
    g = zarr.open_group(POWER_ZARR, mode="r")
    lat, lon = g["lat"][:], g["lon"][:]
    months = slice(0, 12)                                   # index 12 is the annual value
    ts = g["TS"][months] + 273.15
    tmax = g["TS_MAX_AVG"][months] + 273.15
    tmin = g["TS_MIN_AVG"][months] + 273.15
    layers = np.stack([ts.mean(axis=0), (tmax - tmin).mean(axis=0), tmax.max(axis=0), tmin.min(axis=0)])
    order = np.argsort(-lat)
    layers = np.stack([_fill_nan_nearest(x) for x in layers[:, order]])
    meta = {k: dict(g[k].attrs) for k in ("TS", "TS_MAX_AVG", "TS_MIN_AVG")}
    return {"lat": lat[order].astype(np.float32), "lon": lon.astype(np.float32),
            "data": layers.astype(np.float32), "meta": json.dumps(meta, default=str)}


def interp_nodes(layer: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat, lon) -> np.ndarray:
    """Bilinear interpolation on a regular grid given by its node coordinates
    (lats descending, lons ascending, longitude wraps around)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    fr = np.interp(-lat, -lats.astype(np.float64), np.arange(len(lats)))        # fractional row
    dlon = float(lons[1] - lons[0])
    fc = ((lon - float(lons[0])) / dlon) % len(lons)                             # fractional col
    r0 = np.clip(np.floor(fr).astype(int), 0, len(lats) - 1)
    r1 = np.clip(r0 + 1, 0, len(lats) - 1)
    c0 = np.floor(fc).astype(int) % len(lons)
    c1 = (c0 + 1) % len(lons)
    wr, wc = fr - np.floor(fr), fc - np.floor(fc)
    top = layer[r0, c0] * (1 - wc) + layer[r0, c1] * wc
    bot = layer[r1, c0] * (1 - wc) + layer[r1, c1] * wc
    return (top * (1 - wr) + bot * wr).astype(np.float32)


# ---------------------------------------------------------------------------
# Shaded-relief textures for the offline Moon / Mars globes
# ---------------------------------------------------------------------------
def shaded_relief(dem_unused, body: str, width: int = 5760):
    """Colour + multi-directional hill-shade picture of the FULL-resolution LOLA / MOLA grid
    (5760 x 2880 px, 16 px/degree) used as the offline Moon / Mars globe texture.
    (The first argument is kept for backward compatibility and ignored.)"""
    from PIL import Image
    if body == "moon":
        z = np.fromfile(RAW / "moon" / "ldem_16.img", dtype="<i2").reshape(2880, 5760).astype(np.float32) * 0.5
        exag, ramp = 5.0, ((88, 88, 94), (206, 203, 197))
    else:
        z = np.fromfile(RAW / "mars" / "megt90n000eb.img", dtype=">i2").reshape(2880, 5760).astype(np.float32)
        exag, ramp = 7.0, ((110, 52, 30), (232, 170, 120))
    z = np.roll(z, z.shape[1] // 2, axis=1)                    # 0..360 E  ->  -180..180
    H, W = z.shape
    px = 2 * np.pi * RADIUS_M[body] / W
    gy, gx = np.gradient(z, px)
    lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H)[:, None]
    gx = gx / np.cos(lat).clip(0.05, 1)                         # east-west pixels shrink towards the poles
    slope = np.arctan(exag * np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    alt = np.radians(42)
    shade = 0
    for az, w in ((315, .5), (270, .25), (0, .25)):             # light from 3 directions: no "dead" slopes
        shade = shade + w * (np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(np.radians(az) - aspect))
    shade = np.clip(shade / np.sin(alt), 0, 1.5)
    lo, hi = np.percentile(z, [1, 99])
    t = np.clip((z - lo) / (hi - lo), 0, 1)[..., None]
    rgb = (np.array(ramp[0]) * (1 - t) + np.array(ramp[1]) * t) * (0.30 + 0.75 * shade[..., None])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def earth_patch_surface(lat, lon, therm: dict, ice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per Earth patch: (thermal features averaged over the 118 km footprint,
    fraction of the footprint that is bare ground and not ice).

    The footprint average matters on Earth because the 0.1 deg MODIS grid is
    much finer than a patch; for the Moon (0.5 deg) and Mars (3.8 deg) grids the
    centre value is already a footprint-scale average."""
    R = RADIUS_M["earth"]
    n = 8                                                     # 8x8 sample points per footprint is plenty
    px = PIXEL_M * PATCH / n
    out_t = np.empty((len(lat), len(THERMAL_FEATURES)), np.float32)
    for s in range(0, len(lat), 4096):
        la, lo = patch_latlon(lat[s:s + 4096], lon[s:s + 4096], R, n=n, pixel_m=px)
        for i in range(len(THERMAL_FEATURES)):
            v = interp_nodes(therm["data"][i], therm["lat"], therm["lon"], la.ravel(), lo.ravel())
            out_t[s:s + 4096, i] = v.reshape(la.shape).mean(axis=(1, 2))
    bare = earth_bare_grid()
    if bare is None:
        return out_t, np.full(len(lat), np.nan, np.float32)
    # bring the ice mask to the 0.05 deg bare grid, then combine
    fy, fx = ice.shape[0] // bare.shape[0], ice.shape[1] // bare.shape[1]
    ice_c = ice[: bare.shape[0] * fy, : bare.shape[1] * fx].reshape(bare.shape[0], fy, bare.shape[1], fx).mean(axis=(1, 3)) > 0.5
    good = (bare & ~ice_c).astype(np.float32)
    frac = sample_patches(good, lat, lon, R).mean(axis=(1, 2))
    return out_t, frac.astype(np.float32)


# ==================================================================================================
# 7. MODIS - Earth temperature and vegetation from Microsoft Planetary Computer
# ==================================================================================================
# Earth surface data from NASA MODIS, read from Microsoft Planetary Computer.
#
# Why MODIS?
#   * Land Surface Temperature (MYD11A2) is MEASURED by a thermal-infrared
#     radiometer from orbit - the same kind of measurement Diviner makes on the
#     Moon.  That is more comparable than a weather-model temperature.
#   * NDVI (MYD13A1) tells us how much living vegetation covers the ground.
#     The Moon and Mars have none, so a good analog should be (nearly) bare.
#
# How the data are stored
# -----------------------
# MODIS cuts the Earth into 460 tiles of 1200 x 1200 km on a SINUSOIDAL grid
# (an equal-area map projection).  Planetary Computer stores every tile as a
# Cloud-Optimised GeoTIFF (COG): a GeoTIFF that also contains shrunken copies
# ("overviews") and can be read in small pieces over HTTP.  We only need ~7 km
# detail, so we read a shrunken copy of each tile -> a few hundred KB per tile
# instead of the full file.
#
# The sinusoidal projection is simple enough to invert by hand:
#     x = R * lon * cos(lat),   y = R * lat        (angles in radians)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/modiseuwest/modis-061-cogs"
R_SIN = 6371007.181                     # sphere radius of the MODIS sinusoidal grid (m)
TILE_M = 1111950.5197665                # tile width/height (m)
X0, Y0 = -20015109.354, 10007554.677    # upper-left corner of tile h=0, v=0

# Aqua satellite: overpass ~13:30 and ~01:30 local time, i.e. close to the
# afternoon maximum and the night-time cooling of the surface.
LST_COLLECTION, LST_PREFIX = "modis-11A2-061", "MYD11A2"
NDVI_COLLECTION, NDVI_PREFIX = "modis-13A1-061", "MYD13A1"
YEAR = 2023
# one 8-day LST composite per month (start day-of-year nearest the middle of the month)
LST_DOYS = [9, 41, 73, 105, 137, 169, 193, 225, 257, 289, 321, 345]
# four 16-day NDVI composites, one per season (Aqua composites start on DOY 9 + 16k)
NDVI_DOYS = [25, 105, 201, 281]


def _gdal_env() -> None:
    """GDAL reads the COGs over HTTP; tell it to trust the same CA bundle as everything else."""
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if os.environ.get(var):
            os.environ.setdefault("CURL_CA_BUNDLE", os.environ[var])
    if os.path.exists("/root/.ccr/ca-bundle.crt"):
        os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def token() -> str:
    """Planetary Computer data are public but need a free, short-lived read token."""
    return _http().get(SAS, timeout=60).json()["token"]


def search(collection: str, prefix: str, doy: int, asset: str) -> dict[tuple[int, int], str]:
    """All tile URLs of one composite: {(h, v): href}."""
    from datetime import date, timedelta
    start = date(YEAR, 1, 1) + timedelta(days=doy - 1)
    body = {"collections": [collection], "limit": 1000,
            "datetime": f"{start}/{start + timedelta(days=7)}"}
    out: dict[tuple[int, int], str] = {}
    url = f"{STAC}/search"
    while True:
        r = _http().post(url, json=body, timeout=120).json()
        for f in r["features"]:
            m = re.match(rf"{prefix}\.A{YEAR}(\d{{3}})\.h(\d\d)v(\d\d)", f["id"])
            if m and int(m.group(1)) == doy:
                out[(int(m.group(2)), int(m.group(3)))] = f["assets"][asset]["href"]
        nxt = [lk for lk in r.get("links", []) if lk.get("rel") == "next"]
        if not nxt:
            return out
        body = nxt[0].get("body", body)


def grid_lookup(height: int, width: int, n: int):
    """For every cell of a global lat/lon grid (row 0 = north), find the MODIS tile
    and the pixel inside a tile read at n x n.  Returns h, v, row, col arrays."""
    lat = np.radians(90 - (np.arange(height) + 0.5) * 180 / height)[:, None]
    lon = np.radians(-180 + (np.arange(width) + 0.5) * 360 / width)[None, :]
    x = R_SIN * lon * np.cos(lat)
    y = R_SIN * lat * np.ones_like(lon)
    fx, fy = (x - X0) / TILE_M, (Y0 - y) / TILE_M
    h, v = np.floor(fx).astype(int), np.floor(fy).astype(int)
    col = np.clip(((fx - h) * n).astype(int), 0, n - 1)
    row = np.clip(((fy - v) * n).astype(int), 0, n - 1)
    return h, v, row, col


def read_tile(href: str, tok: str, n: int, scale: float, nodata_below: float) -> np.ndarray | None:
    """Read one COG at n x n (GDAL picks the closest overview), apply the scale factor
    and turn fill values into NaN."""
    import rasterio
    from rasterio.enums import Resampling
    for attempt in range(3):
        try:
            with rasterio.open(f"{href}?{tok}") as ds:
                a = ds.read(1, out_shape=(n, n), resampling=Resampling.average, masked=True)
            a = a.astype(np.float32).filled(np.nan) * scale
            a[a < nodata_below] = np.nan
            return a
        except Exception:                                   # network hiccup: retry
            if attempt == 2:
                return None
    return None


def composite_grid(collection: str, prefix: str, doy: int, asset: str, height: int, width: int,
                   n: int, scale: float, nodata_below: float, tok: str, lookup, workers: int = 12):
    """Assemble one global composite on a lat/lon grid."""
    hrefs = search(collection, prefix, doy, asset)
    h, v, row, col = lookup
    out = np.full((height, width), np.nan, np.float32)
    with ThreadPoolExecutor(workers) as ex:
        tiles = dict(zip(hrefs, ex.map(lambda k: read_tile(hrefs[k], tok, n, scale, nodata_below), hrefs)))
    for (th, tv), a in tiles.items():
        if a is None:
            continue
        sel = (h == th) & (v == tv)
        out[sel] = a[row[sel], col[sel]]
    return out, len(hrefs), sum(a is None for a in tiles.values())


def fetch_all(log=print) -> dict:
    """Download the LST and NDVI composites. Returns arrays ready to save."""
    _gdal_env()
    tok = token()
    out = {}
    H, W = 1800, 3600                                         # 0.1 deg LST grid
    look = grid_lookup(H, W, 150)
    day, night = [], []
    for i, doy in enumerate(LST_DOYS):
        if i % 4 == 0:
            tok = token()                                     # tokens expire after ~1 h
        d, nt, bad = composite_grid(LST_COLLECTION, LST_PREFIX, doy, "LST_Day_1km", H, W, 150,
                                    0.02, 150.0, tok, look)
        nn, _, bad2 = composite_grid(LST_COLLECTION, LST_PREFIX, doy, "LST_Night_1km", H, W, 150,
                                     0.02, 150.0, tok, look)
        day.append(d)
        night.append(nn)
        log(f"  LST {YEAR} day {doy:3d}: {nt} tiles, {bad + bad2} failed, "
            f"{np.isfinite(d).mean() * 100:.0f}% of grid has data")
    out["lst_day"] = np.stack(day).astype(np.float16)
    out["lst_night"] = np.stack(night).astype(np.float16)
    H2, W2 = 3600, 7200                                       # 0.05 deg NDVI grid
    look2 = grid_lookup(H2, W2, 300)
    nd = []
    tok = token()
    for doy in NDVI_DOYS:
        a, nt, bad = composite_grid(NDVI_COLLECTION, NDVI_PREFIX, doy, "500m_16_days_NDVI", H2, W2, 300,
                                    1e-4, -0.25, tok, look2)
        nd.append(a)
        log(f"  NDVI {YEAR} day {doy:3d}: {nt} tiles, {bad} failed")
    out["ndvi_max"] = np.nanmax(np.stack(nd), axis=0).astype(np.float16)
    return out


# ==================================================================================================
# 8. EARTH TEXTURE - true-colour globe from MODIS MCD43A4
# ==================================================================================================
# A detailed true-colour Earth for the globe, built from Microsoft Planetary Computer.
#
# Land colour  : MODIS MCD43A4 v6.1 (Nadir BRDF-Adjusted Reflectance, 500 m), bands
#                1 (red), 4 (green), 3 (blue).  "BRDF-adjusted" means the colours are
#                corrected to a standard sun/view geometry, so neighbouring tiles from
#                different orbits match - ideal for a seamless global picture.
# Composite    : northern-summer dates for the northern hemisphere, southern-summer
#                dates for the south (less snow, polar day), averaged where both exist.
# Relief       : hill-shading from ETOPO 2022 multiplies the colour (mountains pop out).
# Oceans       : coloured by depth from ETOPO bathymetry (shelves light, abyss dark).
#
# Output: a tile pyramid (512-px WebP tiles, geographic tiling, levels 0-4, final
# level 16384 x 8192 px ~ 2.4 km/px) in static/textures/earth/{z}/{x}/{y}.webp
# that CesiumJS reads offline.
COLLECTION, PREFIX = "modis-43A4-061", "MCD43A4"
NORTH_DOYS = [179, 210]       # 28 Jun, 29 Jul 2023
SOUTH_DOYS = [15, 349]        # 15 Jan, 15 Dec 2023
BANDS = ["Nadir_Reflectance_Band1", "Nadir_Reflectance_Band4", "Nadir_Reflectance_Band3"]   # R, G, B
TEX_W, TEX_H = 16384, 8192
TEX_TILE = 512
N_READ = 600                  # read each 2400-px MODIS tile at 600 px (~1.85 km)


_GROUPS: dict[int, tuple] = {}


def _strip_groups(r0: int, r1: int):
    """For texture rows r0..r1: pixels grouped by MODIS tile, computed once and cached.
    Returns {(h, v): (flat pixel indices, rows in tile, cols in tile)}."""
    if r0 not in _GROUPS:
        h, v, row, col = _strip_lookup(r0, r1)
        tid = (h.astype(np.int32) * 64 + v).ravel()
        order = np.argsort(tid, kind="stable")
        ts = tid[order]
        uniq, starts = np.unique(ts, return_index=True)
        ends = np.append(starts[1:], len(ts))
        rr, cc = row.ravel(), col.ravel()
        g = {}
        for u, a0, a1 in zip(uniq, starts, ends):
            idx = order[a0:a1]
            g[(int(u) // 64, int(u) % 64)] = (idx.astype(np.int32), rr[idx], cc[idx])
        _GROUPS[r0] = g
    return _GROUPS[r0]


def _date_rgb(doy: int, tok: str, log=print) -> np.ndarray:
    """(3, TEX_H, TEX_W) uint16 reflectance x 10000 for one day (0 = no data)."""
    from concurrent.futures import ThreadPoolExecutor
    out = np.zeros((3, TEX_H, TEX_W), np.uint16)
    hrefs = {b: search(COLLECTION, PREFIX, doy, b) for b in BANDS}
    keys = sorted(set.intersection(*[set(h) for h in hrefs.values()]))
    for bi, band in enumerate(BANDS):
        with ThreadPoolExecutor(12) as ex:
            tiles = dict(zip(keys, ex.map(lambda k: read_tile(hrefs[band][k], tok, N_READ, 1.0, 1.0), keys)))
        for s in range(0, TEX_H, 512):                          # fill the texture in strips (bounded memory)
            e = min(s + 512, TEX_H)
            strip = np.zeros((e - s) * TEX_W, np.uint16)
            for key, (idx, rr, cc) in _strip_groups(s, e).items():
                a = tiles.get(key)
                if a is not None:
                    strip[idx] = np.nan_to_num(np.clip(a[rr, cc], 0, 16000), nan=0).astype(np.uint16)
            out[bi, s:e] = strip.reshape(e - s, TEX_W)
        log(f"    day {doy} {band[-5:]}: {len(keys)} tiles, {sum(a is None for a in tiles.values())} failed")
    return out


def _strip_lookup(r0: int, r1: int):
    """MODIS sinusoidal tile/pixel for rows r0..r1 of the texture grid."""
    lat = np.radians(90 - (np.arange(r0, r1) + 0.5) * 180 / TEX_H)[:, None]
    lon = np.radians(-180 + (np.arange(TEX_W) + 0.5) * 360 / TEX_W)[None, :]
    x = R_SIN * lon * np.cos(lat)
    y = R_SIN * lat * np.ones_like(lon)
    fx, fy = (x - X0) / TILE_M, (Y0 - y) / TILE_M
    h, v = np.floor(fx).astype(np.int16), np.floor(fy).astype(np.int16)
    col = np.clip(((fx - h) * N_READ).astype(np.int16), 0, N_READ - 1)
    row = np.clip(((fy - v) * N_READ).astype(np.int16), 0, N_READ - 1)
    return h, v, row, col


def _tone(refl: np.ndarray) -> np.ndarray:
    """Reflectance (0..1) -> display 0..1.  A gentle S-curve on a power law, similar in
    spirit to NASA's true-colour products: dark vegetation stays visible, deserts glow,
    snow does not blow out."""
    v = np.clip(refl / 0.42, 0, 1) ** 0.62
    return np.clip(1.06 * v - 0.03, 0, 1)


def make_earth_texture(log=print) -> None:
    from PIL import Image
    import tifffile
    _gdal_env()
    tok = token()
    dates = {}
    cache = RAW_DIR / "modis"
    cache.mkdir(parents=True, exist_ok=True)
    for i, doy in enumerate(NORTH_DOYS + SOUTH_DOYS):
        f = cache / f"mcd43a4_2023_{doy:03d}.npy"
        if f.exists():                                       # downloaded before: reuse (raw data, not shipped)
            dates[doy] = np.load(f, mmap_mode="r")
            continue
        if i == 2:
            tok = token()
        log(f"  MCD43A4 day {doy} ...")
        dates[doy] = _date_rgb(doy, tok, log)
        np.save(f, dates[doy])

    log("  compositing + relief ...")
    z = tifffile.imread(RAW_DIR / "earth" / "ETOPO_2022_v1_60s_N90W180_surface.tif").astype(np.float32)
    zimg = Image.fromarray(z).resize((TEX_W, TEX_H), Image.BILINEAR)
    z = np.asarray(zimg, dtype=np.float32)
    sea = _sea_mask(TEX_H, TEX_W)
    rgb8 = np.zeros((TEX_H, TEX_W, 3), np.uint8)
    lat_rows = 90 - (np.arange(TEX_H) + 0.5) * 180 / TEX_H
    px_m = 2 * np.pi * RADIUS_M["earth"] / TEX_W
    for s in range(0, TEX_H, 512):
        e = min(s + 512, TEX_H)
        pad0, pad1 = max(s - 1, 0), min(e + 1, TEX_H)
        zz = z[pad0:pad1]
        gy, gx = np.gradient(zz, px_m)
        coslat = np.cos(np.radians(lat_rows[pad0:pad1]))[:, None].clip(0.05, 1)
        gx = gx / coslat                                            # metres per pixel shrink with latitude
        sl = slice(s - pad0, s - pad0 + (e - s))
        gx, gy, zz = gx[sl], gy[sl], zz[sl]
        # hill-shade: light from the north-west, 40 deg up, vertical exaggeration x6
        slope = np.arctan(6 * np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        az, alt = np.radians(315), np.radians(40)
        shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
        shade = np.clip(shade / np.sin(alt), 0, 1.6)
        north = lat_rows[s:e] >= 0
        pref = np.where(north[:, None, None], 0, 1)                 # 0 = use north dates first
        a = [dates[d][:, s:e].astype(np.float32) for d in NORTH_DOYS]
        b = [dates[d][:, s:e].astype(np.float32) for d in SOUTH_DOYS]

        def mean_valid(lst):
            stack = np.stack(lst)                                   # (n, 3, h, w)
            ok = (stack > 0).all(axis=1, keepdims=True)
            cnt = ok.sum(0)
            return np.where(cnt > 0, (stack * ok).sum(0) / np.maximum(cnt, 1), 0), cnt[0] > 0

        ma, oka = mean_valid(a)
        mb, okb = mean_valid(b)
        first = np.where(pref[:, :, 0][None] == 0, ma, mb)
        second = np.where(pref[:, :, 0][None] == 0, mb, ma)
        ok1 = np.where(pref[:, :, 0] == 0, oka, okb)
        ok2 = np.where(pref[:, :, 0] == 0, okb, oka)
        land = np.where(ok1[None], first, second) / 10000.0
        has_land = ok1 | ok2
        col = _tone(land)                                           # (3, h, w)
        col = col * (0.78 + 0.32 * shade)[None]
        # oceans: depth ramp + faint sea-floor relief
        depth = np.clip(-zz, 0, 8000)
        t = np.sqrt(depth / 8000.0)
        shallow, deep = np.array([0.30, 0.55, 0.68]), np.array([0.02, 0.07, 0.17])
        ocean = (shallow[:, None, None] * (1 - t) + deep[:, None, None] * t) * (0.86 + 0.18 * shade)[None]
        ocean = np.where((depth < 30)[None], shallow[:, None, None] * 1.05, ocean)
        # land without MODIS data (rare): neutral terrain colour, ice-white if very high/polar
        fallback = np.array([0.55, 0.52, 0.46])[:, None, None] * (0.7 + 0.4 * shade)[None]
        polar = (np.abs(lat_rows[s:e]) > 60)[:, None] & (zz > 0)
        fallback = np.where(polar[None], np.array([0.92, 0.94, 0.97])[:, None, None] * (0.8 + 0.25 * shade)[None], fallback)
        # Ocean is decided by ELEVATION, not by MODIS: MODIS "land" tiles also cover shallow coastal sea,
        # where its reflectance is dark and noisy.  Lakes and dry depressions (Death Valley) keep MODIS colour.
        is_ocean = sea[s:e]
        img = np.where(is_ocean[None], ocean, np.where(has_land[None], col, fallback))
        rgb8[s:e] = (np.clip(img, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    del dates
    log("  writing tile pyramid ...")
    write_pyramid(Image.fromarray(rgb8), log)
    Image.fromarray(rgb8).resize((4096, 2048), Image.LANCZOS).save(STATIC_DIR / "textures" / "earth_preview.jpg", quality=88)


def _sea_mask(h: int, w: int) -> np.ndarray:
    """Sea = large connected regions at or below sea level, or anything deeper than -450 m
    (same rule as the dataset, section 6), on the ETOPO grid, then resized."""
    from PIL import Image
    from scipy import ndimage as ndi
    import tifffile
    z = tifffile.imread(RAW_DIR / "earth" / "ETOPO_2022_v1_60s_N90W180_surface.tif").astype(np.float32)
    z = z.reshape(5400, 2, 10800, 2).mean(axis=(1, 3))
    lab, n = ndi.label(z <= 0)
    idx = np.arange(1, n + 1)
    big = np.bincount(lab.ravel(), minlength=n + 1)[1:] > 20_000
    deep = ndi.minimum(z, lab, idx) < -450.0
    sea = np.concatenate([[False], big | deep])[lab]
    return np.asarray(Image.fromarray(sea.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127


def write_pyramid(img, log=print) -> None:
    """Geographic tiling (2 x 1 tiles at level 0), 512-px WebP tiles."""
    from PIL import Image
    root = STATIC_DIR / "textures" / "earth"
    levels = int(np.log2(TEX_W // (2 * TEX_TILE)))                         # 16384 -> levels 0..4
    for zl in range(levels + 1):
        nx, ny = 2 * 2 ** zl, 2 ** zl
        lvl = img.resize((nx * TEX_TILE, ny * TEX_TILE), Image.LANCZOS) if zl < levels else img
        for x in range(nx):
            (root / str(zl) / str(x)).mkdir(parents=True, exist_ok=True)
            for y in range(ny):
                lvl.crop((x * TEX_TILE, y * TEX_TILE, (x + 1) * TEX_TILE, (y + 1) * TEX_TILE)).save(
                    root / str(zl) / str(x) / f"{y}.webp", quality=84, method=4)
        log(f"    level {zl}: {nx} x {ny} tiles")


# ==================================================================================================
# 9. NEURAL NETWORK - the AnalogNet terrain encoder
# ==================================================================================================
# The neural network: a small convolutional encoder that turns a terrain patch
# into a 32-number "terrain fingerprint".
#
# Plain-language picture
# ----------------------
# A convolution slides a tiny 3x3 window over the elevation map and reacts to a
# local pattern (an edge, a bowl, a ridge).  Stacking layers lets later windows
# react to combinations of earlier patterns (a ring of edges = a crater rim).
# At the end we summarise "how much of each pattern is present anywhere in the
# patch" (global pooling), and squeeze that into 32 numbers.
#
# The network is deliberately small (~0.2 million weights): the data are small
# 32x32 height maps, it must train on a laptop CPU in minutes, and a small model
# is easier to explain and less likely to memorise sensor quirks.
def _block(cin: int, cout: int) -> nn.Sequential:
    """Two 3x3 convolutions; the second one halves the image size (stride 2).
    GroupNorm keeps activations well-scaled and, unlike BatchNorm, behaves the
    same in training and inference and does not mix statistics across bodies."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, stride=2, padding=1), nn.GroupNorm(4, cout), nn.ReLU(inplace=True),
    )


class TerrainEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.features = nn.Sequential(_block(2, 24), _block(24, 48), _block(48, 96))
        self.head = nn.Sequential(nn.Linear(96 * 2, 96), nn.ReLU(inplace=True), nn.Linear(96, embed_dim))

    @staticmethod
    def prepare(elev: torch.Tensor) -> torch.Tensor:
        """(B, H, W) elevation in metres -> (B, 2, H, W) network input.
        Same recipe as theia.features.network_input (see there for why)."""
        z = elev - elev.mean(dim=(1, 2), keepdim=True)
        ch0 = z / ELEV_SCALE_M
        ch1 = z / z.std(dim=(1, 2), keepdim=True).clamp_min(1.0)
        return torch.stack([ch0, ch1], dim=1)

    def forward(self, elev: torch.Tensor) -> torch.Tensor:
        f = self.features(self.prepare(elev))
        # average pooling = "how much of each pattern overall",
        # max pooling     = "is there at least one strong instance"
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        # L2-normalise: every fingerprint has length 1, so the dot product of two
        # fingerprints is their cosine similarity (1 = same direction).
        return TF.normalize(self.head(pooled), dim=1)


def dihedral_views(x: torch.Tensor) -> list[torch.Tensor]:
    """The 8 rotations/mirror images of a square patch.  A landscape does not
    change when the map is rotated, so at inference we average the fingerprint
    over all 8 to make it exactly orientation-independent."""
    out = []
    for flip in (False, True):
        y = x.flip(-1) if flip else x
        for k in range(4):
            out.append(torch.rot90(y, k, dims=(-2, -1)))
    return out


@torch.no_grad()
def embed(model: TerrainEncoder, elev, batch: int = 2048, tta: bool = True):
    """Fingerprints for an array of patches (N, H, W) in metres -> (N, D) float32 numpy."""
    import numpy as np
    model.eval()
    outs = []
    for s in range(0, len(elev), batch):
        x = torch.as_tensor(np.asarray(elev[s:s + batch], dtype=np.float32))
        if tta:
            z = torch.stack([model(v) for v in dihedral_views(x)]).mean(0)
            z = TF.normalize(z, dim=1)
        else:
            z = model(x)
        outs.append(z.numpy())
    return np.concatenate(outs)


# ==================================================================================================
# 10. TRAINING - self-supervised contrastive learning
# ==================================================================================================
# Self-supervised contrastive training (SimCLR-style).
#
# The problem: nobody has labelled "this Earth valley is similar to that Mars
# valley", so we cannot train with labels.  Instead we teach the network a game:
#
#   1. Take a terrain patch.  Make two slightly different "views" of it
#      (rotate, mirror, shift, rescale a little, add sensor-like noise/blur).
#   2. Do this for 256 patches at once -> 512 views.
#   3. Ask the network: for each view, which of the other 511 is its twin?
#
# To win, the network must describe WHAT THE TERRAIN IS (craters, dunes, ridges)
# and ignore what we randomised (orientation, exact position, sensor noise).
# Patches from Earth, Moon and Mars are mixed in every batch, and the noise/blur
# tricks imitate the differences between the three instruments, so the network
# has no reason to care which body a patch came from.
#
# The loss (NT-Xent) is just "cross-entropy of picking the right twin".
def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """One random 'view' of each training tile.  x: (B, 48, 48) metres -> (B, 28, 28)."""
    b = x.shape[0]
    # random window: every tile gets its own 28x28 crop position.  Because the
    # two views of a tile come from two separate calls, they may overlap fully,
    # partly or not at all - the network must recognise the LANDSCAPE.
    hi = x.shape[1] - CROP + 1
    i = torch.randint(0, hi, (b,), generator=gen)
    j = torch.randint(0, hi, (b,), generator=gen)
    ar = torch.arange(CROP)
    x = x[torch.arange(b)[:, None, None], (i[:, None] + ar)[:, :, None], (j[:, None] + ar)[:, None, :]]
    # random rotation by 0/90/180/270 degrees and random mirror
    x = torch.rot90(x, int(torch.randint(0, 4, (1,), generator=gen)), dims=(1, 2))
    if torch.rand(1, generator=gen) < 0.5:
        x = x.flip(2)
    x = x - x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True)
    # height scaled by +-15 %: "the same landform, slightly taller or flatter"
    x = x * (0.85 + 0.30 * torch.rand(b, 1, 1, generator=gen))
    # sensor-like blur on half the patches (instruments differ in sharpness)
    blur = TF.avg_pool2d(x[:, None], 3, stride=1, padding=1, count_include_pad=False)[:, 0]
    mix = (torch.rand(b, 1, 1, generator=gen) < 0.5).float() * torch.rand(b, 1, 1, generator=gen)
    x = x * (1 - mix) + blur * mix
    # sensor-like noise: up to 4 % of the patch's own relief, plus up to 3 m
    sigma = std * 0.04 * torch.rand(b, 1, 1, generator=gen) + 3.0 * torch.rand(b, 1, 1, generator=gen)
    return x + sigma * torch.randn(x.shape, generator=gen)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE):
    """Contrastive loss.  z1[i] and z2[i] are fingerprints of two views of patch i.
    Returns (loss, fraction of views whose nearest other view is the true twin)."""
    z = torch.cat([z1, z2])                              # (2B, D), unit length
    sim = z @ z.t() / temperature                        # cosine similarities, sharpened
    sim.fill_diagonal_(float("-inf"))                    # a view may not pick itself
    b = z1.shape[0]
    target = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)])
    loss = TF.cross_entropy(sim, target)
    acc = (sim.argmax(dim=1) == target).float().mean()
    return loss, acc


def pca2(x: np.ndarray, basis: np.ndarray | None = None):
    """Project points to 2-D with PCA (the two directions of largest spread)."""
    xc = x - x.mean(axis=0)
    if basis is None:
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        basis = vt[:2]
    return xc @ basis.T, basis


def train_member(patches: dict[str, np.ndarray], seed: int, steps: int = TRAIN_STEPS,
                 batch: int = BATCH, probe: dict[str, np.ndarray] | None = None,
                 snapshot_every: int = 100, should_stop: Callable[[], bool] | None = None
                 ) -> Iterator[dict]:
    """
    Train one encoder.  This is a GENERATOR: it yields a small progress report
    every few steps so that a script can print it and the website can draw
    it live ("Watch the AI think").  The last report has key 'model'.

    patches : {body: int16/float array (N, 48, 48) in metres} - the training tiles
    probe   : optional {body: (n, 32, 32)} patches fixed sample whose fingerprints are
              projected to 2-D at every snapshot, to watch clusters form.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TerrainEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(     # warm-up, then smooth cosine decay
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    bodies = list(patches)
    per_body = batch // len(bodies)
    # each ensemble member sees a different random 80 % of the data -> diversity
    pools = {b: rng.permutation(len(patches[b]))[: int(0.8 * len(patches[b]))] for b in bodies}
    basis = None
    t0 = time.time()
    ema_loss = ema_acc = None
    for step in range(1, steps + 1):
        if should_stop is not None and should_stop():
            break
        # body-balanced batch: equal numbers of Earth, Moon and Mars patches
        x = np.concatenate([patches[b][np.sort(rng.choice(pools[b], per_body, replace=False))]
                            for b in bodies]).astype(np.float32)
        x = torch.from_numpy(x)
        model.train()
        loss, acc = nt_xent(model(augment(x, gen)), model(augment(x, gen)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema_loss = loss.item() if ema_loss is None else 0.95 * ema_loss + 0.05 * loss.item()
        ema_acc = acc.item() if ema_acc is None else 0.95 * ema_acc + 0.05 * acc.item()
        if step % 10 == 0 or step == steps:
            rep = {"step": step, "steps": steps, "loss": round(ema_loss, 4),
                   "twin_acc": round(ema_acc, 4), "seconds": round(time.time() - t0, 1)}
            if probe is not None and (step % snapshot_every == 0 or step == steps or step == 10):
                model.eval()
                with torch.no_grad():
                    emb = np.concatenate([model(torch.from_numpy(probe[b].astype(np.float32))).numpy()
                                          for b in bodies])
                xy, new_basis = pca2(emb)
                if basis is not None:          # keep the picture from flipping between frames
                    for k in range(2):
                        if np.dot(new_basis[k], basis[k]) < 0:
                            xy[:, k] *= -1
                            new_basis[k] *= -1
                basis = new_basis
                rep["xy"] = np.round(xy, 3).tolist()
                rep["mixing"] = round(body_mixing(emb, [len(probe[b]) for b in bodies]), 3)
            yield rep
    yield {"model": model, "seconds": round(time.time() - t0, 1)}


def body_mixing(emb: np.ndarray, counts: list[int], k: int = 10) -> float:
    """
    How well are the bodies mixed in fingerprint space?  For every probe patch
    look at its k nearest neighbours and count how many come from a DIFFERENT
    body.  Divided by the value expected if bodies were perfectly shuffled:
      0 = each body sits in its own island, 1 = perfectly mixed.
    (Perfect mixing is not the goal - the Moon really is more cratered than
    Earth - but an island per body would mean the network learned the sensor.)
    """
    labels = np.repeat(np.arange(len(counts)), counts)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argpartition(-sim, k, axis=1)[:, :k]
    other = (labels[nn_idx] != labels[:, None]).mean()
    n = len(labels)
    expected = 1.0 - sum(c * (c - 1) for c in counts) / (n * (n - 1))
    return float(other / expected)


# ==================================================================================================
# 11. MATCHING ENGINE - scores, uncertainty, votes, explanations
# ==================================================================================================
# The matching engine: given a Moon or Mars site, rank every Earth patch.
#
# How a match score is built (the part you must be able to explain)
# -----------------------------------------------------------------
# For a target site T and each Earth patch E we measure three DISTANCES,
# one per axis (smaller = more alike):
#
#   terrain  : 1 - cosine similarity of the two neural fingerprints.
#              Computed separately for each of the 5 ensemble networks.
#   mobility : how far apart their slope/roughness statistics are
#              (each statistic first log-transformed and standardised).
#   thermal  : how far apart their temperature statistics are
#              (mean, day-night swing, max, min; standardised).
#   exposure : how much of the Earth patch is NOT bare ground (vegetation or
#              ice sheet, from MODIS NDVI + glacier outlines).  Moon and Mars
#              are 100 % bare, so a fully bare Earth patch has distance 0.
#
# Each axis distance is divided by a fixed "typical distance" (the median over
# random Earth-vs-Moon/Mars pairs) so that the three axes speak the same units.
# The mission preset then mixes them:  D = w_terrain*d_t + w_mob*d_m + w_th*d_h.
#
# The MATCH SCORE is an absolute similarity:  score = 100 * exp(-D).
#   100 = identical on every axis;  ~37 = as different as a typical random
#   Earth/Moon-or-Mars pair (D = 1 by construction of the "typical distance").
# We also report "top x %": the share of Earth candidates that are at least as
# close (a rank).  Because the terrain distance exists five times (once per
# network), we get five scores per candidate; we report their mean, their spread
# and range (the uncertainty bar), and how many of the five networks put the
# candidate in their own top-100 list (the vote).
#
# All arrays are numpy; torch is only needed to fingerprint a NEW location.
VOTE_TOP = 100          # a network "votes" for a candidate if it is in that network's top 100
MIN_SEP_KM = 300.0      # results closer than this to a better result are suppressed (no duplicates)
ICE_MAX = 0.8           # Earth patches more than 80 % ice-sheet are skipped by default


class Engine:
    def __init__(self, asset_dir: Path = ASSET_DIR):
        self.dir = Path(asset_dir)
        self.P = {}
        for b in BODIES:
            d = np.load(self.dir / f"patches_{b}.npz")
            self.P[b] = {k: d[k] for k in d.files}
        emb = np.load(self.dir / "embeddings.npz")
        self.E = {b: emb[b].astype(np.float32) for b in BODIES}       # (members, N, D)
        self.members = self.E["earth"].shape[0]
        names = self.dir / "names_earth.json"
        self.names = json.loads(names.read_text()) if names.exists() else None
        self._fit_scalers()

    # ------------------------------------------------------------------
    # Standardisation: pooled over all three bodies so no body is "home".
    # ------------------------------------------------------------------
    def _fit_scalers(self) -> None:
        self.mob_idx = [MORPH_FEATURES.index(f) for f in MOBILITY_FEATURES]
        allm = np.concatenate([to_comparable(self.P[b]["morph"]) for b in BODIES])
        allt = np.concatenate([self.P[b]["thermal"] for b in BODIES])
        self.m_mu, self.m_sd = allm.mean(0), allm.std(0) + 1e-6
        self.t_mu, self.t_sd = allt.mean(0), allt.std(0) + 1e-6
        for b in ("moon", "mars"):
            self.P[b]["barefrac"] = np.ones(len(self.P[b]["lat"]), np.float32)
        bf = self.P["earth"].get("barefrac")
        self.has_exposure = bf is not None and np.isfinite(bf).all()
        if not self.has_exposure:
            self.P["earth"]["barefrac"] = np.ones(len(self.P["earth"]["lat"]), np.float32)
        self.bare_sd = float(np.concatenate([self.P[b]["barefrac"] for b in BODIES]).std() + 1e-6)
        for b in BODIES:
            self.P[b]["mz"] = (to_comparable(self.P[b]["morph"]) - self.m_mu) / self.m_sd
            self.P[b]["tz"] = (self.P[b]["thermal"] - self.t_mu) / self.t_sd
        # typical distances per axis (median over random Earth x other-body pairs)
        rng = np.random.default_rng(SEED)
        e = rng.integers(0, len(self.P["earth"]["lat"]), 20000)
        o = {b: rng.integers(0, len(self.P[b]["lat"]), 10000) for b in ("moon", "mars")}
        dt, dm, dh = [], [], []
        for b in ("moon", "mars"):
            ee, oo = e[:10000], o[b]
            dt.append(1 - np.sum(self.E["earth"][0, ee] * self.E[b][0, oo], axis=1))
            dm.append(self._mob_dist(self.P["earth"]["mz"][ee], self.P[b]["mz"][oo]))
            dh.append(self._th_dist(self.P["earth"]["tz"][ee], self.P[b]["tz"][oo]))
        self.scale = {"terrain": float(np.median(np.concatenate(dt))),
                      "mobility": float(np.median(np.concatenate(dm))),
                      "thermal": float(np.median(np.concatenate(dh))),
                      # exposure: mean "not-bare" share of Earth land (a median could be 0)
                      "exposure": float(max(np.mean(1 - self.P["earth"]["barefrac"]), 1e-3))}

    def _mob_dist(self, a, b):
        d = a[..., self.mob_idx] - b[..., self.mob_idx]
        return np.sqrt((d ** 2).mean(-1))

    @staticmethod
    def _th_dist(a, b):
        return np.sqrt(((a - b) ** 2).mean(-1))

    # ------------------------------------------------------------------
    # Lazily loaded heavy pieces (only needed for new locations)
    # ------------------------------------------------------------------
    @cached_property
    def dem(self) -> dict:
        out = {}
        for b in BODIES:
            d = np.load(self.dir / f"dem_{b}.npz")
            out[b] = d["dem"]          # int16 metres; sampling converts to float (half the memory of float32)
        return out

    @cached_property
    def thermal_grid(self) -> dict:
        return {b: dict(np.load(self.dir / f"thermal_{b}.npz")) for b in BODIES}

    @cached_property
    def has_torch(self) -> bool:
        """PyTorch is optional for serving: without it, new locations use the nearest
        pre-computed patch fingerprint and saliency maps are skipped."""
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @cached_property
    def encoders(self) -> list:
        import torch
        out = []
        for k in range(self.members):
            m = TerrainEncoder()
            m.load_state_dict(torch.load(self.dir / f"encoder_{k}.pt", map_location="cpu", weights_only=True))
            m.eval()
            out.append(m)
        return out

    # ------------------------------------------------------------------
    # Profile of any location: patch, features, fingerprints
    # ------------------------------------------------------------------
    def profile(self, body: str, lat: float, lon: float) -> dict:
        patch = sample_patches(self.dem[body], [lat], [lon], RADIUS_M[body])
        morph = morphometry(patch)
        tg = self.thermal_grid[body]
        thermal = np.array([[interp_nodes(tg["data"][i], tg["lat"], tg["lon"], lat, lon)
                             for i in range(len(THERMAL_FEATURES))]], dtype=np.float32)
        if self.has_torch:
            emb = np.stack([embed(m, patch)[0] for m in self.encoders])        # (members, D)
        else:                                                                   # nearest stored fingerprint
            emb = self.E[body][:, self.nearest_patch(body, lat, lon)[0]]
        return {"body": body, "lat": float(lat), "lon": float(lon), "patch": patch[0],
                "morph": morph[0], "thermal": thermal[0], "emb": emb,
                "mz": ((to_comparable(morph) - self.m_mu) / self.m_sd)[0],
                "tz": ((thermal - self.t_mu) / self.t_sd)[0]}

    def patch_profile(self, body: str, idx: int) -> dict:
        """Profile of an existing database patch (no network evaluation needed)."""
        p = self.P[body]
        return {"body": body, "lat": float(p["lat"][idx]), "lon": float(p["lon"][idx]),
                "morph": p["morph"][idx], "thermal": p["thermal"][idx], "emb": self.E[body][:, idx],
                "mz": p["mz"][idx], "tz": p["tz"][idx], "idx": int(idx)}

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------
    def candidates(self, exclude_ice: bool = True) -> np.ndarray:
        e = self.P["earth"]
        keep = np.ones(len(e["lat"]), bool)
        if exclude_ice:
            keep &= e["icefrac"] <= ICE_MAX
        return np.where(keep)[0]

    def axis_distances(self, prof: dict, idx: np.ndarray, members: list[int] | None = None) -> dict:
        """Normalised distance from the target to Earth patches idx on each axis.
        terrain has shape (members, n); the others (n,)."""
        members = list(range(self.members)) if members is None else members
        e = self.P["earth"]
        terr = np.stack([1 - self.E["earth"][k, idx] @ prof["emb"][k] for k in members])
        return {"terrain": terr / self.scale["terrain"],
                "mobility": self._mob_dist(e["mz"][idx], prof["mz"]) / self.scale["mobility"],
                "thermal": self._th_dist(e["tz"][idx], prof["tz"]) / self.scale["thermal"],
                "exposure": np.abs(prof.get("bare", 1.0) - e["barefrac"][idx]) / self.scale["exposure"]}

    @staticmethod
    def combine(d: dict, weights: dict) -> np.ndarray:
        """Weighted sum of axis distances -> (members, n)."""
        total = weights.get("terrain", 0) * d["terrain"]
        for a in ("mobility", "thermal", "exposure"):
            total = total + weights.get(a, 0) * d[a][None]
        return total

    @staticmethod
    def percentile_score(dist: np.ndarray) -> np.ndarray:
        """Distance -> 0..100 score (100 = closest), row by row."""
        n = dist.shape[-1]
        ranks = np.argsort(np.argsort(dist, axis=-1), axis=-1)       # 0 = closest
        return 100.0 * (1.0 - ranks / max(n - 1, 1))

    def match(self, prof: dict, weights: dict, k: int = 10, exclude_ice: bool = True,
              min_sep_km: float = MIN_SEP_KM) -> dict:
        idx = self.candidates(exclude_ice)
        d = self.axis_distances(prof, idx)
        D = self.combine(d, weights) / max(sum(weights.values()), 1e-9)   # (members, n), weights sum to 1
        score_m = 100.0 * np.exp(-D)                                  # absolute similarity per network
        score, spread = score_m.mean(0), score_m.std(0)
        top_pct = 100.0 * (np.argsort(np.argsort(-score)) + 1) / len(idx)   # 0.01 = best 0.01 %
        # votes: in how many networks' top-VOTE_TOP is each candidate?
        top_sets = np.argpartition(D, VOTE_TOP, axis=1)[:, :VOTE_TOP]
        votes = np.zeros(len(idx), int)
        for row in top_sets:
            votes[row] += 1
        # greedy non-maximum suppression -> k well-separated results
        e = self.P["earth"]
        order = np.argsort(-score)
        chosen: list[int] = []
        for j in order:
            if len(chosen) >= k:
                break
            if chosen:
                gc = great_circle_km(e["lat"][idx[j]], e["lon"][idx[j]],
                                        e["lat"][idx[chosen]], e["lon"][idx[chosen]], RADIUS_M["earth"])
                if gc.min() < min_sep_km:
                    continue
            chosen.append(j)
        axis_scores = {a: self.percentile_score(d[a].mean(0) if a == "terrain" else d[a]) for a in AXES}
        results = []
        for rank, j in enumerate(chosen, 1):
            gi = int(idx[j])
            results.append({
                "rank": rank, "idx": gi, "lat": float(e["lat"][gi]), "lon": float(e["lon"][gi]),
                "score": round(float(score[j]), 2), "spread": round(float(spread[j]), 2),
                "score_min": round(float(score_m[:, j].min()), 2),
                "score_max": round(float(score_m[:, j].max()), 2),
                "top_pct": round(float(top_pct[j]), 3),
                "votes": int(votes[j]), "members": self.members,
                "axes": {a: round(float(axis_scores[a][j]), 1) for a in AXES},
                "terrain_cos": round(float(1 - d["terrain"][:, j].mean() * self.scale["terrain"]), 3),
                "place": self.place_name(gi),
                "known": self.nearest_known(e["lat"][gi], e["lon"][gi], prof["body"]),
                "icefrac": round(float(e["icefrac"][gi]), 2),
                "barefrac": round(float(e["barefrac"][gi]), 2),
            })
        # data for the animated search: score histogram and the 300 best candidates
        top = order[:300]
        hist, edges = np.histogram(score, bins=50, range=(0, 100))
        return {"results": results, "n_candidates": int(len(idx)),
                "hist": {"counts": hist.tolist(), "edges": edges.tolist()},
                "sweep": [[float(e["lat"][idx[j]]), float(e["lon"][idx[j]]), round(float(score[j]), 2)]
                          for j in top],
                "weights": weights}

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------
    def explain(self, prof: dict, earth_idx: int, saliency: bool = True) -> dict:
        """Why does Earth patch earth_idx match the target? Feature-by-feature
        comparison plus (optionally) occlusion saliency maps from the networks."""
        ep = self.patch_profile("earth", earth_idx)
        rows = []
        for j, name in enumerate(MORPH_FEATURES):
            dz = float(ep["mz"][j] - prof["mz"][j])
            rows.append({"feature": name, "group": "mobility" if name in MOBILITY_FEATURES else "terrain",
                         "target": float(prof["morph"][j]), "analog": float(ep["morph"][j]), "dz": round(dz, 2)})
        for j, name in enumerate(THERMAL_FEATURES):
            dz = float(ep["tz"][j] - prof["tz"][j])
            rows.append({"feature": name, "group": "thermal", "target": float(prof["thermal"][j]),
                         "analog": float(ep["thermal"][j]), "dz": round(dz, 2)})
        bare = float(self.P["earth"]["barefrac"][earth_idx])
        rows.append({"feature": "bare_fraction", "group": "exposure", "target": 1.0, "analog": bare,
                     "dz": round((bare - 1.0) / self.bare_sd, 2)})
        ranked = sorted(rows, key=lambda r: abs(r["dz"]))
        out = {"features": rows,
               "agree": [r["feature"] for r in ranked[:4]],
               "disagree": [r["feature"] for r in ranked[::-1][:3] if abs(r["dz"]) > 0.75],
               "summary": self._sentence(rows, prof["body"])}
        if saliency and self.has_torch:
            tp = prof.get("patch")
            if tp is None:
                tp = sample_patches(self.dem[prof["body"]], [prof["lat"]], [prof["lon"]], RADIUS_M[prof["body"]])[0]
            ap = sample_patches(self.dem["earth"], [ep["lat"]], [ep["lon"]], RADIUS_M["earth"])[0]
            st, sa = self.occlusion(tp, ap)
            out.update({"target_patch": np.round(tp - tp.mean()).tolist(),
                        "analog_patch": np.round(ap - ap.mean()).tolist(),
                        "target_saliency": np.round(st, 4).tolist(),
                        "analog_saliency": np.round(sa, 4).tolist()})
        return out

    def _sentence(self, rows: list[dict], body: str) -> str:
        """One plain-English sentence for non-experts."""
        by = {r["feature"]: r for r in rows}
        bits = []
        rel_t, rel_a = by["relief_m"]["target"], by["relief_m"]["analog"]
        bits.append(f"relief {rel_a:,.0f} m on Earth vs {rel_t:,.0f} m on the {body.title()}")
        s_t, s_a = by["slope_mean_deg"]["target"], by["slope_mean_deg"]["analog"]
        bits.append(f"mean slope {s_a:.1f} deg vs {s_t:.1f} deg")
        tm_t, tm_a = by["t_min_K"]["target"], by["t_min_K"]["analog"]
        bits.append(f"coldest surface {tm_a - 273.15:.0f} C vs {tm_t - 273.15:.0f} C")
        bits.append(f"{by['bare_fraction']['analog'] * 100:.0f}% bare ground")
        return "; ".join(bits) + "."

    def occlusion(self, target: np.ndarray, analog: np.ndarray, win: int = 8, stride: int = 4):
        """Occlusion saliency: hide one 8x8 window (~30 km) at a time, flattening it to the
        patch mean, and measure how much the fingerprint similarity DROPS.  Large drop =
        that part of the landscape was important for the match.  Averaged over the
        ensemble.  Returns two 32x32 maps (target, analog) scaled to 0..1."""
        import torch

        def maps(src: np.ndarray, other: np.ndarray) -> np.ndarray:
            pos = list(range(0, PATCH - win + 1, stride))
            batch = [src]
            for i in pos:
                for j in pos:
                    x = src.copy()
                    x[i:i + win, j:j + win] = src.mean()
                    batch.append(x)
            batch = np.stack(batch).astype(np.float32)
            heat = np.zeros((PATCH, PATCH), np.float32)
            cnt = np.zeros_like(heat)
            for m in self.encoders:
                z = embed(m, batch)
                zo = embed(m, other[None].astype(np.float32))[0]
                sims = z @ zo
                drop = sims[0] - sims[1:]
                n = 0
                for i in pos:
                    for j in pos:
                        heat[i:i + win, j:j + win] += max(drop[n], 0)
                        cnt[i:i + win, j:j + win] += 1
                        n += 1
            heat = heat / np.maximum(cnt, 1)
            return heat / max(heat.max(), 1e-9)

        with torch.no_grad():
            return maps(target, analog), maps(analog, target)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def place_name(self, earth_idx: int) -> str:
        if not self.names:
            return ""
        reg = self.names["region"][earth_idx]
        cty = self.names["country"][earth_idx]
        return ", ".join(x for x in (reg, cty) if x)

    def nearest_known(self, lat: float, lon: float, body: str) -> dict | None:
        best = None
        for a in KNOWN_ANALOGS:
            d = float(great_circle_km(lat, lon, a[1], a[2], RADIUS_M["earth"]))
            if best is None or d < best["km"]:
                best = {"name": a[0], "km": round(d), "for_body": body in a[3]}
        return best

    def nearest_patch(self, body: str, lat: float, lon: float) -> tuple[int, float]:
        p = self.P[body]
        d = great_circle_km(lat, lon, p["lat"], p["lon"], RADIUS_M[body])
        i = int(np.argmin(d))
        return i, float(d[i])


# ==================================================================================================
# 12. REPORT - Markdown + PDF export
# ==================================================================================================
# One-click report: a ranked list of Earth analogs for one Moon/Mars site,
# written as Markdown (with PNG figures) and as a multi-page PDF.
#
# Only matplotlib is used, so the report works fully offline.
REPORT_FEATURE_NAMES = {
    "relief_m": "Relief (m)", "elev_std_m": "Height spread (m)", "slope_mean_deg": "Mean slope (deg)",
    "slope_p90_deg": "Steep-part slope (deg)", "rough_small_m": "Roughness 11 km (m)", "rough_mid_m": "Roughness 33 km (m)",
    "curvature": "Curvature", "hypsometric": "Plateau vs basin", "elev_skew": "Peaks vs pits", "pit_density": "Pit density",
    "pit_depth_m": "Pit depth (m)", "anisotropy": "Ridge alignment", "t_mean_K": "Mean temp (K)",
    "t_diurnal_K": "Day-night swing (K)", "t_max_K": "Max temp (K)", "t_min_K": "Min temp (K)",
    "bare_fraction": "Bare ground (fraction)",
}
DARK = "#0b0f17"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40] or "site"


def _world_map(eng, ax, results, body: str) -> None:
    """Shaded-relief Earth with the analogs numbered and known analogs as stars."""
    z = eng.dem["earth"][::8, ::8]
    land = z > 0
    gy, gx = np.gradient(z)
    shade = np.clip(0.6 + 0.004 * (gx - gy), 0, 1)
    rgb = np.where(land[..., None], np.stack([0.35 + 0.3 * shade, 0.38 + 0.3 * shade, 0.33 + 0.25 * shade], -1),
                   np.array([0.08, 0.12, 0.2]))
    ax.imshow(rgb, extent=[-180, 180, -90, 90], origin="upper")
    for a in KNOWN_ANALOGS:
        if body in a[3]:
            ax.plot(a[2], a[1], marker="*", ms=7, color="#f6e05e", mec="k", mew=0.4)
    for r in results:
        ax.plot(r["lon"], r["lat"], "o", ms=9, color="#4fd1c5", mec="k")
        ax.text(r["lon"] + 2.5, r["lat"] + 2, str(r["rank"]), color="white", fontsize=8, weight="bold")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks([])
    ax.set_yticks([])


def _patch_figure(eng, prof, r, ex, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])
    for k, (key, sal, title) in enumerate([("target_patch", "target_saliency", f"Target ({prof['body']})"),
                                           ("analog_patch", "analog_saliency", f"#{r['rank']} Earth analog")]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(np.array(ex[key]), cmap="terrain")
        ax.contour(np.array(ex[sal]), levels=[0.5, 0.75], colors=["#4fd1c5", "white"], linewidths=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    feats = ex["features"]
    dz = [f["dz"] for f in feats]
    cols = ["#38a169" if abs(d) < 0.5 else "#dd6b20" if abs(d) < 1 else "#e53e3e" for d in dz]
    ax.barh([REPORT_FEATURE_NAMES.get(f["feature"], f["feature"]) for f in feats], dz, color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-0.5, 0.5, color="#38a169", alpha=0.08)
    ax.set_xlim(-3.5, 3.5)
    ax.invert_yaxis()
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Earth minus target (standard deviations)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(eng, prof: dict, name: str, weights: dict, preset: str, exclude_ice: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    now = dt.datetime.now()
    out = REPORT_DIR / f"{now:%Y%m%d_%H%M%S}_{_slug(name)}"
    out.mkdir(parents=True, exist_ok=True)
    body = prof["body"]
    res = eng.match(prof, weights, k=10, exclude_ice=exclude_ice)
    results = res["results"]
    explains = [eng.explain(prof, r["idx"], saliency=True) for r in results[:3]]
    val_path = ASSET_DIR / "validation.json"
    val = json.loads(val_path.read_text()) if val_path.exists() else None

    # ---- figures ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.6))
    _world_map(eng, ax, results, body)
    ax.set_title(f"Top Earth analogs for {name}  (teal = ranked analogs, yellow stars = known {body} analogs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_map.png", dpi=130)
    plt.close(fig)
    for r, ex in zip(results, explains):
        _patch_figure(eng, prof, r, ex, out / f"fig_match_{r['rank']}.png")

    # ---- markdown ----------------------------------------------------
    w_txt = ", ".join(f"{a} {weights.get(a, 0) * 100:.0f}%" for a in AXES)
    L = [f"# THEIA report - Earth analogs for {name}", "",
         f"*Generated {now:%Y-%m-%d %H:%M} with THEIA {__version__} (Terrestrial Habitat Evaluation and Interplanetary Analogue), matching model AnalogNet.*", "",
         "## Target", "",
         f"- **Site:** {name} ({body.title()}, {prof['lat']:.3f} deg, {prof['lon']:.3f} deg)",
         f"- **Mission preset:** {PRESETS.get(preset, {}).get('label', preset)}  -  weights: {w_txt}",
         f"- **Candidates compared:** {res['n_candidates']:,} Earth land patches of {FOOTPRINT_KM:.0f} km"
         f" ({'ice sheets excluded' if exclude_ice else 'ice sheets included'})",
         f"- **Target patch:** relief {prof['morph'][0]:,.0f} m, mean slope {prof['morph'][2]:.2f} deg, "
         f"surface temperature {prof['thermal'][3]:.0f}-{prof['thermal'][2]:.0f} K", "",
         "![map](fig_map.png)", "",
         "## Ranked analogs", "",
         "Score = 100 x exp(-D), D = weighted distance (100 = identical, ~37 = typical random Earth place). "
         f"Range = min-max over the {eng.members}-network ensemble. Top % = share of Earth land at least as close. "
         "Votes = networks ranking the site in their top 100. Axis columns = per-axis percentiles.", "",
         "| # | Place | Lat, Lon | Score | Range | Top % | Votes | Terrain | Mobility | Thermal | Bare | Nearest known analog |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        kn = f"{r['known']['name']} ({r['known']['km']} km)" if r["known"] else ""
        L.append(f"| {r['rank']} | {r['place'] or '-'} | {r['lat']:.2f}, {r['lon']:.2f} | {r['score']:.1f} | "
                 f"{r['score_min']:.1f}-{r['score_max']:.1f} | {r['top_pct']:.3f} | {r['votes']}/{r['members']} | {r['axes']['terrain']:.0f} | "
                 f"{r['axes']['mobility']:.0f} | {r['axes']['thermal']:.0f} | {r['axes'].get('exposure', 0):.0f} | {kn} |")
    L += ["", "## Why the top 3 match", ""]
    for r, ex in zip(results, explains):
        agree = ", ".join(REPORT_FEATURE_NAMES.get(f, f) for f in ex["agree"])
        dis = ", ".join(REPORT_FEATURE_NAMES.get(f, f) for f in ex["disagree"]) or "none large"
        L += [f"### #{r['rank']} {r['place'] or ''} ({r['lat']:.2f}, {r['lon']:.2f})", "",
              f"{ex['summary']}  Most alike: {agree}. Most different: {dis}.", "",
              f"![match {r['rank']}](fig_match_{r['rank']}.png)", "",
              "Left: relative elevation of the two patches, outlines = regions the networks relied on most "
              "(occlusion saliency). Right: feature differences in standard deviations.", ""]
    if val and body in val["bodies"]:
        B = val["bodies"][body]
        L += ["## How much to trust this (validation)", "",
              f"Known {body} analogs ({len(B['known_sites'])} sites) were never used for training. "
              "Mean site score = percentile of the best patch within 120 km of each known site.", "",
              "| Method | Mean site score | Sites in top 10 % | AUC | p (vs random) |", "|---|---|---|---|---|"]
        names = {m["key"]: m["name"] for m in val["methods"]}
        for k, m in B["methods"].items():
            L.append(f"| {names.get(k, k)} | {m['mean_site_score']} | {m['recall_top10'] * 100:.0f}% | {m['auc']} | {m['p_value']} |")
        L.append(f"| Random land sites | {B['methods']['analognet']['random_mean']} | "
                 f"{B['methods']['analognet']['random_recall_top10'] * 100:.0f}% | 0.5 | - |")
        if B.get("calibration"):
            L += ["", f"Ensemble calibration (leave-one-network-out): expected calibration error "
                      f"{B['calibration']['ece']}."]
        L.append("")
    L += ["## Method in brief", "",
          f"Elevation patches ({PATCH}x{PATCH} px at {PIXEL_M / 1000:.1f} km) from LOLA (Moon), MOLA (Mars) and ETOPO 2022 "
          f"(Earth) are encoded by an ensemble of {eng.members} convolutional networks trained with self-supervised contrastive "
          "learning on all three bodies. Matching combines the neural terrain distance with slope/roughness (mobility), "
          "surface temperature (thermal) and bare-ground (exposure) distances, weighted by mission purpose.", "",
          "**Limitations:** regional scale only (no rover-scale rocks), no mineralogy layer yet, Mars temperature from a "
          "climate model, Earth temperature at MODIS overpass times. New candidates are hypotheses for field validation.", "",
          "## Data sources", ""]
    for s in SOURCES:
        L.append(f"- **{s['body']} - {s['variable']}:** {s['product']}. {s['agency']}. Licence: {s['license']}. {s['cite']}")
    L.append("")
    md = out / f"THEIA_{_slug(name)}.md"
    md.write_text("\n".join(L), encoding="utf-8")

    # ---- PDF ----------------------------------------------------------
    with PdfPages(md.with_suffix(".pdf")) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
        fig.text(0.06, 0.96, f"THEIA - Earth analogs for {name}", fontsize=16, weight="bold")
        fig.text(0.06, 0.935, f"{body.title()}  {prof['lat']:.3f}, {prof['lon']:.3f}   |   preset: "
                 f"{PRESETS.get(preset, {}).get('label', preset)}   |   {now:%Y-%m-%d %H:%M}", fontsize=9)
        ax = fig.add_axes([0.06, 0.6, 0.88, 0.32])
        _world_map(eng, ax, results, body)
        rows = [[str(r["rank"]), (r["place"] or "-")[:34], f"{r['lat']:.1f}, {r['lon']:.1f}", f"{r['score']:.1f}",
                 f"{r['score_min']:.0f}-{r['score_max']:.0f}", f"{r['votes']}/{r['members']}",
                 f"{r['axes']['terrain']:.0f}/{r['axes']['mobility']:.0f}/{r['axes']['thermal']:.0f}/{r['axes'].get('exposure', 0):.0f}"]
                for r in results]
        tax = fig.add_axes([0.06, 0.2, 0.88, 0.36])
        tax.axis("off")
        tb = tax.table(cellText=rows, colLabels=["#", "Place", "Lat, Lon", "Score", "Range", "Votes", "T/M/Th/B"],
                       loc="upper center", colWidths=[0.04, 0.4, 0.13, 0.08, 0.09, 0.08, 0.16])
        tb.auto_set_font_size(False)
        tb.set_fontsize(7.5)
        tb.scale(1, 1.4)
        fig.text(0.06, 0.16, "Score: 100 x exp(-distance), 100 = identical, ~37 = typical random place. Range: across the network ensemble.\n"
                 "Votes: networks ranking the site in their top 100. T/M/Th/B: per-axis percentiles "
                 "(terrain / mobility / thermal / bare ground).", fontsize=7.5, color="#444")
        pdf.savefig(fig)
        plt.close(fig)
        for r in results[:3]:
            img = plt.imread(out / f"fig_match_{r['rank']}.png")
            ex = explains[r["rank"] - 1]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.95, f"#{r['rank']} {r['place'] or ''}  ({r['lat']:.2f}, {r['lon']:.2f})", fontsize=13, weight="bold")
            fig.text(0.06, 0.92, f"Score {r['score']:.1f} (range {r['score_min']:.1f}-{r['score_max']:.1f}), "
                     f"{r['votes']}/{r['members']} networks agree", fontsize=9)
            fig.text(0.06, 0.88, _wrap(ex["summary"], 110), fontsize=9, va="top")
            ax = fig.add_axes([0.03, 0.45, 0.94, 0.38])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=(8.27, 11.69))
        text = "\n".join(l for l in L[L.index("## Method in brief"):] if not l.startswith("!["))
        fig.text(0.06, 0.96, _wrap(text.replace("**", "").replace("## ", ""), 120), fontsize=7, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    return md


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join("\n".join(textwrap.wrap(p, width)) if p else "" for p in text.split("\n"))


# ==================================================================================================
# 13. 3-D GLOBE - CesiumJS as a Streamlit component
# ==================================================================================================
# The 3-D globe for the Streamlit web app, as a Streamlit custom component.
#
# Streamlit serves the folder `ui/` (CesiumJS + a small page) from the app's own
# address, so the globe's web workers load same-origin and the globe textures come from
# Streamlit's static file server (/app/static/textures/...).  Clicks travel back to Python:
# clicking a result pin returns {"rank": n}; with pick=True, clicking the Moon/Mars returns
# {"lat": .., "lon": ..}.
_globe = components.declare_component("theia_globe", path=str(UI_DIR))

TREK = {
    "moon": "https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0/default/default028mm/{TileMatrix}/{TileRow}/{TileCol}.jpg",
    "mars": "https://trek.nasa.gov/tiles/Mars/EQ/Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{TileMatrix}/{TileRow}/{TileCol}.jpg",
}


def globe(body: str, target: dict | None = None, results: list[dict] | None = None, selected: int | None = None,
          known: list[dict] | None = None, heat: list | None = None, s2_url: str | None = None,
          fly: tuple[float, float, float] | None = None, online: bool = True, pick: bool = False,
          height: int = 620, style: str = "holo", key: str = "globe"):
    """Draw the globe; returns the last click ({"rank"} or {"lat", "lon"}) or None."""
    cfg = {"body": body, "target": target, "results": results or [], "selected": selected, "known": known or [],
           "heat": heat or [], "s2": s2_url, "trek": TREK.get(body), "fly": list(fly) if fly else None,
           "online": bool(online), "pick": bool(pick), "height": height, "style": style}
    return _globe(cfg=cfg, key=key, default=None)


# ==================================================================================================
# 14. THE WEBSITE - pages: Explore, Watch the AI think, Validation, Data & method
# ==================================================================================================
# ------------------------------------------------------------------ design tokens
CYAN, LAV, EARTH, MOON, MARS = "#5fd0ff", "#b8acf0", "#5aa9e6", "#c9ccd6", "#e07a4f"
INK, INK2, INK3, LINE, ABLATE = "#eaf3ff", "#9fb3cc", "#5f7591", "rgba(95,208,255,0.12)", "#3b4c66"
GOOD, WARN, BAD = "#6fe0b0", "#f2c46b", "#ff8a8a"
BODY_COLOR = {"earth": EARTH, "moon": MOON, "mars": MARS}
FEATURE_NAMES = {
    "relief_m": "Relief (m)", "elev_std_m": "Height spread (m)", "slope_mean_deg": "Mean slope (deg)",
    "slope_p90_deg": "Steep-part slope (deg)", "rough_small_m": "Roughness, 11 km (m)", "rough_mid_m": "Roughness, 33 km (m)",
    "curvature": "Curvature", "hypsometric": "Plateau vs basin", "elev_skew": "Peaks vs pits", "pit_density": "Pit density",
    "pit_depth_m": "Pit / crater depth (m)", "anisotropy": "Ridge alignment", "t_mean_K": "Mean temperature (K)",
    "t_diurnal_K": "Day-night swing (K)", "t_max_K": "Max temperature (K)", "t_min_K": "Min temperature (K)",
    "bare_fraction": "Bare ground (fraction)"}
AXIS_NAMES = {"terrain": "Terrain (neural)", "mobility": "Rover mobility", "thermal": "Thermal", "exposure": "Bare ground"}
PLOT_CFG = {"displayModeBar": False}



def plot_layout(**kw) -> dict:
    base = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=12),
                margin=dict(l=50, r=12, t=10, b=40), xaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
                yaxis=dict(gridcolor=LINE, zerolinecolor=LINE), legend=dict(orientation="h", y=1.08))
    for k, v in kw.items():
        base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def fmt_lat(v): return f"{abs(v):.2f}° {'N' if v >= 0 else 'S'}"
def fmt_lon(v): return f"{abs(v):.2f}° {'E' if v >= 0 else 'W'}"
def fmt_pct(p): return f"{p:.3f}%" if p < 0.1 else f"{p:.2f}%" if p < 1 else f"{p:.1f}%"
def short_place(r): return (r.get("place") or "").split(",")[0] or f"{fmt_lat(r['lat'])}, {fmt_lon(r['lon'])}"
def country(r): return ", ".join(x.strip() for x in (r.get("place") or "").split(",")[1:])


def glass(key: str, **kw):
    """A frosted-glass card (styled by the CSS rule on keys that start with 'glass')."""
    return st.container(key=f"glass{key}", **kw)


def dot(color=CYAN):
    return f"<span class='dot' style='background:{color};box-shadow:0 0 10px {color}'></span>"


# ------------------------------------------------------------------ cached science
@st.cache_resource(show_spinner="Loading the trained ensemble and the open data...")
def engine():
    return Engine()


@st.cache_data(show_spinner=False)
def load_json(name: str):
    p = ASSET_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource(show_spinner=False, max_entries=64)
def profile(body: str, lat: float, lon: float):
    return engine().profile(body, lat, lon)


@st.cache_data(show_spinner="Comparing the site with every Earth land patch...", max_entries=64)
def run_match(body: str, lat: float, lon: float, weights: tuple, exclude_ice: bool):
    return engine().match(profile(body, lat, lon), dict(weights), k=10, exclude_ice=exclude_ice)


@st.cache_data(show_spinner="Explaining the match with the 5 networks...", max_entries=128)
def run_explain(body: str, lat: float, lon: float, earth_idx: int):
    return engine().explain(profile(body, lat, lon), earth_idx, saliency=True)


@st.cache_data(show_spinner=False, ttl=3600)
def s2_mosaic():
    """Register a Sentinel-2 mosaic with the Microsoft Planetary Computer data API (tiles rendered by
    the server, nothing stored).  Returns a tile URL template or None when unreachable."""
    import urllib.request
    body = {"filter-lang": "cql2-json", "filter": {"op": "and", "args": [
        {"op": "=", "args": [{"property": "collection"}, "sentinel-2-l2a"]},
        {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 10]},
        {"op": "anyinteracts", "args": [{"property": "datetime"}, {"interval": ["2023-01-01T00:00:00Z", "2024-12-31T23:59:59Z"]}]}]},
        "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}]}
    try:
        req = urllib.request.Request("https://planetarycomputer.microsoft.com/api/data/v1/mosaic/register",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=6) as r:
            sid = json.loads(r.read())["id"]
    except Exception:
        return None
    return (f"https://planetarycomputer.microsoft.com/api/data/v1/mosaic/{sid}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x"
            "?collection=sentinel-2-l2a&assets=visual&asset_bidx=visual%7C1%2C2%2C3&nodata=0&format=png")


@st.cache_data(show_spinner=False)
def heat_points(key: str):
    m = np.load(ASSET_DIR / "analog_maps.npz")
    if f"{key}_score" not in m.files:
        return []
    e = engine().P["earth"]
    cand = m["cand"]
    sc = m[f"{key}_score"].astype(np.float32)
    keep = sc >= 60
    return np.round(np.column_stack([e["lat"][cand][keep], e["lon"][cand][keep], sc[keep]]), 2)[::2].tolist()


def patch_figure(z, sal):
    fig = go.Figure([go.Heatmap(z=np.array(z), colorscale=[[0, "#06121f"], [.35, "#12436e"], [.7, "#3aa6e0"], [1, "#d8f4ff"]],
                                showscale=False, hovertemplate="%{z:.0f} m<extra></extra>")])
    if sal is not None:
        fig.add_trace(go.Contour(z=np.array(sal), showscale=False, contours=dict(coloring="lines", start=0.5, end=1, size=0.25),
                                 line=dict(width=2), colorscale=[[0, LAV], [1, "#ffffff"]], hoverinfo="skip"))
    fig.update_layout(**plot_layout(margin=dict(l=2, r=2, t=2, b=2), height=260,
                                    xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed", scaleanchor="x")))
    return fig


def is_phone() -> bool:
    """Phone or small tablet, from the browser's user agent (used to size the 3-D globe)."""
    try:
        ua = st.context.headers.get("User-Agent", "")
    except Exception:
        return False
    return bool(re.search(r"Mobi|Android|iPhone|iPod", ua))


# ================================================================== EXPLORE
def page_explore():
    targets = targets_as_dicts()
    with st.sidebar:
        st.subheader("Target site")
        body = st.segmented_control("World", ["Moon", "Mars"], default="Moon", key="body", label_visibility="collapsed") or "Moon"
        b = body.lower()
        options = [t for t in targets if t["body"] == b]
        PICK, CUSTOM = "Pick on the globe", "Custom coordinates"
        choice = st.selectbox("Site", [t["name"] for t in options] + [PICK, CUSTOM], key=f"site_{b}")
        picking = choice == PICK
        if picking:
            pk = st.session_state.get(f"pick_{b}")
            target = ({"id": "picked", "name": f"Site at {fmt_lat(pk[0])}, {fmt_lon(pk[1])}", "body": b, "lat": pk[0], "lon": pk[1],
                       "note": "Picked on the globe. Click again to move it."} if pk else
                      dict(options[0], note="Click the globe to choose your own site (showing the first site until then)."))
        elif choice == CUSTOM:
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", -90.0, 90.0, -85.0 if b == "moon" else 20.0, 0.5, key=f"lat_{b}")
            lon = c2.number_input("Longitude", -180.0, 180.0, 30.0 if b == "moon" else 70.0, 0.5, key=f"lon_{b}")
            target = {"id": "custom", "name": f"Site at {fmt_lat(lat)}, {fmt_lon(lon)}", "body": b, "lat": lat, "lon": lon,
                      "note": "Your own coordinates."}
        else:
            target = next(t for t in options if t["name"] == choice)

        st.subheader("Mission purpose")
        presets = PRESETS
        pkey = st.radio("Purpose", list(presets), format_func=lambda k: presets[k]["label"], key="preset", label_visibility="collapsed")
        st.caption(presets[pkey]["why"])
        with st.expander("Adjust axis weights"):
            raw = {a: st.slider(AXIS_NAMES[a], 0, 100, int(round(presets[pkey]["weights"].get(a, 0) * 100)), 5, key=f"w_{pkey}_{a}")
                   for a in AXES}
            st.caption(" ".join(PLANNED_PRESETS.values()))
        tot = sum(raw.values()) or 1
        weights = {a: raw[a] / tot for a in AXES}
        custom = raw != {a: int(round(presets[pkey]["weights"].get(a, 0) * 100)) for a in AXES}
        exclude_ice = st.checkbox("Skip Earth's ice sheets", True)
        online = st.checkbox("Online detail in satellite view", True,
                             help="NASA Trek mosaics for Moon/Mars and Sentinel-2 via Microsoft Planetary Computer when zoomed in on Earth.")

    res = run_match(b, float(target["lat"]), float(target["lon"]), tuple(sorted(weights.items())), exclude_ice)
    results = res["results"]
    if "sel_pending" in st.session_state:                     # a result pin was clicked on the globe
        st.session_state["sel_rank"] = st.session_state.pop("sel_pending")
    try:                                                       # the widget can hand back text after the result list changes
        sel = int(st.session_state.get("sel_rank", 1))
    except (TypeError, ValueError):
        sel = 1
    if not 1 <= sel <= len(results):
        sel = st.session_state["sel_rank"] = 1
    r = results[sel - 1]

    # ---------------- hero
    st.markdown(f"<div class='hero-title'>Earth analogs for <span class='place'>{target['name']}</span></div>"
                f"<div class='hero-sub'>Best 10 of {res['n_candidates']:,} Earth land patches · "
                f"{'Custom weights' if custom else presets[pkey]['label']}</div>", unsafe_allow_html=True)
    phone = is_phone()
    gh = 420 if phone else 600

    # ---------------- globe + side cards
    left, right = st.columns([1.75, 1], gap="medium")
    with left:
        t1, t2, t3 = st.columns([1, 1, 1.3], vertical_alignment="bottom")
        style = t1.segmented_control("Globe style", ["Hologram", "Satellite"], default="Hologram", key="gstyle") or "Hologram"
        if picking:
            st.session_state["view"] = "Target"
        view = t2.segmented_control("Globe shows", ["Analogs", "Target"], default="Analogs", key="view") or "Analogs"
        t3.markdown("<div class='toolbar-hint'>Drag to rotate · scroll to zoom · click a pin</div>", unsafe_allow_html=True)
        gstyle = "satellite" if style == "Satellite" else "holo"
        with glass("globe"):
            if view == "Target":
                click = globe(b, target={"name": target["name"], "lat": target["lat"], "lon": target["lon"]},
                              fly=(target["lat"], target["lon"], 5.2e6 if b == "moon" else 1.0e7), online=online, pick=picking,
                              style=gstyle, height=gh, key=f"globe_{b}")
            else:
                hk = st.session_state.get("heatkey")
                click = globe("earth", results=[{"rank": x["rank"], "name": short_place(x), "lat": x["lat"], "lon": x["lon"]} for x in results],
                              selected=sel, known=analogs_as_dicts(), heat=heat_points(hk) if hk else None,
                              s2_url=s2_mosaic() if online and gstyle == "satellite" else None,
                              fly=(r["lat"], r["lon"], 1.2e7), online=online, style=gstyle, height=gh, key="globe_earth")
        if click and click.get("t") != st.session_state.get("last_click"):
            st.session_state["last_click"] = click.get("t")
            if "rank" in click:
                st.session_state["sel_pending"] = int(click["rank"])
                st.rerun()
            elif "lat" in click and picking:
                st.session_state[f"pick_{b}"] = (round(click["lat"], 3), round(click["lon"], 3))
                st.rerun()
        g1, g2 = st.columns([1, 2])
        with g1:
            heat_on = st.toggle("Analog heat map", False, key="heat", disabled=view == "Target")
        with g2:
            if heat_on:
                st.session_state["heatkey"] = st.selectbox(
                    "Colour Earth by", ["mars", "moon", "mars_landform", "moon_landform"], label_visibility="collapsed",
                    format_func=lambda k: {"mars": "Mars-like (balanced)", "moon": "Moon-like (balanced)",
                                           "mars_landform": "Mars-like (shape only)", "moon_landform": "Moon-like (shape only)"}[k])
            else:
                st.session_state["heatkey"] = None
                st.caption("Click a pin to inspect a result. Drag to rotate, scroll to zoom.")

    with right:
        with glass("target"):
            st.markdown(f"<div class='lbl'><span>{dot()}Target site</span><span class='chip {b}'>{body}</span></div>"
                        f"<div style='font-size:1.35rem;margin-top:6px' class='place'>{target['name']}</div>"
                        f"<div class='sub'>{fmt_lat(target['lat'])}, {fmt_lon(target['lon'])}</div>"
                        f"<div class='sub' style='margin-top:6px;color:#9fb3cc'>{target.get('note', '')}</div>", unsafe_allow_html=True)
        with glass("best"):
            votes = "".join(f"<i class='{'on' if k < r['votes'] else ''}'></i>" for k in range(r["members"]))
            axes = "".join(f"<span>{AXIS_NAMES[a]}</span><span class='bar'><span style='width:{r['axes'][a]}%'></span></span>"
                           f"<span class='v'>{r['axes'][a]:.0f}</span>" for a in AXES)
            st.markdown(f"<div class='lbl'><span>{dot(LAV)}Selected analog · #{r['rank']}</span><span class='votes'>{votes}</span></div>"
                        f"<div style='display:flex;justify-content:space-between;align-items:flex-end;margin-top:4px'>"
                        f"<div><div class='place' style='font-size:1.45rem'>{short_place(r)}</div><div class='sub'>{country(r)}</div></div>"
                        f"<div class='big'>{r['score']:.1f}<small>/ 100</small></div></div>"
                        f"<div class='sub' style='text-align:right'>range {r['score_min']:.1f}–{r['score_max']:.1f} · top {fmt_pct(r['top_pct'])}</div>"
                        f"<div class='axis'>{axes}</div>", unsafe_allow_html=True)
        with glass("list", height=330):
            rows = []
            for x in results:
                vts = "".join(f"<i class='{'on' if k < x['votes'] else ''}'></i>" for k in range(x["members"]))
                kn = (f"<div class='known'>◇ {x['known']['km']} km from {x['known']['name'].split(',')[0]}, a known analog</div>"
                      if x["known"] and x["known"]["km"] < 250 else "")
                rows.append(f"""<div class="res-row {'sel' if x['rank'] == sel else ''}"><div class="rk">{x['rank']}</div>
                  <div class="pl"><span class="place">{short_place(x)}</span><small>{country(x)}</small>
                  <div style="display:flex;align-items:center"><div class="ubar" style="flex:1"><div class="rng" style="left:{x['score_min']}%;width:{max(x['score_max'] - x['score_min'], 1.5)}%"></div>
                  <div class="mid" style="left:{x['score']}%"></div></div><span class="votes">{vts}</span></div>{kn}</div>
                  <div class="sc">{x['score']:.1f}<small>top {fmt_pct(x['top_pct'])}</small></div></div>""")
            st.markdown(f"<div class='lbl' style='margin-bottom:6px'><span>{dot()}Ranked analogs</span>"
                        "<span class='sub'>bar = range across 5 networks</span></div>" + "".join(rows), unsafe_allow_html=True)
        st.selectbox("Inspect result", [x["rank"] for x in results], index=sel - 1, key="sel_rank",
                     format_func=lambda k: f"{k}. {short_place(results[k - 1])}")

    # ---------------- stat cards
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    with s1, glass("stat1"):
        h, e = res["hist"]["counts"], res["hist"]["edges"]
        cut = results[-1]["score"]
        st.markdown(f"<div class='lbl'><span>Earth patches compared</span></div><div class='big'>{res['n_candidates']:,}</div>", unsafe_allow_html=True)
        fig = go.Figure([go.Bar(x=[(e[i] + e[i + 1]) / 2 for i in range(len(h))], y=h,
                                marker_color=[CYAN if e[i + 1] >= cut else "#1f3550" for i in range(len(h))])])
        fig.update_layout(**plot_layout(height=96, margin=dict(l=0, r=0, t=4, b=0), xaxis=dict(visible=False, range=[0, 100]),
                                        yaxis=dict(visible=False, type="log"), bargap=0.15))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="mini_hist")
        st.markdown("<div class='sub'>score distribution · cyan = the best 10</div>", unsafe_allow_html=True)
    with s2, glass("stat2"):
        agree = sum(1 for x in results if x["votes"] >= x["members"] - 1)
        st.markdown(f"<div class='lbl'><span>Results the networks agree on</span></div><div class='big'>{agree}<small>of {len(results)}</small></div>"
                    f"<div class='sub' style='margin-top:10px'>At least {results[0]['members'] - 1} of {results[0]['members']} networks rank it in "
                    f"their top 100. In tests, a 4-of-4 vote is confirmed by an unseen network 91% of the time.</div>", unsafe_allow_html=True)
    with s3, glass("stat3"):
        cnt = Counter((country(x) or "Unnamed region").replace("People's Republic of China", "China")
                      .replace("United States of America", "USA") for x in results).most_common(4)
        bars = "".join(f"<span>{c}</span><span class='bar'><span style='width:{n / len(results) * 100}%'></span></span><span class='v'>{n}</span>"
                       for c, n in cnt)
        st.markdown(f"<div class='lbl'><span>Top analogs by country</span></div><div class='axis' style='grid-template-columns:1.4fr 2fr 24px'>{bars}</div>",
                    unsafe_allow_html=True)
    with s4, glass("stat4"):
        known_hits = sum(1 for x in results if x["known"] and x["known"]["km"] < 250 and x["known"]["for_body"])
        st.markdown(f"<div class='lbl'><span>Near a known {body} analog</span></div><div class='big'>{known_hits}<small>of {len(results)}</small></div>"
                    f"<div class='sub' style='margin-top:10px'>Within 250 km of a site experts already use for {body} analog work. "
                    f"The others are candidates for field checks.</div>", unsafe_allow_html=True)

    # ---------------- why this matches
    ex = run_explain(b, float(target["lat"]), float(target["lon"]), int(r["idx"]))
    st.markdown(f"<div class='hero-title' style='font-size:1.6rem;margin-top:1.4rem'>Why <span class='place'>{short_place(r)}</span> matches</div>",
                unsafe_allow_html=True)
    conf = "high" if r["votes"] >= r["members"] - 1 and r["spread"] < 3 else "medium" if r["votes"] >= 3 else "low"
    agree_f = ", ".join(FEATURE_NAMES.get(f, f).lower() for f in ex["agree"])
    dis = ", ".join(FEATURE_NAMES.get(f, f).lower() for f in ex["disagree"]) or "nothing large"
    st.markdown(f"<div class='hero-sub'>{ex['summary']} {r['votes']} of {r['members']} networks agree: {conf} confidence. "
                f"Closest on {agree_f}; furthest apart on {dis}.</div>", unsafe_allow_html=True)
    w1, w2, w3 = st.columns([1.25, 1.35, 0.8], gap="medium")
    with w1, glass("feat"):
        st.markdown(f"<div class='lbl'><span>{dot()}Feature by feature</span><span class='sub'>Earth minus target, in standard deviations</span></div>",
                    unsafe_allow_html=True)
        feats = list(reversed(ex["features"]))
        fig = go.Figure([go.Bar(orientation="h", x=[f["dz"] for f in feats], y=[FEATURE_NAMES.get(f["feature"], f["feature"]) for f in feats],
                                marker_color=[GOOD if abs(f["dz"]) < .5 else WARN if abs(f["dz"]) < 1 else BAD for f in feats],
                                customdata=[[f"{f['target']:.2f}", f"{f['analog']:.2f}"] for f in feats],
                                hovertemplate="%{y}<br>target %{customdata[0]} · Earth %{customdata[1]}<br>%{x:.2f} σ<extra></extra>")])
        fig.add_vrect(x0=-.5, x1=.5, fillcolor="rgba(111,224,176,.07)", line_width=0)
        fig.update_layout(**plot_layout(height=420, margin=dict(l=150, r=10, t=6, b=30), xaxis=dict(range=[-3.5, 3.5])))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG, key="feat_chart")
    with w2, glass("sal"):
        st.markdown(f"<div class='lbl'><span>{dot(LAV)}What the networks looked at</span><span class='sub'>outline = terrain that mattered most</span></div>",
                    unsafe_allow_html=True)
        if "target_saliency" in ex:
            p1, p2 = st.columns(2)
            p1.plotly_chart(patch_figure(ex["target_patch"], ex["target_saliency"]), width="stretch", config=PLOT_CFG, key="pt")
            p1.markdown(f"<div style='text-align:center' class='place'>{target['name']}</div>", unsafe_allow_html=True)
            p2.plotly_chart(patch_figure(ex["analog_patch"], ex["analog_saliency"]), width="stretch", config=PLOT_CFG, key="pa")
            p2.markdown(f"<div style='text-align:center' class='place'>{short_place(r)}</div>", unsafe_allow_html=True)
            st.markdown("<div class='sub' style='margin-top:8px'>Each picture is 118 km across; colour is relative height. Hiding a 30 km "
                        "square inside the outline lowers the similarity the most.</div>", unsafe_allow_html=True)
        else:
            st.info("Saliency maps need PyTorch on the server.")
    with w3, glass("report"):
        st.markdown(f"<div class='lbl'><span>{dot()}Report</span></div><div class='sub' style='margin:6px 0 12px'>The ranked list with maps, "
                    "explanations, validation numbers and data citations.</div>", unsafe_allow_html=True)
        if st.button("Prepare report", width="stretch"):
            st.session_state["report"] = build_report_bytes(b, target, weights, pkey, exclude_ice)
        if st.session_state.get("report"):
            zb, pdfb, stem = st.session_state["report"]
            st.download_button("Download PDF", pdfb, f"{stem}.pdf", "application/pdf", width="stretch")
            st.download_button("Markdown + figures", zb, f"{stem}.zip", "application/zip", width="stretch")
    st.session_state["last_match"] = res


def build_report_bytes(b, target, weights, preset, exclude_ice):
    with tempfile.TemporaryDirectory() as tmp:
        global REPORT_DIR
        REPORT_DIR = Path(tmp)
        md = build_report(engine(), profile(b, float(target["lat"]), float(target["lon"])), target["name"], weights, preset, exclude_ice)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in md.parent.iterdir():
                if f.suffix != ".pdf":
                    z.write(f, f.name)
        return buf.getvalue(), md.with_suffix(".pdf").read_bytes(), md.stem


# ================================================================== WATCH THE AI THINK
def page_think():
    log, val = load_json("training_log.json"), load_json("validation.json")
    counts = {b: len(engine().P[b]["lat"]) for b in BODIES}
    st.markdown("<div class='hero-title'>Watch the AI think</div><div class='hero-sub'>How AnalogNet, the model inside THEIA, learned "
                "to compare landscapes across three worlds.</div><br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="steps">
      <div><b>1</b>Cut Earth, Moon and Mars into {sum(counts.values()):,} terrain patches, each 118 km wide at the same 3.7 km pixel size.</div>
      <div><b>2</b>A small neural network turns every patch into a 32-number terrain fingerprint.</div>
      <div><b>3</b>It learns by playing "spot the twin": two altered views of one landscape must get the same fingerprint.</div>
      <div><b>4</b>To find an analog, THEIA looks for Earth patches whose fingerprint is closest to the target's.</div></div>""",
                unsafe_allow_html=True)
    m0 = log["members"][0]
    c1, c2 = st.columns(2, gap="medium")
    with c1, glass("learn"):
        st.markdown(f"<div class='lbl'><span>{dot()}Learning</span></div>", unsafe_allow_html=True)
        cur = np.array(m0["curve"])
        fig = go.Figure([go.Scatter(x=cur[:, 0], y=cur[:, 1], name="Contrastive loss (lower = better)", line=dict(color=CYAN, width=2.5)),
                         go.Scatter(x=cur[:, 0], y=cur[:, 2] * 100, name="Twin found (%)", yaxis="y2", line=dict(color=LAV, width=2.5))])
        fig.update_layout(**plot_layout(height=470, xaxis=dict(title="training step"), yaxis=dict(title="loss"),
                                        yaxis2=dict(overlaying="y", side="right", range=[0, 100], title="twin found %", gridcolor="rgba(0,0,0,0)")))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        st.markdown(f"<div class='sub'>Network 1 of {len(log['members'])}: {cur[-1, 0]:.0f} steps, {m0['seconds'] / 60:.1f} min on a laptop CPU. "
                    "Twin found = how often it picks the right partner view out of hundreds of decoys (chance: under 1 %).</div>", unsafe_allow_html=True)
    with c2, glass("space"):
        st.markdown(f"<div class='lbl'><span>{dot()}The fingerprint space</span></div>", unsafe_allow_html=True)
        mode = st.segmented_control("Map", ["Training replay", "Final map (t-SNE)"], default="Training replay", label_visibility="collapsed") or "Training replay"
        if mode == "Training replay":
            snaps = m0["snapshots"]
            n = len(snaps[0]["xy"]) // 3

            def traces(sn):
                xy = np.array(sn["xy"])
                return [go.Scatter(x=xy[k * n:(k + 1) * n, 0], y=xy[k * n:(k + 1) * n, 1], mode="markers", name=b.title(),
                                     marker=dict(color=BODY_COLOR[b], size=5, opacity=.8)) for k, b in enumerate(BODIES)]
            fig = go.Figure(traces(snaps[0]), frames=[go.Frame(data=traces(sn), name=str(sn["step"])) for sn in snaps])
            fig.update_layout(**plot_layout(height=440, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                                            margin=dict(l=4, r=4, t=36, b=4)),
                              updatemenus=[dict(type="buttons", x=0, y=1.12, xanchor="left", showactive=False, bgcolor="rgba(95,208,255,.12)",
                                                bordercolor=CYAN, font=dict(color=CYAN), buttons=[dict(label="▶ Replay training", method="animate",
                                                args=[None, dict(frame=dict(duration=220, redraw=True), fromcurrent=True)])])],
                              sliders=[dict(currentvalue=dict(prefix="step ", font=dict(color=INK2)), pad=dict(t=10), font=dict(color=INK3),
                                            steps=[dict(label=str(sn["step"]), method="animate",
                                                        args=[[str(sn["step"])], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                                                   for sn in snaps])])
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            st.markdown(f"<div class='sub'>900 fixed patches projected to 2-D at each stage of training. Body mixing at the end: "
                        f"{snaps[-1]['mixing']:.2f} (1 = Earth, Moon and Mars fully interleaved).</div>", unsafe_allow_html=True)
        else:
            ts = val["tsne"]
            fig = go.Figure()
            for bb in ts["bodies"]:
                xy = np.array(bb["xy"])
                fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", name=bb["body"].title(),
                                           marker=dict(color=BODY_COLOR[bb["body"]], size=5, opacity=.6),
                                           text=[f"{bb['body']} {la}, {lo} · relief {rl:.0f} m" for la, lo, rl in zip(bb["lat"], bb["lon"], bb["relief"])],
                                           hoverinfo="text"))
            kn = [s for s in ts["special"] if s["kind"] == "known"]
            tg = [s for s in ts["special"] if s["kind"] == "target"]
            fig.add_trace(go.Scatter(x=[s["xy"][0] for s in kn], y=[s["xy"][1] for s in kn], mode="markers", name="Known Earth analogs",
                                     text=[s["name"] for s in kn], hoverinfo="text", marker=dict(symbol="star", size=11, color="#fff", line=dict(color="#000", width=1))))
            fig.add_trace(go.Scatter(x=[s["xy"][0] for s in tg], y=[s["xy"][1] for s in tg], mode="markers", name="Moon/Mars targets",
                                     text=[s["name"] for s in tg], hoverinfo="text",
                                     marker=dict(symbol="diamond", size=9, color=[BODY_COLOR[s["body"]] for s in tg], line=dict(color="#fff", width=1))))
            fig.update_layout(**plot_layout(height=470, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), margin=dict(l=4, r=4, t=30, b=4)))
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            st.markdown("<div class='sub'>Final fingerprint space of the whole ensemble, flattened with t-SNE (nearby = similar terrain). "
                        "Stars are Earth sites experts already use as analogs.</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="medium")
    res = st.session_state.get("last_match")
    with c3, glass("search"):
        st.markdown(f"<div class='lbl'><span>{dot()}The search</span></div>", unsafe_allow_html=True)
        if res:
            h, e = res["hist"]["counts"], res["hist"]["edges"]
            cut = res["results"][-1]["score"]
            fig = go.Figure([go.Bar(x=[(e[i] + e[i + 1]) / 2 for i in range(len(h))], y=h,
                                    marker_color=[CYAN if e[i + 1] >= cut else "#1f3550" for i in range(len(h))])])
            fig.update_layout(**plot_layout(height=340, xaxis=dict(title="match score (100 = identical)", range=[0, 100]),
                                            yaxis=dict(title="Earth patches", type="log")))
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            st.markdown(f"<div class='sub'>Every one of the {res['n_candidates']:,} Earth land patches gets a score; the cyan bars hold the best 10.</div>",
                        unsafe_allow_html=True)
        else:
            st.info("Open Explore first: the score distribution of your latest search appears here.")
    with c4, glass("live"):
        st.markdown(f"<div class='lbl'><span>{dot(LAV)}Train a network live</span></div>"
                    "<div class='hero-sub' style='margin:6px 0 12px'>Train a brand-new small network from scratch on this server, on fresh tiles "
                    "cut from the maps. It takes about a minute; the shipped ensemble is not changed.</div>", unsafe_allow_html=True)
        if st.button("Start live training"):
            live_training()


def live_training():
    eng = engine()
    rng = np.random.default_rng()
    tiles, probe = {}, {}
    for b in BODIES:
        idx = rng.choice(len(eng.P[b]["lat"]), 1500, replace=False)
        tiles[b] = sample_patches(eng.dem[b], eng.P[b]["lat"][idx], eng.P[b]["lon"][idx], RADIUS_M[b], n=CONTEXT)
        m = (CONTEXT - PATCH) // 2
        probe[b] = tiles[b][:200, m:m + PATCH, m:m + PATCH]
    chart, bar = st.empty(), st.progress(0.0)
    curve = []
    for rep in train_member(tiles, seed=int(rng.integers(1_000_000)), steps=250, batch=192, probe=probe, snapshot_every=50):
        if "model" in rep:
            break
        curve.append((rep["step"], rep["loss"], rep["twin_acc"]))
        bar.progress(rep["step"] / rep["steps"], text=f"step {rep['step']} / {rep['steps']} · loss {rep['loss']:.2f} · twin found {rep['twin_acc'] * 100:.0f}%")
        if rep["step"] % 20 == 0:
            c = np.array(curve)
            fig = go.Figure([go.Scatter(x=c[:, 0], y=c[:, 1], name="loss", line=dict(color=CYAN)),
                             go.Scatter(x=c[:, 0], y=c[:, 2] * 100, name="twin %", yaxis="y2", line=dict(color=LAV))])
            fig.update_layout(**plot_layout(height=260, yaxis2=dict(overlaying="y", side="right", range=[0, 100], gridcolor="rgba(0,0,0,0)")))
            chart.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    st.success(f"Done. A new network learned from scratch; twin-finding rose from chance to {curve[-1][2] * 100:.0f}% in {curve[-1][0]} steps.")


# ================================================================== VALIDATION
def page_validate():
    V = load_json("validation.json")
    st.markdown("<div class='hero-title'>Validation</div><div class='hero-sub'>The model never saw the list of known analog sites. "
                "Here we check whether it rediscovers them. Site score = percentile of the best Earth patch within 120 km of the site "
                "(100 = the most Mars- or Moon-like place on Earth).</div><br>", unsafe_allow_html=True)
    body = (st.segmented_control("Analogs of", ["Mars", "Moon"], default="Mars", label_visibility="collapsed") or "Mars").lower()
    B, names = V["bodies"][body], {m["key"]: m["name"] for m in V["methods"]}
    M = B["methods"]
    main, base = M["analognet"], M["baseline_raw"]
    best = "analognet" if body == "mars" else "neural_only"
    cards = [(f"Mean site score · {'AnalogNet, 4 axes' if best == 'analognet' else 'neural fingerprint'}", f"{M[best]['mean_site_score']:.1f}",
              f"random land: {main['random_mean']:.1f} · p = {M[best]['p_value']}"),
             ("Known sites in Earth's top 10 %", f"{M[best]['recall_top10'] * 100:.0f}%", f"random land: {main['random_recall_top10'] * 100:.0f}%"),
             ("Baseline without neural net", f"{base['auc']:.2f}", "AUC · 0.5 = chance"),
             ("Calibration error of the votes", f"{B['calibration']['ece']:.3f}", "slightly over-confident")]
    for i, (col, (lab, val, sub)) in enumerate(zip(st.columns(4, gap="medium"), cards)):
        with col, glass(f"kpi{i}"):
            st.markdown(f"<div class='lbl'><span>{lab}</span></div><div class='big'>{val}</div><div class='sub'>{sub}</div>", unsafe_allow_html=True)
    keys = [kk for kk in names if kk in M]
    c1, c2 = st.columns(2, gap="medium")
    with c1, glass("methods"):
        st.markdown(f"<div class='lbl'><span>{dot()}Methods compared, with ablations</span></div>", unsafe_allow_html=True)
        fig = go.Figure([go.Bar(orientation="h", y=[names[kk] for kk in keys], x=[M[kk]["mean_site_score"] for kk in keys],
                                marker_color=[CYAN if kk == "analognet" else LAV if kk.startswith("baseline") else ABLATE for kk in keys],
                                text=[f"{M[kk]['mean_site_score']:.1f}" for kk in keys], textposition="outside", cliponaxis=False,
                                customdata=[[M[kk]["auc"], M[kk]["p_value"]] for kk in keys],
                                hovertemplate="%{y}<br>mean site score %{x:.1f}<br>AUC %{customdata[0]:.2f} · p %{customdata[1]}<extra></extra>")])
        fig.add_vline(x=main["random_mean"], line_dash="dash", line_color=BAD, annotation_text="random", annotation_font_color=BAD)
        fig.update_layout(**plot_layout(height=520, margin=dict(l=300, r=30, t=20, b=40), xaxis=dict(range=[0, 108], title="mean site score"),
                                        yaxis=dict(autorange="reversed")))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    with c2, glass("sites"):
        st.markdown(f"<div class='lbl'><span>{dot()}Known sites, one by one</span></div>", unsafe_allow_html=True)
        sites = [s for s in main["sites"] if s["score"] is not None]
        bs = {s["name"]: s["score"] for s in base["sites"]}
        ys = [s["name"].split(",")[0] for s in sites]
        fig = go.Figure([go.Bar(orientation="h", y=ys, x=[s["score"] for s in sites], name="AnalogNet", marker_color=CYAN),
                         go.Bar(orientation="h", y=ys, x=[bs[s["name"]] for s in sites], name="Baseline (raw features)", marker_color=LAV)])
        fig.update_layout(**plot_layout(height=520, barmode="group", margin=dict(l=200, r=10, t=30, b=40), xaxis=dict(range=[0, 100]),
                                        yaxis=dict(autorange="reversed")))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    c3, c4 = st.columns(2, gap="medium")
    with c3, glass("cal"):
        st.markdown(f"<div class='lbl'><span>{dot()}Is the uncertainty honest?</span></div>", unsafe_allow_html=True)
        rel = [x for x in B["calibration"]["reliability"] if x["observed"] is not None]
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfectly calibrated", line=dict(dash="dash", color=INK3)),
                         go.Scatter(x=[x["expected"] for x in rel], y=[x["observed"] for x in rel], mode="lines+markers",
                                    name="AnalogNet ensemble", line=dict(color=CYAN, width=2.5), marker=dict(size=9))])
        fig.update_layout(**plot_layout(height=380, xaxis=dict(title="share of the other networks voting top-100"), yaxis=dict(title="held-out network agrees")))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        st.markdown("<div class='sub'>If 3 of 4 networks vote for a site, the 5th should agree about 75 % of the time. "
                    "Points below the line mean the votes are slightly over-confident.</div>", unsafe_allow_html=True)
    with c4, glass("probe"):
        st.markdown(f"<div class='lbl'><span>{dot()}What did the network learn?</span></div>", unsafe_allow_html=True)
        P = V["probes"]
        fr = sorted(P["feature_r2"].items(), key=lambda x: -x[1])
        fig = go.Figure([go.Bar(x=[FEATURE_NAMES.get(f, f) for f, _ in fr], y=[max(v, 0) for _, v in fr], marker_color=CYAN)])
        fig.update_layout(**plot_layout(height=380, yaxis=dict(range=[0, 1], title="R² read from fingerprint"), xaxis=dict(tickangle=-35), margin=dict(b=110)))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        st.markdown(f"<div class='sub'>Nobody told the network about relief, slope or craters, yet they can be read back from its fingerprint. "
                    f"Which world a patch is from can still be guessed {P['body_id_accuracy_embedding'] * 100:.0f}% of the time "
                    f"(raw features {P['body_id_accuracy_raw_features'] * 100:.0f}%, chance 33%).</div>", unsafe_allow_html=True)
    T, TL = B["targets"], B.get("targets_landform")
    with glass("targets"):
        st.markdown(f"<div class='lbl'><span>{dot()}Per target: top-20 matches near known analogs</span></div>"
                    f"<div class='hero-sub'>Balanced preset: {T['hit_rate'] * 100:.1f}% vs {T['random_rate'] * 100:.1f}% for random land "
                    f"({T['lift']}× lift)." + (f" Shape-only preset: {TL['hit_rate'] * 100:.1f}% ({TL['lift']}× lift)." if TL else "") + "</div>",
                    unsafe_allow_html=True)
        st.dataframe([{"Target": x["target"], "Hits / 20": x["hits_in_top20"], "Top-5 Earth matches": "; ".join(x["examples"])}
                      for x in T["per_target"]], hide_index=True, width="stretch")
    with glass("newc"):
        st.markdown(f"<div class='lbl'><span>{dot(LAV)}New candidate analogs, at least 300 km from every known site</span>"
                    "<span class='sub'>hypotheses for field validation</span></div>", unsafe_allow_html=True)
        cols = st.columns(2) if B.get("new_candidates_landform") else [st.container()]
        for col, (title, lst) in zip(cols, [("Balanced preset (shape + environment)", B["new_candidates"]),
                                            ("Shape only (blind to environment)", B.get("new_candidates_landform", []))]):
            with col:
                st.markdown(f"**{title}**")
                st.dataframe([{"Place": c["place"], "Lat": c["lat"], "Lon": c["lon"], "Score": round(c["score"], 2),
                               "Relief (m)": c["relief_m"], "Bare": f"{c['barefrac'] * 100:.0f}%"} for c in lst],
                             hide_index=True, width="stretch")


# ================================================================== ABOUT
def page_about():
    st.image("static/brand/theia_lockup.png", width=520)
    c1, c2 = st.columns(2, gap="medium")
    with c1, glass("what"):
        st.markdown("""#### What THEIA does
- **Same ruler everywhere.** Earth, Moon and Mars are cut into patches 118 km wide with 3.7 km pixels, sampled on a flat local grid so the poles are not distorted.
- **Learned terrain fingerprint.** AnalogNet, a small convolutional network, turns each elevation patch into 32 numbers. It is trained self-supervised (contrastive learning) on all three worlds at once; no human labels.
- **Four matching axes.** Terrain (neural), rover mobility (slope and roughness), thermal environment and bare ground. Mission presets weight them differently.
- **Uncertainty.** Five networks with different seeds and data subsets: their spread is the error bar, their votes the confidence, and the votes are tested for calibration.
- **Validation.** Known analog sites are used only for testing, against random sites, a non-neural baseline and ablations.

#### Limitations
- Regional scale only (3.7 km pixels): rocks and metre-scale roughness are not seen.
- No mineralogy layer for all three worlds yet; the lava-tube/cave purpose is planned.
- Mars temperatures come from a climate model; MODIS measures Earth at 13:30 and 01:30.
- New candidates are hypotheses for field checks.""")
    with c2, glass("data"):
        st.markdown("#### What is new, and what is not")
        st.write("Learned similarity by itself is not new: Meng et al. (2025, arXiv:2501.08584) trained an Earth–Mars image similarity "
                 "for the Qaidam Basin, and foundation models exist for the Moon and Mars. To our knowledge none puts Earth, Moon and Mars "
                 "elevation in one shared space at one physical scale and ranks the whole Earth with uncertainty and a rediscovery test.")
        st.markdown("#### Data sources (all open)")
        st.dataframe([{"Body": s["body"], "Variable": s["variable"], "Product": s["product"], "Licence": s["license"]} for s in SOURCES]
                     + [{"Body": s["body"], "Variable": "Globe imagery", "Product": s["what"], "Licence": s["license"]} for s in IMAGERY],
                     hide_index=True, width="stretch")
        with st.expander("Citations"):
            for s in SOURCES:
                st.markdown(f"- {s['cite']}")




def run_app() -> None:
    """The website: page settings, styling, navigation."""
    st.set_page_config(page_title="THEIA - Earth analogs for Moon & Mars bases", page_icon="static/brand/favicon.png",
                       layout="wide", initial_sidebar_state="auto")
    st.markdown("""<style>
    @font-face { font-family: "Newsreader"; font-style: italic; font-weight: 500; src: url(app/static/fonts/newsreader-latin-500-italic.woff2) format("woff2"); }
    :root { --cyan: #5fd0ff; --lav: #b8acf0; --ink2: #9fb3cc; --ink3: #5f7591; --edge: rgba(95,208,255,.18); }

    /* deep-space backdrop: a soft glow above the globe and a faint survey grid */
    .stApp { background:
        radial-gradient(1200px 600px at 55% -8%, rgba(40,110,190,.35), transparent 60%),
        radial-gradient(900px 500px at 100% 100%, rgba(120,90,220,.12), transparent 60%),
        repeating-linear-gradient(0deg, rgba(95,208,255,.035) 0 1px, transparent 1px 64px),
        repeating-linear-gradient(90deg, rgba(95,208,255,.035) 0 1px, transparent 1px 64px),
        #040811; }
    [data-testid="stHeader"] { background: rgba(4,8,17,.92); border-bottom: 1px solid var(--edge); }
    [data-testid="stMainBlockContainer"] { padding-top: 4.2rem; max-width: 1560px; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(10,20,38,.92), rgba(4,8,17,.94)); border-right: 1px solid var(--edge); }

    /* pill navigation */
    [data-testid="stTopNavLink"] { border-radius: 999px !important; padding: 4px 16px !important; border: 1px solid transparent; transition: background .2s, border-color .2s; }
    [data-testid="stTopNavLink"]:hover { border-color: var(--edge); background: rgba(95,208,255,.06); }
    [data-testid="stTopNavLink"][aria-current="page"] { border-color: rgba(95,208,255,.55); background: rgba(95,208,255,.10);
        box-shadow: 0 0 18px rgba(95,208,255,.35), inset 0 0 12px rgba(95,208,255,.12); }

    /* glass cards: any container whose key starts with "glass" */
    /* "glass" look without backdrop-filter: blur is recomputed on every scroll and was the main cause of lag */
    [class*="st-key-glass"] { background: linear-gradient(160deg, rgba(20,37,64,.88), rgba(7,13,26,.92)); border: 1px solid var(--edge);
        border-radius: 22px; padding: 16px 18px 14px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 12px 28px -16px rgba(0,0,0,.9); }
    [class*="st-key-glassglobe"] { padding: 8px; overflow: hidden; box-shadow: inset 0 1px 0 rgba(255,255,255,.06),
        0 0 0 1px rgba(95,208,255,.10), 0 0 32px -8px rgba(60,160,255,.35); }
    [class*="st-key-glassglobe"] iframe { border-radius: 16px; }
    [class*="st-key-glass"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }   /* last line was clipped */
    [class*="st-key-glass"] { overflow: visible; }
    [class*="st-key-glassglobe"] { overflow: hidden; }

    .place { font-family: "Newsreader", Georgia, serif; font-style: italic; font-weight: 500; }
    .lbl { color: var(--ink2); font-size: .82rem; letter-spacing: .02em; display: flex; justify-content: space-between; align-items: center; }
    .lbl .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 10px var(--cyan); display: inline-block; margin-right: 8px; }
    .big { font-size: 2.3rem; font-weight: 600; line-height: 1.05; font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
    .big small { font-size: .9rem; color: var(--ink2); font-weight: 400; margin-left: 6px; }
    .sub { color: var(--ink3); font-size: .8rem; }
    .hero-title { font-size: clamp(1.45rem, .9rem + 1.5vw, 2.25rem); font-weight: 600; letter-spacing: -.015em; line-height: 1.15; margin: 0; }
    .hero-sub { color: var(--ink2); margin-top: 4px; }
    .chip { display: inline-block; padding: 2px 10px; border-radius: 999px; border: 1px solid var(--edge); color: var(--ink2); font-size: .75rem; margin-right: 6px; }
    .chip.moon { border-color: rgba(201,204,214,.5); color: #dfe1e8; } .chip.mars { border-color: rgba(224,122,79,.6); color: #f0a07e; }

    .res-row { display: grid; grid-template-columns: 22px 1fr auto; gap: 0 12px; padding: 9px 6px; border-radius: 12px; border: 1px solid transparent; }
    .res-row + .res-row { margin-top: 2px; }
    .res-row:hover { background: rgba(95,208,255,.05); }
    .res-row .rk { color: var(--ink3); font-weight: 600; text-align: right; padding-top: 3px; }
    .res-row.sel { background: rgba(184,172,240,.10); border-color: rgba(184,172,240,.45); box-shadow: 0 0 22px -6px rgba(184,172,240,.45); }
    .res-row.sel .rk { color: var(--lav); }
    .res-row .pl .place { font-size: 1.04rem; display: block; }
    .res-row .pl small { color: var(--ink3); }
    .res-row .sc { text-align: right; font-size: 1.22rem; font-weight: 600; line-height: 1.1; font-variant-numeric: tabular-nums; }
    .res-row .sc small { display: block; font-size: .7rem; font-weight: 400; color: var(--ink3); }
    .ubar { position: relative; height: 5px; background: rgba(95,208,255,.10); border-radius: 3px; margin-top: 7px; }
    .ubar .rng { position: absolute; top: 0; bottom: 0; background: linear-gradient(90deg, rgba(95,208,255,.25), rgba(95,208,255,.6)); border-radius: 3px; }
    .ubar .mid { position: absolute; top: -3px; width: 2px; height: 11px; background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }
    .votes { display: inline-flex; gap: 3px; margin-left: 10px; }
    .votes i { width: 7px; height: 7px; border-radius: 2px; background: rgba(95,208,255,.15); }
    .votes i.on { background: var(--cyan); box-shadow: 0 0 6px rgba(95,208,255,.8); }
    .known { font-size: .76rem; color: #cfd6e6; margin-top: 3px; }
    .axis { display: grid; grid-template-columns: 1fr 2fr 34px; gap: 6px 10px; align-items: center; font-size: .82rem; color: var(--ink2); margin-top: 6px; }
    .axis .bar { height: 5px; background: rgba(95,208,255,.10); border-radius: 3px; overflow: hidden; }
    .axis .bar span { display: block; height: 100%; background: linear-gradient(90deg, #2f8fd8, var(--cyan)); box-shadow: 0 0 8px rgba(95,208,255,.6); }
    .axis .v { text-align: right; color: #eaf3ff; font-variant-numeric: tabular-nums; }
    .steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 1rem; }
    .steps div { padding: 16px 18px; color: var(--ink2); font-size: .92rem; line-height: 1.45; border-radius: 18px; border: 1px solid var(--edge);
        background: linear-gradient(160deg, rgba(22,40,70,.55), rgba(6,12,24,.62)); }
    .steps b { display: block; color: var(--cyan); font-size: 1.5rem; margin-bottom: 4px; text-shadow: 0 0 14px rgba(95,208,255,.6); }
    .stButton button, .stDownloadButton button { border-radius: 999px !important; border: 1px solid rgba(95,208,255,.45) !important;
        background: rgba(95,208,255,.08) !important; box-shadow: 0 0 18px -6px rgba(95,208,255,.6); }
    .stButton button:hover, .stDownloadButton button:hover { background: rgba(95,208,255,.16) !important; }
    .toolbar-hint { color: var(--ink3); font-size: .78rem; text-align: right; padding-top: 1.9rem; }

    /* ---------- tablet */
    @media (max-width: 1100px) {
      [data-testid="stMainBlockContainer"] { padding-left: 1.2rem; padding-right: 1.2rem; }
      .steps { grid-template-columns: repeat(2, 1fr); }
      .toolbar-hint { display: none; }
    }
    /* ---------- phone */
    @media (max-width: 640px) {
      [data-testid="stMainBlockContainer"] { padding: 3.6rem .7rem 2rem; }
      [class*="st-key-glass"] { border-radius: 16px; padding: 12px 12px 10px; }
      [class*="st-key-glassglobe"] { padding: 4px; }
      [class*="st-key-glassglobe"] iframe { border-radius: 12px; }
      .big { font-size: 1.9rem; }
      .hero-sub { font-size: .9rem; }
      .steps { grid-template-columns: 1fr; }
      .axis { grid-template-columns: 1fr 1.4fr 30px; font-size: .78rem; }
      [data-testid="stTopNavLink"] { padding: 3px 10px !important; }
      [data-testid="stSegmentedControl"] button { padding-left: 10px; padding-right: 10px; }
    }
    </style>""", unsafe_allow_html=True)
    pg = st.navigation([st.Page(page_explore, title="Explore", icon=":material/public:", default=True),
                        st.Page(page_think, title="Watch the AI think", icon=":material/neurology:"),
                        st.Page(page_validate, title="Validation", icon=":material/fact_check:"),
                        st.Page(page_about, title="Data & method", icon=":material/menu_book:")], position="top")
    st.logo("static/brand/theia_wordmark.png", size="large", icon_image="static/brand/favicon.png")
    with st.sidebar:
        st.markdown("<div style='color:#9fb3cc;font-size:.85rem;margin:-6px 0 6px'>Terrestrial Habitat Evaluation and Interplanetary Analogue</div>",
                    unsafe_allow_html=True)
    pg.run()


# ==================================================================================================
# 15. BUILD PIPELINE - rebuild everything from the original open data
# ==================================================================================================

# ---- build download: download the raw open data (~3.5 GB, internet once)
# Step 1 - download the raw open data (about 3.4 GB, needs internet once).
#
# Run:  python streamlit_app.py build download
#
# You only need this if you want to REBUILD the dataset from scratch.  The app
# itself ships with the processed assets and runs offline.
MCD_CGI = "https://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py"
MCD_BASE = "https://www-mars.lmd.jussieu.fr/mcd_python/"


def fetch(url: str, dest: Path) -> None:
    """Stream one file to disk (skips files that already exist)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have   {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _http().get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    print(f"  got    {dest.name}  ({dest.stat().st_size / 1e6:.0f} MB)")


def fetch_mcd() -> None:
    """Ask the Mars Climate Database web interface for 48 global surface-temperature
    maps (4 seasons x 12 local times).  We pause between requests to be polite to
    a shared scientific server."""
    out = RAW_DIR / "mcd"
    out.mkdir(parents=True, exist_ok=True)
    for ls in MCD_SEASONS:
        for lt in MCD_LOCAL_TIMES:
            dest = out / f"tsurf_ls{ls}_lt{lt}.txt"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            params = dict(var1="tsurf", var2="none", var3="none", var4="none", datekeyhtml=1,
                          ls=ls, localtime=lt, latitude="all", longitude="all", altitude="10.",
                          zkey=3, isfixedlt="on", dust=1, hrkey=1, zonmean="off", diumean="off",
                          averaging="off", dpi=80, islog="off", colorm="jet", proj="cyl", iswind="off")
            html = _http().get(MCD_CGI, params=params, timeout=300).text
            m = re.search(r"txt/[a-f0-9]+\.txt", html)
            if not m:
                raise RuntimeError(f"MCD did not return data for Ls={ls}, LT={lt}")
            dest.write_text(_http().get(MCD_BASE + m.group(0), timeout=120).text)
            print(f"  got    {dest.name}")
            time.sleep(1.0)


def build_download() -> None:
    for src in SOURCES:
        print(f"{src['body']:5s} | {src['variable']}: {src['product']}")
        for rel, url in src["files"].items():
            fetch(url, RAW_DIR / rel)
    print("Mars  | surface temperature maps from the Mars Climate Database")
    fetch_mcd()
    print("Earth | skin temperature is read directly from the NASA POWER cloud store in step 2.")
    print("Done.")


# ---- build modis: MODIS temperature + NDVI from Microsoft Planetary Computer
# Step 1b - download MODIS land-surface temperature and NDVI (Earth) from
# Microsoft Planetary Computer (https://planetarycomputer.microsoft.com/catalog).
#
# Collections: modis-11A2-061 (MYD11A2 LST, 12 monthly composites of 2023, day + night)
#              modis-13A1-061 (MYD13A1 NDVI, 4 seasonal composites of 2023)
# Only reduced-resolution overviews are read (~7 km for LST, ~3.7 km for NDVI),
# about 1 GB of HTTP range requests in total.  Needs the 'rasterio' package.
def build_modis() -> None:
    t0 = time.time()
    out = fetch_all(log=lambda s: print(s, f"({time.time() - t0:.0f} s)", flush=True))
    dest = RAW_DIR / "modis" / "modis_2023.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    print("saved", dest)


# ---- build dataset: global grids, patches and features
# Step 2 - build the processed assets from the raw downloads.
#
# Run:  python streamlit_app.py build dataset
#
# Produces (in assets/):
#   dem_<body>.npz        global elevation grid, int16 metres (+ land mask for Earth)
#   thermal_<body>.npz    4 temperature layers on that body's thermal grid
#   patches_<body>.npz    patch centres + hand-crafted features + thermal features
# and (in data/processed/, not shipped) the elevation patches used for training.
def build_dataset() -> None:
    ASSET_DIR.mkdir(exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    tex = STATIC_DIR / "textures"
    tex.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 1. global grids ---------------------------------------------------
    print("Reading elevation grids ...")
    dem = {"moon": moon_dem(), "mars": mars_dem()}
    dem["earth"], land, ice = earth_dem()
    for b in BODIES:
        extra = {"land": np.packbits(land), "ice": np.packbits(ice)} if b == "earth" else {}
        np.savez_compressed(ASSET_DIR / f"dem_{b}.npz", dem=dem[b], **extra)
        print(f"  {b:5s} {dem[b].shape}  min {dem[b].min()} m  max {dem[b].max()} m")
    for b in ("moon", "mars"):
        shaded_relief(None, b).save(tex / f"{b}.jpg", quality=90)

    print("Reading temperature data ...")
    therm = {"moon": moon_thermal(), "mars": mars_thermal(), "earth": earth_thermal()}
    for b in BODIES:
        np.savez_compressed(ASSET_DIR / f"thermal_{b}.npz", **therm[b])
        d = therm[b]["data"]
        print(f"  {b:5s} grid {d.shape[1:]}  " + "  ".join(
            f"{n}: {d[i].min():.0f}..{d[i].max():.0f}" for i, n in enumerate(THERMAL_FEATURES)))

    # ---- 2. patches ----------------------------------------------------------
    for b in BODIES:
        lat, lon = fibonacci_lattice(RADIUS_M[b], CENTRE_SPACING_KM)
        grid = dem[b].astype(np.float32)
        if b == "earth":
            # keep only centres whose footprint is at least half land
            frac = sample_patches(land.astype(np.float32), lat, lon, RADIUS_M[b]).mean(axis=(1, 2))
            keep = frac >= MIN_LAND_FRACTION
            lat, lon, frac = lat[keep], lon[keep], frac[keep]
            icefrac = sample_patches(ice.astype(np.float32), lat, lon, RADIUS_M[b]).mean(axis=(1, 2))
        else:
            frac, icefrac = np.ones_like(lat), np.zeros_like(lat)
        # 48x48 training tile; its central 32x32 window IS the patch (same pixel grid)
        context = sample_patches(grid, lat, lon, RADIUS_M[b], n=CONTEXT)
        m = (CONTEXT - PATCH) // 2
        patches = context[:, m:m + PATCH, m:m + PATCH]
        morph = morphometry(patches)
        th = therm[b]
        thermal = np.stack([interp_nodes(th["data"][i], th["lat"], th["lon"], lat, lon)
                            for i in range(len(THERMAL_FEATURES))], axis=1)
        np.save(PROC_DIR / f"context_{b}.npy", context.round().astype(np.int16))
        np.savez_compressed(ASSET_DIR / f"patches_{b}.npz", lat=lat, lon=lon,
                            morph=morph, thermal=thermal, landfrac=frac.astype(np.float32),
                            icefrac=icefrac.astype(np.float32))
        print(f"  {b:5s} {len(lat):6d} patches   ({time.time() - t0:.0f} s)")
    print("Done.")


# ---- build names: offline place names for Earth patches
# Step 2b - give every Earth patch a human-readable place name (offline).
#
# Uses Natural Earth (public domain) country polygons and named geographic
# regions (deserts, ranges, plateaus, basins ...).  When several regions overlap
# a patch centre, the smallest one wins (more specific: "Atacama" beats "Andes").
#
# Output: assets/names_earth.json  {"region": [...], "country": [...]}
NAMES_W, NAMES_H = 7200, 3600                           # 0.05 deg lookup raster
SKIP = {"Continent", "Dragons-be-here"}


def read(name: str):
    import shapefile
    z = zipfile.ZipFile(RAW_DIR / "naturalearth" / f"{name}.zip")
    return shapefile.Reader(shp=io.BytesIO(z.read(f"{name}.shp")), shx=io.BytesIO(z.read(f"{name}.shx")),
                            dbf=io.BytesIO(z.read(f"{name}.dbf")))


def burn(reader, label_of, keep=lambda rec: True) -> tuple[np.ndarray, list[str]]:
    """Rasterise polygons with integer ids; larger polygons are drawn first so that
    smaller (more specific) ones overwrite them."""
    from PIL import Image, ImageDraw
    items = []
    for shp, rec in zip(reader.shapes(), reader.records()):
        if not keep(rec) or not shp.points:
            continue
        x0, y0, x1, y1 = shp.bbox
        items.append(((x1 - x0) * (y1 - y0), shp, label_of(rec)))
    items.sort(key=lambda t: -t[0])
    labels = [""]
    img = Image.new("I", (NAMES_W, NAMES_H), 0)
    draw = ImageDraw.Draw(img)
    for _, shp, label in items:
        labels.append(label)
        pts = np.asarray(shp.points)
        parts = list(shp.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            r = pts[parts[i]:parts[i + 1]]
            if len(r) < 3:
                continue
            xy = np.column_stack([(r[:, 0] + 180) / 360 * NAMES_W, (90 - r[:, 1]) / 180 * NAMES_H])
            draw.polygon([tuple(p) for p in xy], fill=len(labels) - 1)
    return np.asarray(img), labels


def build_names() -> None:
    p = np.load(ASSET_DIR / "patches_earth.npz")
    rows = np.clip(((90 - p["lat"]) / 180 * NAMES_H).astype(int), 0, NAMES_H - 1)
    cols = np.clip(((p["lon"] + 180) / 360 * NAMES_W).astype(int), 0, NAMES_W - 1)
    cty, cl = burn(read("ne_10m_admin_0_countries"), lambda r: r["NAME_EN"] or r["NAME"])
    reg, rl = burn(read("ne_10m_geography_regions_polys"), lambda r: r["NAME_EN"] or r["NAME"],
                   keep=lambda r: r["FEATURECLA"] not in SKIP)
    out = {"country": [cl[i] for i in cty[rows, cols]], "region": [rl[i] for i in reg[rows, cols]]}
    # Antarctica has no country polygon of its own in some versions - label it explicitly
    out["country"] = ["Antarctica" if (c == "" and la < -60) else c for c, la in zip(out["country"], p["lat"])]
    (ASSET_DIR / "names_earth.json").write_text(json.dumps(out))
    named = sum(1 for r in out["region"] if r)
    print(f"{len(rows)} patches, {named} with a region name, "
          f"{sum(1 for c in out['country'] if c)} with a country")


# ---- build surface: Earth temperature and bare ground from MODIS
# Step 2c - Earth surface temperature and bare-ground fraction from MODIS.
#
# Needs data/raw/modis/modis_2023.npz, created by:
#     python streamlit_app.py build modis (reads Microsoft Planetary Computer)
#
# Updates assets/thermal_earth.npz and the 'thermal' + 'barefrac' fields of
# assets/patches_earth.npz.  The neural network is NOT affected (it only sees
# elevation), so this step does not require re-training.
def build_surface() -> None:
    therm = earth_thermal()
    # float16 = 0.25 K steps near 300 K, far finer than the data's accuracy; halves the file size
    np.savez_compressed(ASSET_DIR / "thermal_earth.npz", **{**therm, "data": therm["data"].astype(np.float16)})
    print("Earth thermal grid:", therm["data"].shape, "| MODIS share per layer:",
          np.round(therm.get("modis_fraction", np.zeros(4)), 2))
    ice = np.unpackbits(np.load(ASSET_DIR / "dem_earth.npz")["ice"])[: 5400 * 10800].reshape(5400, 10800).astype(bool)
    p = dict(np.load(ASSET_DIR / "patches_earth.npz"))
    old = p["thermal"].copy()
    p["thermal"], p["barefrac"] = earth_patch_surface(p["lat"], p["lon"], therm, ice)
    np.savez_compressed(ASSET_DIR / "patches_earth.npz", **p)
    print("Patch thermal change vs previous (median |dK|):",
          np.round(np.median(np.abs(p["thermal"] - old), axis=0), 1))
    print(f"Bare-ground fraction: median {np.median(p['barefrac']):.2f}, "
          f"{(p['barefrac'] > 0.8).mean() * 100:.0f}% of land patches are >80% bare")


# ---- build texture: true-colour Earth globe texture
# Step 2d - build the detailed true-colour Earth texture for the globe (needs internet once).
#
# Reads MODIS MCD43A4 surface reflectance from Microsoft Planetary Computer, adds ETOPO
# relief shading and ocean depth colouring, and writes an offline tile pyramid to
# static/textures/earth/.  About 10-15 minutes and ~6 GB of RAM.
def build_texture() -> None:
    t0 = time.time()
    make_earth_texture(log=lambda m: print(f"{m}  ({time.time() - t0:.0f} s)", flush=True))
    print("done")


# ---- build hologram: hologram globe textures
# Step 2e - stylised "hologram" globe textures (the default look of the globe).
#
# Earth: dark ocean with faint sea-floor relief, land drawn as a glowing cyan dot matrix whose
# brightness follows the terrain, bright coastlines and a faint 15-degree graticule.
# Moon / Mars: the LOLA / MOLA hill-shade recoloured from deep navy to cyan.
# Output: static/textures/{earth,moon,mars}_holo.jpg
OUT = STATIC_DIR / "textures"


def hillshade(z, px, exag):
    gy, gx = np.gradient(z.astype(np.float32), px)
    lat = np.radians(90 - (np.arange(z.shape[0]) + 0.5) * 180 / z.shape[0])[:, None]
    gx = gx / np.cos(lat).clip(0.05, 1)
    slope = np.arctan(exag * np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    alt, az = np.radians(40), np.radians(315)
    s = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(s / np.sin(alt), 0, 1.5)


def graticule(h, w, step_deg=15, width=1):
    g = np.zeros((h, w), np.float32)
    for lat in range(-90 + step_deg, 90, step_deg):
        r = int((90 - lat) / 180 * h); g[max(r - width // 2, 0):r + width // 2 + 1] = 1
    for lon in range(-180, 180, step_deg):
        c = int((lon + 180) / 360 * w); g[:, max(c - width // 2, 0):c + width // 2 + 1] = 1
    return g


def colorize(stack):
    return Image.fromarray(np.clip(stack, 0, 255).astype(np.uint8))


def holo_earth():
    W, H = 8192, 4096
    dem = np.load(ASSET_DIR / "dem_earth.npz")
    z = Image.fromarray(dem["dem"].astype(np.float32)).resize((W, H), Image.BILINEAR)
    z = np.asarray(z)
    land = np.unpackbits(dem["land"])[: 5400 * 10800].reshape(5400, 10800)
    lakes_or_land = land.astype(bool) | (dem["dem"] != 0)          # sea is flattened to exactly 0 m in the dataset
    land = np.asarray(Image.fromarray(lakes_or_land.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    shade = hillshade(z, 2 * np.pi * RADIUS_M["earth"] / W, 10)
    # ocean: near-black navy, slightly lighter on shelves
    ocean = np.zeros((H, W, 3), np.float32) + np.array([3, 9, 20])
    # land base: deep blue, lit by relief
    base = np.array([8, 30, 54])[None, None] * (0.7 + 0.5 * shade[..., None])
    img = np.where(land[..., None], base, ocean)
    # dot matrix on land: a dot every 7 px, brighter on lit slopes and high ground
    yy, xx = np.mgrid[0:H, 0:W]
    dots = (((yy % 7) - 3) ** 2 + ((xx % 7) - 3) ** 2) <= 2
    hi = np.clip(z / 3000, 0, 1)
    dval = np.clip(0.45 + 0.45 * shade + 0.35 * hi, 0, 1.3) * (dots & land)
    img += dval[..., None] * np.array([70, 190, 255])[None, None] * 0.85
    # coastline glow
    edge = land ^ ndi.binary_erosion(land, iterations=2)
    glow = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), np.float32) / 255
    img += (edge[..., None] * np.array([120, 230, 255]) * 0.9) + glow[..., None] * np.array([40, 150, 255]) * 1.6
    img += graticule(H, W, 15, 2)[..., None] * np.array([12, 45, 80])
    colorize(img).save(OUT / "earth_holo.jpg", quality=90)


def holo_body(name):
    if name == "moon":
        z = np.fromfile(RAW_DIR / "moon" / "ldem_16.img", dtype="<i2").reshape(2880, 5760).astype(np.float32) * 0.5
        exag = 5.0
    else:
        z = np.fromfile(RAW_DIR / "mars" / "megt90n000eb.img", dtype=">i2").reshape(2880, 5760).astype(np.float32)
        exag = 7.0
    z = np.roll(z, z.shape[1] // 2, axis=1)
    s = hillshade(z, 2 * np.pi * RADIUS_M[name] / z.shape[1], exag)
    lo, hi = np.percentile(z, [1, 99])
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    v = np.clip(0.15 + 0.55 * s + 0.25 * t, 0, 1.2)[..., None]
    c0, c1 = np.array([2, 8, 20]), np.array([80, 200, 255]) if name == "moon" else np.array([120, 170, 255])
    img = c0 * (1 - v) + c1 * v
    img += graticule(*z.shape, 15, 2)[..., None] * np.array([10, 40, 75])
    colorize(img).save(OUT / f"{name}_holo.jpg", quality=90)


def build_hologram() -> None:
    holo_earth(); holo_body("moon"); holo_body("mars")
    print("wrote", sorted(p.name for p in OUT.glob("*_holo.jpg")))


# ---- build tiles: cut globe textures into tiles (fast zoom)
# Step 2f - cut the single-image globe textures into tile pyramids (fast, light zooming).
#
# A 8192 x 4096 picture loaded in one go costs ~130 MB of GPU memory and stalls the
# first frame. Tiles (512 px, WebP, geographic 2x1 layout at level 0) let the globe load
# only what is on screen at the current zoom - instant zoom, low memory.
# Output: static/textures/<name>/<z>/<x>/<y>.webp for earth_holo, moon_holo, mars_holo, moon, mars.
PYR_TILE, PYR_LEVELS = 512, 4                    # level 3 = 16 x 8 tiles = 8192 x 4096 px
Image.MAX_IMAGE_PIXELS = None


def pyramid(name: str) -> int:
    src = Image.open(STATIC_DIR / "textures" / f"{name}.jpg").convert("RGB")
    root = STATIC_DIR / "textures" / name
    n = 0
    for z in range(PYR_LEVELS):
        nx, ny = 2 * 2 ** z, 2 ** z
        lvl = src.resize((nx * PYR_TILE, ny * PYR_TILE), Image.LANCZOS)
        for x in range(nx):
            (root / str(z) / str(x)).mkdir(parents=True, exist_ok=True)
            for y in range(ny):
                lvl.crop((x * PYR_TILE, y * PYR_TILE, (x + 1) * PYR_TILE, (y + 1) * PYR_TILE)).save(
                    root / str(z) / str(x) / f"{y}.webp", quality=82, method=4)
                n += 1
    return n


def build_tiles() -> None:
    for name in ("earth_holo", "moon_holo", "mars_holo", "moon", "mars"):
        print(name, pyramid(name), "tiles")


# ---- build train: train the 5-network ensemble
# Step 3 - train the ensemble of terrain encoders and embed every patch.
#
# Run:  python streamlit_app.py build train (about 30-40 min on a 4-core laptop CPU)
#       python streamlit_app.py build train --quick (2 members x 400 steps, for testing)
#
# Produces (in assets/):
#   encoder_<k>.pt        weights of ensemble member k
#   embeddings.npz        fingerprints of every patch from every member (float16)
#   training_log.json     loss curves + embedding snapshots (replayed in the app)
def build_train(quick: bool = False) -> None:
    members = 2 if quick else ENSEMBLE
    steps = 400 if quick else TRAIN_STEPS
    torch.set_num_threads(max(1, torch.get_num_threads()))

    tiles = {b: np.load(PROC_DIR / f"context_{b}.npy", mmap_mode="r") for b in BODIES}
    tiles = {b: np.ascontiguousarray(t) for b, t in tiles.items()}   # into RAM (~600 MB)
    m = (CONTEXT - PATCH) // 2
    patches = {b: t[:, m:m + PATCH, m:m + PATCH] for b, t in tiles.items()}

    # fixed probe set: the same 300 patches per body are projected at every snapshot
    rng = np.random.default_rng(SEED)
    probe_idx = {b: np.sort(rng.choice(len(patches[b]), 300, replace=False)) for b in BODIES}
    probe = {b: patches[b][probe_idx[b]] for b in BODIES}

    log = {"members": [], "probe_idx": {b: probe_idx[b].tolist() for b in BODIES},
           "config": {"steps": steps, "batch": BATCH, "embed_dim": EMBED_DIM,
                      "temperature": TEMPERATURE, "lr": LR, "pixel_m": PIXEL_M,
                      "patch": PATCH, "context": CONTEXT, "crop": CROP}}
    all_emb = {b: [] for b in BODIES}
    for k in range(members):
        print(f"=== ensemble member {k + 1}/{members} ===", flush=True)
        curve, snaps = [], []
        for rep in train_member(tiles, seed=SEED + k, steps=steps, probe=probe,
                                snapshot_every=50 if steps >= 1000 else 20):
            if "model" in rep:
                model, secs = rep["model"], rep["seconds"]
                break
            curve.append([rep["step"], rep["loss"], rep["twin_acc"]])
            if "xy" in rep:
                snaps.append({"step": rep["step"], "xy": rep["xy"], "mixing": rep["mixing"]})
                print(f"  step {rep['step']:5d}  loss {rep['loss']:.3f}  twin-acc {rep['twin_acc']:.3f}"
                      f"  body-mixing {rep['mixing']:.2f}  ({rep['seconds']:.0f} s)", flush=True)
        torch.save(model.state_dict(), ASSET_DIR / f"encoder_{k}.pt")
        t0 = time.time()
        for b in BODIES:
            all_emb[b].append(embed(model, patches[b]).astype(np.float16))
        print(f"  embedded all patches in {time.time() - t0:.0f} s", flush=True)
        log["members"].append({"seed": SEED + k, "seconds": secs, "curve": curve, "snapshots": snaps})
        # write after every member so a crash does not lose finished work
        (ASSET_DIR / "training_log.json").write_text(json.dumps(log))

    np.savez_compressed(ASSET_DIR / "embeddings.npz",
                        **{b: np.stack(all_emb[b]) for b in BODIES})   # (members, N, D)
    print("Done.")


# ---- build evaluate: validation against known analogs
# Step 4 - validation: does AnalogNet rediscover the analogs experts already use?
#
# Run:  python streamlit_app.py build evaluate (~5-10 min)
#
# IMPORTANT for honesty: the network is trained WITHOUT any labels, and the list
# of known analog sites (section 2) is never used for training or for
# choosing weights.  It is used here only, as an answer key.
#
# Outputs:
#   assets/validation.json   all numbers shown in the app's Validation dashboard
#   assets/analog_maps.npz   global "Moon-likeness"/"Mars-likeness" per Earth patch + t-SNE map
KNN = 20                 # an Earth patch's "body-likeness" = mean distance to its 20 closest body patches
BODY_SAMPLE = 8000       # body patches used as the reference set
SITE_RADIUS_KM = 120.0   # a known analog "is found" through any candidate patch within this radius
N_RANDOM = 2000          # random site sets for the significance test
RNG = np.random.default_rng(SEED)
BAL = PRESETS["balanced"]["weights"]


def methods(eng: Engine) -> list[dict]:
    """Every method is a recipe for a distance between Earth patches and body patches."""
    def w(**kw):
        return {a: kw.get(a, 0.0) for a in AXES}
    out = [
        {"key": "analognet", "name": "AnalogNet (4 axes, balanced, 5-network ensemble)", "weights": BAL},
        {"key": "neural_only", "name": "Neural terrain fingerprint only", "weights": w(terrain=1)},
        {"key": "baseline_raw", "name": "Baseline: cosine similarity on raw features (no neural net)",
         "baseline": "all"},
        {"key": "baseline_morph", "name": "Baseline: raw terrain features only (cosine)", "baseline": "morph"},
        {"key": "no_terrain", "name": "Ablation: without neural terrain", "weights": {**BAL, "terrain": 0}},
        {"key": "no_mobility", "name": "Ablation: without mobility", "weights": {**BAL, "mobility": 0}},
        {"key": "no_thermal", "name": "Ablation: without thermal", "weights": {**BAL, "thermal": 0}},
        {"key": "no_exposure", "name": "Ablation: without bare-ground exposure", "weights": {**BAL, "exposure": 0}},
        {"key": "single_net", "name": "Ablation: single network instead of ensemble", "weights": BAL, "members": [0]},
        {"key": "aligned", "name": "Variant: body-mean-centred fingerprint (domain alignment)", "weights": BAL, "aligned": True},
    ]
    if not eng.has_exposure:
        out = [m for m in out if m["key"] != "no_exposure"]
    return out


def raw_vectors(eng: Engine, body: str, idx, kind: str) -> np.ndarray:
    p = eng.P[body]
    parts = [p["mz"][idx]]
    if kind == "all":
        parts += [p["tz"][idx], ((p["barefrac"][idx] - 1.0) / eng.bare_sd)[:, None]]
    v = np.concatenate(parts, axis=1)
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-6)


def body_scores(eng: Engine, body: str, cand: np.ndarray, meths: list[dict]) -> dict:
    """For every method: (members, n_cand) 'body-likeness' percentile score of each Earth candidate."""
    ref = RNG.choice(len(eng.P[body]["lat"]), min(BODY_SAMPLE, len(eng.P[body]["lat"])), replace=False)
    E, P = eng.E, eng.P
    # domain-aligned fingerprints: subtract each body's mean fingerprint, re-normalise
    Ec = {}
    for b in ("earth", body):
        x = E[b] - E[b].mean(axis=1, keepdims=True)
        Ec[b] = x / np.linalg.norm(x, axis=2, keepdims=True).clip(1e-6)
    sc_al = float(np.median(1 - np.sum(Ec["earth"][0, :5000] * Ec[body][0, RNG.integers(0, len(ref), 5000) % Ec[body].shape[1]], axis=1)))
    raw_ref = {k: raw_vectors(eng, body, ref, k) for k in ("all", "morph")}
    acc = {m["key"]: [] for m in meths}
    for s in range(0, len(cand), 1536):
        ci = cand[s:s + 1536]
        terr = np.stack([1 - E["earth"][k, ci] @ E[body][k, ref].T for k in range(eng.members)]) / eng.scale["terrain"]
        terr_al = np.stack([1 - Ec["earth"][k, ci] @ Ec[body][k, ref].T for k in range(eng.members)]) / sc_al
        mob = eng._mob_dist(P["earth"]["mz"][ci][:, None], P[body]["mz"][ref][None]) / eng.scale["mobility"]
        th = eng._th_dist(P["earth"]["tz"][ci][:, None], P[body]["tz"][ref][None]) / eng.scale["thermal"]
        ex = np.repeat((np.abs(1 - P["earth"]["barefrac"][ci]) / eng.scale["exposure"])[:, None], len(ref), 1)
        for m in meths:
            if "baseline" in m:
                rv = raw_vectors(eng, "earth", ci, m["baseline"])
                D = (1 - rv @ raw_ref[m["baseline"]].T)[None]
            else:
                mem = m.get("members", list(range(eng.members)))
                wt = m["weights"]
                tt = terr_al if m.get("aligned") else terr
                D = wt.get("terrain", 0) * tt[mem] + (wt.get("mobility", 0) * mob + wt.get("thermal", 0) * th
                                                         + wt.get("exposure", 0) * ex)[None]
            knn = np.partition(D, KNN, axis=2)[:, :, :KNN].mean(axis=2)       # (members, chunk)
            acc[m["key"]].append(knn)
    return {k: eng.percentile_score(np.concatenate(v, axis=1)) for k, v in acc.items()}


def random_site_cache(cand_lat, cand_lon, n_pick: int) -> list[np.ndarray]:
    """Neighbourhood index lists of random candidate locations (shared by all methods)."""
    R = RADIUS_M["earth"]
    out = []
    for _ in range(N_RANDOM):
        pick = RNG.choice(len(cand_lat), n_pick, replace=False)
        out.append([np.where(great_circle_km(cand_lat[j], cand_lon[j], cand_lat, cand_lon, R) <= SITE_RADIUS_KM)[0]
                    for j in pick])
    return out


def site_metrics_fast(score, site_nbrs, names, rand_nbrs) -> dict:
    per_site = [{"name": n, "score": round(float(score[nb].max()), 1) if len(nb) else None}
                for n, nb in zip(names, site_nbrs)]
    vals = np.array([p["score"] for p in per_site if p["score"] is not None])
    rand_means = np.array([np.mean([score[nb].max() for nb in sset]) for sset in rand_nbrs])
    rand_recall = np.array([np.mean([score[nb].max() >= 90 for nb in sset]) for sset in rand_nbrs])
    pos_mask = np.zeros(len(score), bool)
    for nb in site_nbrs:
        pos_mask[nb] = True
    pos, neg = score[pos_mask], score[~pos_mask]
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return {"sites": per_site, "n_sites": int(len(vals)),
            "mean_site_score": round(float(vals.mean()), 1),
            "median_site_score": round(float(np.median(vals)), 1),
            "recall_top10": round(float((vals >= 90).mean()), 3),
            "recall_top5": round(float((vals >= 95).mean()), 3),
            "random_mean": round(float(rand_means.mean()), 1),
            "random_recall_top10": round(float(rand_recall.mean()), 3),
            "p_value": round(float((np.sum(rand_means >= vals.mean()) + 1) / (len(rand_means) + 1)), 4),
            "auc": round(float(auc), 3)}


def calibration(eng: Engine, body: str, cand: np.ndarray, n_targets: int = 60, top: int = 100) -> dict:
    """Leave-one-network-out reliability: if k of the other 4 networks put a candidate in
    their top-100, how often does the 5th (unseen) network agree?  Well calibrated
    votes give ~k/4."""
    M = eng.members
    if M < 3:
        return {}
    hits, totals = np.zeros(M), np.zeros(M)
    for t in RNG.choice(len(eng.P[body]["lat"]), n_targets, replace=False):
        prof = eng.patch_profile(body, t)
        D = eng.combine(eng.axis_distances(prof, cand), BAL)
        tops = np.zeros((M, len(cand)), bool)
        part = np.argpartition(D, top, axis=1)[:, :top]
        for k in range(M):
            tops[k, part[k]] = True
        for h in range(M):
            votes = tops[np.arange(M) != h].sum(0)
            for v in range(M):
                sel = votes == v
                hits[v] += tops[h, sel].sum()
                totals[v] += sel.sum()
    rel = [{"votes": v, "of": M - 1, "expected": round(v / (M - 1), 3),
            "observed": round(float(hits[v] / totals[v]), 3) if totals[v] else None,
            "count": int(totals[v])} for v in range(M)]
    # ECE over the bins where at least one network votes: the 0-vote bin holds millions of trivially
    # correct "no" answers and would make any ensemble look perfectly calibrated.
    voted = [r for r in rel if r["votes"] > 0 and r["observed"] is not None]
    ece = sum(abs(r["observed"] - r["expected"]) * r["count"] for r in voted) / max(sum(r["count"] for r in voted), 1)
    return {"reliability": rel, "ece": round(float(ece), 4), "ece_note": "count-weighted over bins with >= 1 vote",
            "targets": n_targets, "top": top}


def target_rediscovery(eng: Engine, body: str, cand: np.ndarray, radius: float = 250.0, weights: dict = BAL) -> dict:
    R = RADIUS_M["earth"]
    known = [(a[1], a[2]) for a in KNOWN_ANALOGS if body in a[3]]
    e = eng.P["earth"]
    near_any = np.zeros(len(cand), bool)
    for la, lo in known:
        near_any |= great_circle_km(la, lo, e["lat"][cand], e["lon"][cand], R) <= radius
    rows = []
    for t in targets_as_dicts():
        if t["body"] != body:
            continue
        i, _ = eng.nearest_patch(body, t["lat"], t["lon"])
        res = eng.match(eng.patch_profile(body, i), weights, k=20, min_sep_km=250)["results"]
        hit = [r for r in res if any(great_circle_km(r["lat"], r["lon"], la, lo, R) <= radius for la, lo in known)]
        rows.append({"target": t["name"], "hits_in_top20": len(hit),
                     "examples": [r["place"] or f"{r['lat']:.1f},{r['lon']:.1f}" for r in res[:5]]})
    rate = np.mean([r["hits_in_top20"] / 20 for r in rows])
    return {"per_target": rows, "hit_rate": round(float(rate), 3),
            "random_rate": round(float(near_any.mean()), 3),
            "lift": round(float(rate / max(near_any.mean(), 1e-9)), 2), "radius_km": radius}


def probes(eng: Engine) -> dict:
    """(a) Can a simple classifier tell WHICH BODY a patch is from? (lower = more body-agnostic)
       (b) Which hand-made terrain properties can be read linearly from the fingerprint?"""
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import cross_val_score, cross_val_predict
    n = 2500
    idx = {b: RNG.choice(len(eng.P[b]["lat"]), n, replace=False) for b in BODIES}
    emb = np.concatenate([eng.E[b][:, idx[b]].mean(0) for b in BODIES])
    raw = np.concatenate([eng.P[b]["mz"][idx[b]] for b in BODIES])
    y = np.repeat(np.arange(3), n)
    clf = LogisticRegression(max_iter=2000)
    acc_emb = cross_val_score(clf, emb, y, cv=5).mean()
    acc_raw = cross_val_score(clf, raw, y, cv=5).mean()
    r2 = {}
    for j, f in enumerate(MORPH_FEATURES):
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), emb, raw[:, j], cv=5)
        r2[f] = round(float(1 - np.mean((pred - raw[:, j]) ** 2) / np.var(raw[:, j])), 3)
    return {"body_id_accuracy_embedding": round(float(acc_emb), 3),
            "body_id_accuracy_raw_features": round(float(acc_raw), 3), "chance": 0.333, "feature_r2": r2}


def tsne_map(eng: Engine) -> dict:
    from sklearn.manifold import TSNE
    n = 900
    idx = {b: np.sort(RNG.choice(len(eng.P[b]["lat"]), n, replace=False)) for b in BODIES}
    special = []
    for a in KNOWN_ANALOGS:
        i, d = eng.nearest_patch("earth", a[1], a[2])
        if d < 80:
            special.append(("earth", i, a[0], "known"))
    for t in targets_as_dicts():
        i, _ = eng.nearest_patch(t["body"], t["lat"], t["lon"])
        special.append((t["body"], i, t["name"], "target"))
    X = np.concatenate([eng.E[b][:, idx[b]].mean(0) for b in BODIES]
                       + [eng.E[s[0]][:, [s[1]]].mean(0) for s in special])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    xy = TSNE(n_components=2, perplexity=35, init="pca", random_state=SEED, metric="cosine").fit_transform(X)
    xy = (xy - xy.mean(0)) / xy.std()
    out = {"bodies": [], "special": []}
    k = 0
    for b in BODIES:
        m = eng.P[b]["morph"][idx[b]]
        out["bodies"].append({"body": b, "idx": idx[b].tolist(), "xy": np.round(xy[k:k + n], 3).tolist(),
                              "relief": np.round(m[:, MORPH_FEATURES.index("relief_m")]).tolist(),
                              "lat": np.round(eng.P[b]["lat"][idx[b]], 2).tolist(),
                              "lon": np.round(eng.P[b]["lon"][idx[b]], 2).tolist()})
        k += n
    for s in special:
        out["special"].append({"body": s[0], "idx": int(s[1]), "name": s[2], "kind": s[3],
                               "xy": np.round(xy[k], 3).tolist()})
        k += 1
    return out


def new_candidates(eng: Engine, body: str, cand: np.ndarray, score: np.ndarray, n: int = 12) -> list[dict]:
    """Best-scoring Earth regions that are NOT within 300 km of any known analog."""
    R = RADIUS_M["earth"]
    e = eng.P["earth"]
    la, lo = e["lat"][cand], e["lon"][cand]
    far = np.ones(len(cand), bool)
    for a in KNOWN_ANALOGS:
        far &= great_circle_km(a[1], a[2], la, lo, R) > 300
    out, chosen = [], []
    for j in np.argsort(-score):
        if not far[j]:
            continue
        if chosen and great_circle_km(la[j], lo[j], la[chosen], lo[chosen], R).min() < 400:
            continue
        chosen.append(j)
        gi = int(cand[j])
        t = e["thermal"][gi]
        out.append({"lat": round(float(la[j]), 2), "lon": round(float(lo[j]), 2), "idx": gi,
                    "score": round(float(score[j]), 1), "place": eng.place_name(gi),
                    "relief_m": round(float(e["morph"][gi, 0])), "barefrac": round(float(e["barefrac"][gi]), 2),
                    "t_min_C": round(float(t[3] - 273.15)), "t_max_C": round(float(t[2] - 273.15))})
        if len(out) >= n:
            break
    return out


def build_evaluate() -> None:
    t0 = time.time()
    eng = Engine()
    cand = eng.candidates(exclude_ice=True)
    e = eng.P["earth"]
    cl, co = e["lat"][cand], e["lon"][cand]
    meths = methods(eng)
    out = {"knn": KNN, "site_radius_km": SITE_RADIUS_KM, "n_random": N_RANDOM, "n_candidates": int(len(cand)),
           "methods": [{"key": m["key"], "name": m["name"]} for m in meths], "bodies": {},
           "scale": eng.scale, "has_exposure": bool(eng.has_exposure)}
    maps = {"cand": cand}
    log = json.loads((ASSET_DIR / "training_log.json").read_text())
    out["training"] = [{"seed": m["seed"], "seconds": m["seconds"], "final_loss": m["curve"][-1][1],
                        "final_twin_acc": m["curve"][-1][2], "final_mixing": m["snapshots"][-1]["mixing"]}
                       for m in log["members"]]
    R = RADIUS_M["earth"]
    for body in ("moon", "mars"):
        print(f"== {body}: body-likeness scores ({time.time() - t0:.0f} s)", flush=True)
        known = [(a[0], a[1], a[2]) for a in KNOWN_ANALOGS if body in a[3]]
        site_nbrs = [np.where(great_circle_km(la, lo, cl, co, R) <= SITE_RADIUS_KM)[0] for _, la, lo in known]
        rand_nbrs = random_site_cache(cl, co, sum(len(nb) > 0 for nb in site_nbrs))
        scores = body_scores(eng, body, cand, meths)
        res = {}
        for m in meths:
            sc = scores[m["key"]]
            met = site_metrics_fast(sc.mean(0), site_nbrs, [k[0] for k in known], rand_nbrs)
            if sc.shape[0] > 1:
                met["score_spread_median"] = round(float(np.median(sc.std(0))), 2)
            res[m["key"]] = met
            print(f"  {m['name'][:58]:58s} mean {met['mean_site_score']:5.1f}  top10% {met['recall_top10']:.2f}"
                  f"  AUC {met['auc']:.3f}  p {met['p_value']:.4f}", flush=True)
        main_score = scores["analognet"]
        maps[f"{body}_score"] = main_score.mean(0).astype(np.float16)
        maps[f"{body}_spread"] = main_score.std(0).astype(np.float16)
        print(f"  calibration / rediscovery / new candidates ({time.time() - t0:.0f} s)", flush=True)
        out["bodies"][body] = {
            "known_sites": [k[0] for k in known],
            "methods": res,
            "calibration": calibration(eng, body, cand),
            "targets": target_rediscovery(eng, body, cand),
            "targets_landform": target_rediscovery(eng, body, cand, weights=PRESETS["landform"]["weights"]),
            "new_candidates": new_candidates(eng, body, cand, main_score.mean(0)),
        }
        print(f"  ECE {out['bodies'][body]['calibration'].get('ece')}  "
              f"target lift {out['bodies'][body]['targets']['lift']}", flush=True)
    print(f"== probes + t-SNE ({time.time() - t0:.0f} s)", flush=True)
    out["probes"] = probes(eng)
    print("  ", out["probes"], flush=True)
    out["tsne"] = tsne_map(eng)
    (ASSET_DIR / "validation.json").write_text(json.dumps(out))
    np.savez_compressed(ASSET_DIR / "analog_maps.npz", **maps)
    print(f"Done in {time.time() - t0:.0f} s.")


# ---- build results: write RESULTS.md and figures
# Step 5 - turn assets/validation.json into docs/RESULTS.md and figures in docs/img/.
#
# Every number in RESULTS.md is copied by this script from the evaluation output,
# never typed by hand, so the write-up cannot drift from the actual results.
DOCS = ROOT
IMG = STATIC_DIR / "img"
COL = {"earth": "#4c9be8", "moon": "#9a9ab0", "mars": "#e4683a", "accent": "#2a9d8f", "warn": "#dd8a2c", "grey": "#9aa3b5"}


def workflow_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.2)
    boxes = [
        (0.1, "Open data", "LOLA, Diviner (Moon)\nMOLA, MCD (Mars)\nETOPO, MODIS (Earth)", "#3b4a6b"),
        (2.25, "Same-scale patches", "132,098 patches\n118 km, 3.7 km pixels\ntangent-plane sampling", "#3b4a6b"),
        (4.4, "Neural fingerprint", "CNN x 5 (ensemble)\nself-supervised\ncontrastive (SimCLR)", COL["accent"]),
        (6.55, "4 matching axes", "terrain (neural)\nmobility | thermal\nbare ground", "#3b4a6b"),
        (8.7, "Score + uncertainty", "100 x exp(-D)\nspread & votes\nexplanations", "#3b4a6b"),
        (10.85, "Validation", "rediscover known\nanalogs vs random,\nbaseline, ablations", COL["warn"]),
    ]
    for i, (x, t, d, c) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.95, 2.4, boxstyle="round,pad=0.05", fc=c, ec="none"))
        ax.text(x + 0.975, 2.4, t, ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(x + 0.975, 1.3, d, ha="center", va="center", color="white", fontsize=8.5, linespacing=1.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 2.2, 1.55), xytext=(x + 2.0, 1.55), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
    fig.savefig(IMG / "workflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def validation_figures(V: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {m["key"]: m["name"] for m in V["methods"]}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, body in zip(axes, ("mars", "moon")):
        M = V["bodies"][body]["methods"]
        keys = [k for k in names if k in M]
        vals = [M[k]["mean_site_score"] for k in keys]
        cols = [COL["accent"] if k == "analognet" else COL["warn"] if k.startswith("baseline") else COL["grey"] for k in keys]
        ax.barh([names[k] for k in keys], vals, color=cols)
        ax.axvline(M["analognet"]["random_mean"], ls="--", color="#c53030", label="random land sites")
        for i, k in enumerate(keys):
            ax.text(vals[i] + 0.5, i, f"{vals[i]:.1f} (AUC {M[k]['auc']:.2f})", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_title(f"{body.title()}: mean score of known analog sites", fontsize=11)
        ax.set_xlabel("site score (percentile of best patch within 120 km)")
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "validation_methods.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, body in zip(axes, ("mars", "moon")):
        cal = V["bodies"][body].get("calibration")
        if not cal:
            continue
        rel = [r for r in cal["reliability"] if r["observed"] is not None]
        ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
        ax.plot([r["expected"] for r in rel], [r["observed"] for r in rel], "o-", color=COL["accent"], label="ensemble votes")
        ax.set_title(f"{body.title()}: calibration (ECE {cal['ece']:.3f})", fontsize=11)
        ax.set_xlabel("share of other networks voting 'top 100'")
        ax.set_ylabel("held-out network agrees")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "calibration.png", dpi=140)
    plt.close(fig)
    log = json.loads((ASSET_DIR / "training_log.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, m in enumerate(log["members"]):
        c = np.array(m["curve"])
        ax.plot(c[:, 0], c[:, 1], lw=1.4, label=f"network {k + 1}")
    ax.set_xlabel("training step")
    ax.set_ylabel("contrastive loss")
    ax.legend(fontsize=8)
    ax.set_title("Training curves of the 5 ensemble networks", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "training.png", dpi=140)
    plt.close(fig)


def build_results() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    workflow_figure()
    V = json.loads((ASSET_DIR / "validation.json").read_text())
    validation_figures(V)
    names = {m["key"]: m["name"] for m in V["methods"]}
    L = ["# THEIA (AnalogNet model) - results", "",
         "*Generated automatically by `python streamlit_app.py build results` from `assets/validation.json`. Do not edit by hand.*", "",
         "## Set-up", "",
         f"- Earth candidates: {V['n_candidates']:,} land patches (ice sheets excluded), 118 km footprint, 3.7 km pixels.",
         f"- Body-likeness of an Earth patch = mean distance to its {V['knn']} nearest Moon/Mars patches (8,000-patch reference sample), "
         "turned into a percentile.",
         f"- **Site score** of a known analog = best percentile within {V['site_radius_km']:.0f} km of the site. "
         f"Random baseline and p-value from {V['n_random']:,} random sets of land sites of the same size.",
         "- Known analog sites were never used for training, for choosing weights or for any design decision.", "",
         "![workflow](static/img/workflow.png)", "",
         "## Training", "",
         "| Network | Seed | Minutes (CPU) | Final loss | Twin found | Body mixing |", "|---|---|---|---|---|---|"]
    for k, t in enumerate(V["training"]):
        L.append(f"| {k + 1} | {t['seed']} | {t['seconds'] / 60:.1f} | {t['final_loss']:.3f} | {t['final_twin_acc'] * 100:.0f}% | {t['final_mixing']:.2f} |")
    L += ["", "*Twin found*: share of augmented views for which the network picks the true partner out of 2x256-1 candidates "
          "(chance below 0.4 %). *Body mixing*: 1 = Earth/Moon/Mars neighbours perfectly interleaved, 0 = separate islands.", "",
          "![training](static/img/training.png)", ""]
    for body in ("mars", "moon"):
        B = V["bodies"][body]
        M = B["methods"]
        L += [f"## {body.title()} analogs ({len(B['known_sites'])} known sites)", "",
              "| Method | Mean site score | Median | In top 10 % | In top 5 % | AUC | p vs random |",
              "|---|---|---|---|---|---|---|"]
        for k, m in M.items():
            L.append(f"| {names[k]} | {m['mean_site_score']} | {m['median_site_score']} | {m['recall_top10'] * 100:.0f}% | "
                     f"{m['recall_top5'] * 100:.0f}% | {m['auc']} | {m['p_value']} |")
        L.append(f"| *Random land sites (expected)* | {M['analognet']['random_mean']} | - | "
                 f"{M['analognet']['random_recall_top10'] * 100:.0f}% | - | 0.5 | - |")
        L += ["", "Per-site scores (AnalogNet / raw-feature baseline):", ""]
        base = {s["name"]: s["score"] for s in M["baseline_raw"]["sites"]}
        for s in M["analognet"]["sites"]:
            L.append(f"- {s['name']}: {s['score']} / {base.get(s['name'])}")
        T = B["targets"]
        L += ["", f"**Per-target check:** of the top-20 Earth matches of each {body} target, {T['hit_rate'] * 100:.0f}% lie within "
              f"{T['radius_km']:.0f} km of a known {body} analog, vs {T['random_rate'] * 100:.0f}% of random land patches "
              f"(**{T['lift']}x lift**, balanced preset)."]
        if B.get("targets_landform"):
            TL = B["targets_landform"]
            L.append(f"With the *landform (shape only)* preset: {TL['hit_rate'] * 100:.1f}% (**{TL['lift']}x lift**).")
        L.append("")
        if B.get("calibration"):
            rel = ", ".join(f"{r['votes']}/{r['of']} votes -> {r['observed'] * 100:.0f}% (ideal {r['expected'] * 100:.0f}%)"
                            for r in B["calibration"]["reliability"] if r["votes"] > 0)
            L.append(f"**Calibration:** expected calibration error of the ensemble votes = {B['calibration']['ece']} "
                     f"({B['calibration']['targets']} random targets, leave-one-network-out, bins with at least one vote). "
                     f"Held-out agreement: {rel}. The votes are slightly over-confident.")
        L += ["", "**New candidate analogs** (best-scoring regions at least 300 km from any known analog):", "",
              "| # | Place | Lat, Lon | Score | Relief | Bare | T min / max |", "|---|---|---|---|---|---|---|"]
        for i, c in enumerate(B["new_candidates"], 1):
            L.append(f"| {i} | {c['place']} | {c['lat']}, {c['lon']} | {c['score']:.2f} | {c['relief_m']} m | "
                     f"{c['barefrac'] * 100:.0f}% | {c['t_min_C']} / {c['t_max_C']} C |")
        if B.get("new_candidates_landform"):
            L += ["", "*Shape-only (neural landform) candidates* - added after seeing the validation (best method for lunar "
                  "analogs); blind to environment, so some are vegetated:", "",
                  "| # | Place | Lat, Lon | Score | Relief | Bare |", "|---|---|---|---|---|---|"]
            for i, c in enumerate(B["new_candidates_landform"], 1):
                L.append(f"| {i} | {c['place']} | {c['lat']}, {c['lon']} | {c['score']:.2f} | {c['relief_m']} m | {c['barefrac'] * 100:.0f}% |")
        L.append("")
    P = V["probes"]
    L += ["![methods](static/img/validation_methods.png)", "", "![calibration](static/img/calibration.png)", "",
          "## What the network learned", "",
          f"- Body identity from the fingerprint (logistic regression, 5-fold CV): **{P['body_id_accuracy_embedding'] * 100:.0f}%** "
          f"vs {P['body_id_accuracy_raw_features'] * 100:.0f}% from raw terrain features (chance 33%).",
          "- Hand-made terrain properties readable from the fingerprint (cross-validated R2 of a linear read-out):", ""]
    for f, r2 in sorted(P["feature_r2"].items(), key=lambda x: -x[1]):
        L.append(f"  - {f}: {r2}")
    L += ["", "## How to read these results honestly", "",
          "- A mean site score well above the random value, with a small p-value, means the method ranks the places experts use "
          "as analogs higher than chance. AUC 0.5 = chance, 1.0 = perfect.",
          "- Compare AnalogNet with the raw-feature baseline and with *neural terrain only*: this shows what the neural "
          "fingerprint adds and what the other axes add. If a simpler method scores as well, say so.",
          "- Known analogs were chosen by experts for many reasons (geology, access, history). A perfect score is neither expected nor desirable.",
          "- The bare-ground axis rewards deserts and polar deserts. Most known analogs are deserts, so check the "
          "*without bare-ground* ablation to see how much of the result comes from this single, physically motivated axis.", ""]
    (DOCS / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", DOCS / "RESULTS.md")


# ---- build zip: build the release zip
# Build the distributable zip:  python streamlit_app.py build zip
#
# Contains the source code, the trained ensemble + processed open data (assets/),
# the offline web UI with CesiumJS/Plotly, docs and the one-click launchers.
# Leaves out the raw downloads, the virtual environment and git history.
sys.path.insert(0, str(ROOT))

TOP_SKIP = {".venv", ".git", "data", "build", "dist"}      # only at the repository root
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".idea", ".vscode"}            # anywhere
SKIP_SUFFIX = {".pyc", ".zip", ".part"}


def build_zip() -> None:
    out = ROOT / "dist" / f"THEIA_v{__version__}.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if rel.parts[0] in TOP_SKIP or any(part in SKIP_DIRS for part in rel.parts) or p.is_dir() \
                    or p.suffix in SKIP_SUFFIX:
                continue
            # the app reads globe tiles only; the single-image source textures are rebuilt by scripts 02d/02e
            if rel.parts[:2] == ("static", "textures") and len(rel.parts) == 3 and p.suffix == ".jpg":
                continue
            info = zipfile.ZipInfo.from_file(p, f"THEIA/{rel.as_posix()}")
            if p.name.endswith(".sh"):
                info.external_attr = 0o755 << 16          # keep the Linux/macOS launcher executable
            info.compress_type = zipfile.ZIP_STORED if p.suffix in {".npz", ".jpg", ".png", ".pt"} else zipfile.ZIP_DEFLATED
            with open(p, "rb") as f:
                z.writestr(info, f.read())
            n += 1
    print(f"{out}  ({out.stat().st_size / 1e6:.0f} MB, {n} files)")


# ==================================================================================================
# 16. ENTRY POINT - website under Streamlit, build commands from the terminal
# ==================================================================================================

BUILD_STEPS = {"download": build_download, "modis": build_modis, "dataset": build_dataset, "names": build_names, "surface": build_surface, "texture": build_texture, "hologram": build_hologram, "tiles": build_tiles, "train": build_train, "evaluate": build_evaluate, "results": build_results, "zip": build_zip}


def cli() -> None:
    ap = argparse.ArgumentParser(description="THEIA build pipeline. Run the website with: python streamlit_app.py  (no arguments)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""steps (in order):
    download  download the raw open data (~3.5 GB, internet once)
    modis     MODIS temperature + NDVI from Microsoft Planetary Computer
    dataset   global grids, patches and features
    names     offline place names for Earth patches
    surface   Earth temperature and bare ground from MODIS
    texture   true-colour Earth globe texture
    hologram  hologram globe textures
    tiles     cut globe textures into tiles (fast zoom)
    train     train the 5-network ensemble
    evaluate  validation against known analogs
    results   write RESULTS.md and figures
    zip       build the release zip
    all       every step above, in order""")
    ap.add_argument("build", choices=["build"])
    ap.add_argument("step", choices=list(BUILD_STEPS) + ["all"])
    ap.add_argument("--quick", action="store_true", help="train: 2 networks x 400 steps (for testing)")
    a = ap.parse_args()
    steps = list(BUILD_STEPS) if a.step == "all" else [a.step]
    for s in steps:
        print(f"== build {s}", flush=True)
        BUILD_STEPS[s](quick=a.quick) if s == "train" else BUILD_STEPS[s]()


def _inside_streamlit() -> bool:
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


if _inside_streamlit():
    run_app()
elif __name__ == "__main__":
    cli()
