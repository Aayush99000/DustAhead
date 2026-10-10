# DustAhead — HaboobCast

**Nowcasting dangerous dust storms with deep learning.**

HaboobCast predicts blowing dust and haboobs **15–60 minutes ahead** along Interstate 10 in Arizona and New Mexico, using only free public data. Most current tools detect dust once it is already visible. HaboobCast tries to predict dust that has *not formed yet* by reading the parent thunderstorm, the wind and the state of the ground.

> ⚠️ **Research prototype, not an official warning.** Nothing produced by this project replaces National Weather Service (NWS) Dust Storm Warnings.

This started as a CS 7150 Deep Learning project (Northeastern University, Fall 2026) and continues as a research program through July 2027.

---

## Why it matters

- A study merging FARS crash records with NOAA Storm Data counted **232 dust-related road deaths** in the US from 2007 to 2017 (about 21 per year).
- I-10 contains the two deadliest dust zones in the US: the "Deadliest 10 Miles" near Picacho Peak, AZ, and Lordsburg Playa, NM.
- About 42,000 vehicles a day pass the deadliest Arizona marker, and trucks make up as much as 40% of traffic on some sections.

A 30-minute dust event at Picacho exposes roughly `42,000 / 24 × 0.5 ≈ 875` vehicles, about 350 of them heavy trucks. A 30-minute warning gives traffic operators time to slow or hold them.

## Research question

> Can a nowcaster that combines satellite storm signals (cloud-top cooling, lightning), weather-model wind and the daily erodibility of the ground predict **newly forming** dust better than methods that only move existing dust forward? And can pretraining on plentiful Gulf and Sahara dust events make it work with few US events?

### What is new

1. **Genesis vs. advection evaluation.** Scores the prediction of new dust separately from moving dust that is already visible.
2. **Physically grounded surface inputs.** Soil moisture, rain history, bare fields and terrain act as a gate on where wind can lift dust.
3. **Impact-weighted scoring.** Forecasts are scored by the number of vehicles actually warned, not only by pixels.
4. **Cross-region transfer.** The model moves from data-rich dust regions to a data-poor one, with a size-matched control.

### Hypotheses

| ID | Phase | Hypothesis | Refuted if |
|---|---|---|---|
| H1 Genesis | 1 | Convective precursors (IR cooling, GLM lightning, HRRR wind/outflow) raise 30–60 min skill on newly forming dust | Genesis-stratified station CSI not higher than dust-only model (event-bootstrap 95% CI includes zero) |
| H3 Impact | 1 | Calibrated forecasts warn earlier than NWS at an acceptable false alarm rate | Median lead over NWS ≤ 0, or FAR > 0.5 |
| H4 Surface state | 1 | Ground-state inputs cut false alarms on stormy days without dust | FAR on null days does not drop |
| H2a Label efficiency | 2 | Self-supervised pretraining on unlabeled GOES helps with few labels | No gain over scratch at 10 and 20 events |
| H2b Cross-region | 3 | Gulf/Sahara pretraining beats an equal-size US control | No gain over the size-matched control |
| H5 Generalization | 3 | Terrain and source inputs help in an unseen region | Skill on held-out El Paso / West Texas drops |

---

## Physical basis

Dust lifts only when the friction velocity exceeds a threshold that rises with soil moisture:

$$u_* = \frac{\kappa\,U(z)}{\ln(z/z_0)}, \qquad u_{*t}(w) = u_{*t,\text{dry}}\cdot f_w$$

Here $f_w$ is the Fécan soil-moisture correction used in the FENGSHA scheme. Recent rain can nearly double the wind needed to raise dust, which is why some storms make haboobs and others do not. The model receives a soft **erodibility prior** as an input channel:

$$E = \sigma\!\left(\frac{u_* - u_{*t}(w)}{s}\right)$$

## Data

All data is free and public. The study area is about 1,000 × 800 km along I-10, reprojected with Satpy to a common **4 km equal-area grid (250 × 200 cells)**.

| Source | Use | Access |
|---|---|---|
| GOES-18 ABI (CONUS, 5 min) | Main satellite input (bands 2, 7, 11, 13, 14, 15) | AWS `noaa-goes18` |
| GOES-18 GLM | Lightning flash counts | AWS `noaa-goes18` |
| GOES-18 ADP | Weak dust labels; baseline | AWS `noaa-goes18` |
| HRRR analyses | Wind, gust, CAPE, DCAPE, RH, soil moisture, rain | AWS `noaa-hrrr-bdp-pds` (via Herbie) |
| ASOS / METAR | Ground-truth station labels | Iowa Environmental Mesonet |
| NWS Dust Storm Warnings | Event list; lead-time comparison | Iowa Environmental Mesonet |
| USGS 3DEP, NLCD, USDA CDL, SoilGrids | Terrain, land cover, fallow fields, soil texture | One-time downloads |
| FHWA HPMS | Traffic volume and truck share (post-prediction only) | One-time download |
| *Phase 2:* GOES-17, MRMS, MODIS NDVI | Extra events, outflow boundaries, vegetation | AWS / NASA Earthdata |
| *Phase 3:* Meteosat SEVIRI, ERA5 | Gulf and Sahara pretraining | EUMETSAT / ECMWF |

**Station label:** visibility ≤ 1 mile **and** a dust code (DU, BLDU, DS, SS) **and** no rain codes. Visibility ≤ ¼ mile is reported separately as severe.

**Splits:** train on 2023–2024, validate on 2025, test on 2026, split by event and date. The test set is evaluated once, after a logged freeze date.

---

## Model

The task is framed as **spatiotemporal probabilistic segmentation**: from the last 30 minutes of observations (62 input channels), predict P(dust) per grid cell at **+15, +30 and +60 min**.

```
Satellite (ABI + GLM)   HRRR (wind, CAPE, soil, rain)   Ground layers (terrain, soil, land cover)
        │                         │                                  │
   Dust stream            Convection stream                 Surface encoder → gate
        └───────────── Fusion: U-Net (Ph. 1) / Earthformer (Ph. 2) ──┘
                                      │
                   Calibrated dust probability maps (+15/+30/+60)
                    │                 │                     │
            Station forecasts   Road risk per mile ← Traffic (HPMS, after prediction only)
                    └──────── Dashboard, replay, alerts ────┘
```

The surface gate allows dust only where the ground can supply it:

$$\hat{P}_\tau(r) = \sigma\big(z_\tau(r)\big)\cdot\sigma\big(h(G, E_t, H_t)(r)\big)$$

## Key references

- Ronneberger et al. (2015), *U-Net*. arXiv:1505.04597
- Gao et al. (2022), *Earthformer*. arXiv:2207.05833
- He et al. (2022), *Masked Autoencoders*. arXiv:2111.06377
- Pulkkinen et al. (2019), *pySTEPS*. doi:10.5194/gmd-12-4185-2019
- Roberts & Lean (2008), *Fractions Skill Score*. doi:10.1175/2007MWR2123.1
- Fécan et al. (1999), *Soil-moisture erosion threshold*. doi:10.1007/s00585-999-0149-7
- Guo et al. (2017), *On calibration of modern neural networks*. arXiv:1706.04599
- *How many people were killed by windblown dust events in the United States?* (2023), BAMS. doi:10.1175/BAMS-D-22-0186.1

The full reference list is in the project proposal.

## License

To be decided.
