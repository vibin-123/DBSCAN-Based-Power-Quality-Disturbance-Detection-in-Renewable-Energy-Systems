# Preprocessing report

Year: 2024. PQ meter: `EPI_TotalLoad` (SPARK). All timestamps are UTC.

## Power-quality channels (5 s)

| channel | rows | duplicates | off-grid | missing (raw) | out of physical range | missing (final) | availability |
|---|---|---|---|---|---|---|---|
| U1 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| U2 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| U3 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| U12 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| U23 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| U31 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| Freq | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| I1 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| I2 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| I3 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| IN | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| THD_U1 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| THD_U2 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| THD_U3 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| THD_I1 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| THD_I2 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| THD_I3 | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| P_total | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| Q_total | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| S_total | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| PF_total | 6,324,480 | 0 | 0 | 2,004,023 | 0 | 2,004,023 | 68.3% |
| U1_h3 | 6,324,480 | 0 | 0 | 2,004,045 | 0 | 2,004,045 | 68.3% |
| U1_h5 | 6,324,480 | 0 | 0 | 2,004,045 | 0 | 2,004,045 | 68.3% |
| U1_h7 | 6,324,480 | 0 | 0 | 2,004,045 | 0 | 2,004,045 | 68.3% |
| U2_h5 | 6,324,480 | 0 | 0 | 2,004,045 | 0 | 2,004,045 | 68.3% |
| U3_h5 | 6,324,480 | 0 | 0 | 2,004,045 | 0 | 2,004,045 | 68.3% |

Share of 5 s samples with all channels present, by month: 1: 66%, 2: 73%, 3: 68%, 4: 76%, 5: 76%, 6: 76%, 7: 76%, 8: 76%, 9: 77%, 10: 76%, 11: 77%, 12: 3%

## PV active power (5 s)

Rows 6,324,480; duplicates 0; missing 4,838,392; out of range 0. 
Valid PV data span: 2024-01-01 00:00:00 to 2024-04-09 00:00:00.

## Regional generation mix and load (15 min)

```json
{
 "rows": 35136,
 "duplicates": 0,
 "missing_share": {
  "nuclear": 0.9194,
  "hydro": 0.0,
  "wind_onshore": 0.0,
  "solar_pv": 0.0,
  "biomass": 0.0,
  "other_renewables": 0.0,
  "natural_gas": 0.0001,
  "hard_coal": 0.0,
  "other_conventional": 0.0001,
  "pumped_storage_generation": 0.0
 },
 "smard_vs_spark_solar_max_abs_diff_mw": 0.0,
 "grid_load_missing": 0
}
```

## Handling rules

- Duplicated timestamps: first occurrence kept.
- Missing 5 s samples: left missing, never interpolated. The meter reports about every 6.5 s, so on the 5 s grid every fourth slot is structurally empty and a complete 10-minute window has ~76.7 % of slots filled. A window is used only if at least 60% of its slots are valid in every channel (about 78 % of the achievable samples).
- Out-of-physical-range samples: set to missing and counted above.
- Units: P, Q, S converted from W/var/VA to kW/kvar/kVA; PV from W to kW.
- Nuclear generation missing after April 2023 set to 0 MW (plant shutdown).
- 15-minute regional values are assigned to 10-minute windows by a backward as-of join (the 15-min interval that contains the window start).