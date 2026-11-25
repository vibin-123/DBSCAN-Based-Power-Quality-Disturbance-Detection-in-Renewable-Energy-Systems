# Dataset verification record

Checked on 10 November 2026 by downloading from the original sources.
Nothing in this project is simulated.

## Sources investigated

| Source | What was actually found | PQ | Renewables | Timestamps | Licence | Decision |
|---|---|---|---|---|---|---|
| **SPARK**, KIT, DOI [10.35097/bjdg3m3rg5jv3skk](https://doi.org/10.35097/bjdg3m3rg5jv3skk) | One 76.2 GB BagIt tar holding 26 zips: 22 machine meters, `EPI_TotalLoad` (facility connection point, 190 channels, 2018-2024), `EPI_PV` (rooftop PV, 2 channels, 2024), metadata, exogenous data, validation files | Yes: three-phase U, I, f, THD-U, THD-I, odd harmonics 3-31, P, Q, S, PF at 5 s | Yes: PV AC power at 5 s (1 Jan - 9 Apr 2024 only); regional generation mix at 15 min | UTC, 5 s | CC BY 4.0 | **Used** |
| **SMARD**, Bundesnetzagentur, chart-data API | Quarter-hourly solar, onshore wind and grid load for the TransnetBW control area, 35,136 rows each for 2024 | No | Yes | UTC, 15 min | CC BY 4.0 | **Used** (grid load; cross-check of solar) |
| UK Power Networks "Power Quality" | 20,750 records; fields: site, transformer, phase, harmonic, latest, latestweek, highest, highestweek, sitename, coordinates. Weekly 95th-percentile harmonic values, one "latest" and one "highest" per site/phase/harmonic | Harmonics only | No | ISO week only | CC BY 4.0 | Rejected: no time series, no generation. Metadata and column names were retrieved through the API; the anonymous CSV export returns the header row only, so the 20,750 records need a registered UKPN account and were not downloaded |
| Mendeley "PQ Disturbances Dataset" v4, DOI 10.17632/nkdpg8mn4f.4 | 15 xlsx files. `PQ Dataset.xlsx` is 775 x 79: wavelet statistics of MATLAB-generated signals, 12 labelled classes | Synthetic features | No | None | CC BY 4.0 | Rejected: synthetic, no timestamps, no generation |
| SEED_PQD_v1, Zenodo 11843312 | 22 MB zip: 17 CSVs of 1000 x 100 samples (one 50 Hz cycle at 5 kHz, amplitude scaled to +/-1), 17 labelled classes | Synthetic waveforms | No | None | CC BY 4.0 | Rejected: synthetic, no timestamps, no generation |
| IEEE PES Open Data entry d866dd02-... | Catalogue entry "IEEEPESOD" pointing to the IEEE PES Intelligent Systems Subcommittee collection (consumption, EV, wind, PV and power-quality data as separate xlsx sets; licence "N/A") | Separate set | Separate sets | Varies | Not stated | Rejected: the PQ, PV and wind sets are unrelated sites and periods; no licence |

The rejected files that could be downloaded are kept under
`data/raw/reference_sources/` as evidence of the inspection.

## Why SPARK + SMARD can be combined

- **Same clock.** Both are in UTC. The SPARK PV series correlates best with
  regional solar generation at zero lag (r = 0.76 at 15-min resolution; r = 0.68
  and 0.74 at -1 h and +1 h), which confirms the SPARK timestamps are UTC.
- **Same source for the regional mix.** The solar series downloaded directly
  from SMARD is identical to the copy shipped inside SPARK (maximum absolute
  difference 0.0 MW over 35,136 rows).
- **Geography.** KIT Campus North (Karlsruhe) lies in the TransnetBW control
  area. The regional series describe the grid the facility is connected to,
  not the facility itself; they are identical for every site in the area.
- **Resolution.** 5 s measurements are aggregated to 10-minute windows
  (IEC 61000-4-30 interval); each window takes the value of the 15-minute
  interval that contains its start. No value is interpolated.

## Verified properties of the files used

| File group | Rows | Columns | Missing | Sampling |
|---|---|---|---|---|
| 26 x `EPI_TotalLoad/<m>/2024_<m>.csv.xz` | 6,324,480 each | `WsDateTime`, `<m>` | 31.7 % | 5 s grid; the meter reports about every 6.5 s, so every fourth slot is empty |
| `EPI_PV/AC_ActivePower/2024_AC_ActivePower.csv.xz` | 6,324,480 | `WsDateTime`, `AC_ActivePower` (W) | 76.5 % | 5 s; data from 1 Jan to 9 Apr 2024 |
| `electricity_emissions_15min_transnetbw_2018_2024.csv` | 245,472 | `utc_datetime`, 10 technologies (MW), emission intensity | nuclear 13 % (after shut-down) | 15 min |
| `smard_grid_load_TransnetBW_2024_15min.csv` | 35,136 | `utc_datetime`, `grid_load_mw` | 0 | 15 min |

Integrity: the three small SPARK zips match the MD5 values in the archive's
BagIt manifest; each measurement file passed the zip CRC-32 check on
extraction; SHA-256 of every raw file is in `data/raw/MANIFEST.csv`.

## Limits that follow from the data

- There is no wind generation at the site. Wind enters only as regional
  onshore wind in the TransnetBW area.
- Local PV data cover 14 winter and early-spring weeks only.
- Values are 5 s RMS aggregates, not waveforms: sub-cycle dips, swells and
  transients cannot be seen.
- The published "clean" SPARK version has already had physically implausible
  samples removed by its authors.
- The SPARK documentation does not state whether the PV feeds in behind or in
  front of the `EPI_TotalLoad` meter.
