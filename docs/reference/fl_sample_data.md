# A sample of reported cases of forced labor

Description

## Usage

``` r
fl_sample_data
```

## Format

A tibble with 10,127 rows and 31 columns corresponding to the features
of the Global Fishing Watch's forced labor risk model:

- ssvid:

  The source-specific vessel identifier, here corresponding to the MMSI
  (Maritime Mobile Service Identity) of the vessel

- indID:

  A unique identity for each vessel-year

- known_offender:

  Whether the vessel-year combination is known to have engaged in forced
  labor practices

- source_id_number:

  An identity number for each forced labor report

- gear:

  The geartype used by the vessel

- engine_power_kw:

  The engine power of the vessel in kilowatts

- tonnage_gt:

  The tonnage of the vessel in giga tonnes

- length_m:

  The length of the vessel in meters

- position_messages:

  The number of position messages detected by AIS

- hours:

  The number of hours at sea recorded by AIS

- fishing_hours:

  The number of hours recorded by AIS per vessel-year where the vessel
  appears to be fishing

- average_daily_fishing_hours:

  The average number of hours per day where the vessel appears to be
  fishing

- fishing_hours_foreign_eez:

  The number of hours per day recorded by AIS per where the vessel
  appears to be fishing in EEZs that do not match the flag the vessel
  flies

- fishing_hours_high_seas:

  The number of hours per day recorded by AIS per where the vessel
  appears to be fishing on the high seas

- max_distance_from_shore_km:

  The maximum distance from shore detected via AIS

- max_distance_from_port_km:

  The maximum distance from port detected via AIS

- number_encounters:

  The number of encounters detected via AIS. Encounter events describe
  when AIS data shows two vessels that appear to be meeting at sea and
  can be indicative of potential transshipment events

- number_forced_labor_encounters:

  The number of encounters detected via AIS involving a vessel with a
  history of forced labor

- average_encounter_duration_hours:

  The average duration of encounters in hours

- gaps_12_hours:

  The number of gaps in AIS that are greater than 12 hours

- average_off_distance_from_port_km:

  The average distance from port during gaps in AIS that are greater
  than 12 hours

- average_off_distance_from_shore_km:

  The average distance from shore during gaps in AIS that are greater
  than 12 hours

- average_gap_days:

  The average length of AIS gaps (in days)

- average_gap_km:

  The average distance between the location where an AIS gap in
  transmission of at least 12 hours begins and the location where the
  gap ends

- number_foreign_port_visits:

  The number of visits to ports, where the Port State is different to
  the vessel's Flag State

- number_loitering_events:

  The number of loitering events detected via AIS. Loitering events are
  recorded when one vessel shows signs of potential encounters, or
  meeting another vessel at sea, but there is no second vessel detected

- average_loitering_duration_hours:

  The average duration of loitering events in hours

- average_voyage_duration_hours:

  The time at sea between port visits

- number_voyages:

  The number of voyages (time at sea between port visits) in a given
  year

- known_non_offender:

  Whether the vessel-year combination is know to have been deemed as
  free of forced labor via a labor-related inspection - Used only in the
  holdout set

@source description, link to the preprint
