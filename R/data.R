#' A sample of reported cases of forced labor
#'
#' Description
#'
#' @format ## `new_training_data`
#' A tibble with 10,072 rows and 30 columns:
#' \describe{
#'  \item{known_offender}{Whether the vessel-year combination is know to have
#'  engaged in forced labor practices}
#'  \item{known_non_offender}{Whether the vessel-year combination is know to
#'  have been deemed as free of forced labor via a labor-related inspection}
#'  \item{indID}{A unique identity for each vessel-year}
#'  \item{source_id_number}{An identity number for each forced labor report}
#'  \item{gear}{The geartype used by the vessel}
#'  \item{engine_power_kw}{The engine power of the vessel in kilowatts}
#'  \item{tonnage_gt}{The tonnage of the vessel in giga tonnes}
#'  \item{length_m}{The length of the vessel in meters}
#'  \item{position_messages}{The number of position messages detected by AIS}
#'  \item{hours}{The number of hours recorded by AIS}
#'  \item{fishing_hours}{The number of hours recorded by AIS per vessel-year
#'  where the vessel appears to be fishing}
#'  \item{average_daily_fishing_hours}{The average number of hours per day
#'  recorded by AIS where the vessel appears to be fishing}
#'  \item{fishing_hours_foreign_eez}{The number of hours per day recorded by AIS
#'  per where the vessel appears to be fishing in EEZs that do not
#'  match the flag the vessel flies}
#'  \item{fishing_hours_high_seas}{The number of hours per day recorded by AIS
#'  per where the vessel appears to be fishing on the high seas}
#'  \item{max_distance_from_shore_km}{The maximum distance from shore detected
#'  via AIS}
#'  \item{max_distance_from_port_km}{The maximum distance from port detected
#'  via AIS}
#'  \item{number_encounters}{The number of encounters detected via AIS. Note,
#'  encounter events describe when AIS data shows two vessels that appear to be
#'  meeting at sea. Encounter events can be indicative of potential
#'  transshipment events}
#'  \item{number_forced_labor_encounters}{The number of encounters detected via
#'  AIS involving a vessel with a history of forced labor}
#'  \item{average_encounter_duration_hours}{The average duration of encounters
#'  in hours}
#'   \item{gaps_12_hours}{The number of gaps in AIS that are greater than 12
#'   hours}
#'   \item{average_off_distance_from_port_km}{The average distance from port
#'   during gaps in AIS that are greater than 12 hours}
#'   \item{average_off_distance_from_shore_km}{The average distance from shore
#'   during gaps in AIS that are greater than 12 hours}
#'   \item{average_gap_days}{The average length of AIS gaps (in days)}
#'   \item{average_gap_km}{The average distance (in a straight line projection)
#'   between the location where an AIS gap in transmission of at least 12 hours
#'   begins and the location where the gap ends}
#'   \item{number_foreign_port_visits}{The number of visits to ports, where the
#'   Port State is different to the vessel's Flag State}
#'   \item{number_loitering_events}{The number of loitering events detected
#'   via AIS. Loitering events are recorded when one vessel shows signs of
#'   potential encounters, or meeting another vessel at sea, but there is no
#'   second vessel detected}
#'   \item{average_loitering_duration_hours}{The average duration of loitering
#'   events in hours}
#'   \item{average_voyage_duration_hours}{The time at sea between port visits}
#'   \item{number_voyages}{The number of voyages (time at sea between port
#'   visits) in a given year}
#'   \item{flag_region}{The region the flag is associated with
#'   (Asian or non-Asian). This is used later for performing fairness tests}
#'  }
#'
#'  @source description, link to the preprint
#'
