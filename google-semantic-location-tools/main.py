import argparse
import csv
import json
import logging
import os
import typing
from datetime import datetime, timedelta
import dataclasses

import dateutil.parser
import dateutil.tz

from geopy.geocoders import Bing

logger = logging.getLogger()


PlaceCoordinates = typing.Tuple[float, float]
PlaceId = str


@dataclasses.dataclass(frozen=True)
class PlaceVisit:
    id: PlaceId
    start_time: datetime
    end_time: datetime
    latitude: float
    longitude: float

    def coordinates(self) -> PlaceCoordinates:
        return (self.latitude, self.longitude)

    def duration(self) -> timedelta:
        duration = self.end_time - self.start_time
        duration_quantized = timedelta(days=duration.days, seconds=duration.seconds)
        return duration_quantized


def semantic_history_rank_filename_by_month_and_year(filename):
    try:
        dt = datetime.strptime(filename, "%Y_%B.json")
        val = dt.timestamp()
        return val
    except ValueError:
        return -1


def semantic_history_get_ordered_files(base_dir: str):
    for root, dirs, files in os.walk(base_dir, topdown=True):
        if dirs:
            dirs.sort()
        if files:
            file_indices_to_remove = [
                i for i in range(len(files)) if os.path.splitext(files[i])[1] != ".json"
            ]
            for i in file_indices_to_remove:
                files.pop(i)
            files.sort(key=semantic_history_rank_filename_by_month_and_year)
            for f in files:
                yield os.path.join(root, f)


def semantic_history_extract_records(file_path: str) -> typing.List[PlaceVisit]:
    _DATE_FMT = "%y-%m-%d %H:%M"
    place_visits = []
    location_id_to_coords: typing.Dict[str, PlaceCoordinates] = dict()

    with open(file_path, "r") as f:
        timeline_json = json.load(f)
        timeline_records = timeline_json["timelineObjects"]
        for record in timeline_records:
            record_type, record = next(iter(record.items()))
            record_duration = record["duration"]
            start_time = dateutil.parser.isoparse(record_duration["startTimestamp"])
            end_time = dateutil.parser.isoparse(record_duration["endTimestamp"])
            duration = end_time - start_time
            quantized_duration = timedelta(days=duration.days, seconds=duration.seconds)

            if record_type == "placeVisit":
                location = record["location"]
                location_id = location.get("placeId")
                location_lat = location.get("latitudeE7")
                location_lon = location.get("longitudeE7")
                if (
                    location_id is None
                    and location_lat is None
                    and location_lon is None
                ):
                    logger.warning(f"Invalid location for record {record}")
                    continue

                # If id is missing then make one from the coordinates
                location_id = (
                    location_id if location_id else f"{location_lat}.{location_lon}"
                )

                # If coordinates are missing then use id to lookup
                location_coords = None
                if location_lat is None and location_lon is None:
                    location_coords = location_id_to_coords.get(location_id)
                else:
                    location_coords = (location_lat / 1e7, location_lon / 1e7)
                    location_id_to_coords[location_id] = location_coords

                # Purely for debugging
                if logger.level >= logging.DEBUG:
                    location_name = location.get("name")
                    location_address = location.get("address", "").replace(",", "")
                    logger.debug(
                        f"{record_type}, {location_name}, {location_id}, {location_address}, {start_time.strftime(_DATE_FMT)} -> {end_time.strftime(_DATE_FMT)}, Duration: {quantized_duration}"
                    )

                entry = PlaceVisit(
                    id=location_id,
                    start_time=start_time,
                    end_time=end_time,
                    latitude=location_coords[0],
                    longitude=location_coords[1],
                )
                place_visits.append(entry)
            elif record_type == "activitySegment":
                logger.debug(
                    f"{record_type}, {start_time.strftime(_DATE_FMT)} -> {end_time.strftime(_DATE_FMT)}, Duration: {quantized_duration}"
                )
    return place_visits


def semantic_history_extract_place_visits(source_dir: str, output_file: str):
    num_rows = 0
    with open(output_file, "w", newline="\n") as output_csv:
        output_field_names = [field.name for field in dataclasses.fields(PlaceVisit)]
        csv_writer = csv.DictWriter(output_csv, fieldnames=output_field_names)
        csv_writer.writeheader()
        for history_file_path in semantic_history_get_ordered_files(source_dir):
            visits = semantic_history_extract_records(history_file_path)
            for v in visits:
                csv_writer.writerow(v.__dict__)
                num_rows += 1
    logger.info(f"Wrote {num_rows} records to {output_file}")


@dataclasses.dataclass(frozen=True)
class PlaceAddress:
    formattedAddress: str
    adminDistrictMinor: str
    adminDistrictMajor: str
    postalCode: str
    countryCode: str


@dataclasses.dataclass(frozen=True)
class PlaceVisitWithAddress(PlaceVisit, PlaceAddress):
    pass


def semantic_history_annotate_place_visits(
    source_file: str,
    output_file: str,
    bing_api_key: str,
):
    geolocator = Bing(bing_api_key)
    num_lookups = 0
    num_rows = 0
    coordinates_to_address: typing.Dict[PlaceCoordinates, PlaceAddress] = dict()
    with open(source_file, "r") as source_csv:
        with open(output_file, "w", newline="\n") as output_csv:
            source_field_names = [
                field.name for field in dataclasses.fields(PlaceVisit)
            ]
            output_field_names = source_field_names + [
                field.name for field in dataclasses.fields(PlaceAddress)
            ]
            csv_writer = csv.DictWriter(output_csv, fieldnames=output_field_names)
            csv_writer.writeheader()
            csv_reader = csv.DictReader(source_csv, fieldnames=source_field_names)
            next(csv_reader)
            for row in csv_reader:
                visit = PlaceVisit(**row)
                coordinates = visit.coordinates()
                address = coordinates_to_address.get(coordinates)
                if not address:
                    geo_result = geolocator.reverse(
                        coordinates, include_country_code=True
                    )
                    num_lookups += 1
                    raw_address = geo_result.raw["address"]
                    address = PlaceAddress(
                        formattedAddress=geo_result.address,
                        adminDistrictMinor=raw_address.get("adminDistrict2"),
                        adminDistrictMajor=raw_address.get("adminDistrict"),
                        postalCode=raw_address.get("postalCode"),
                        countryCode=raw_address["countryRegionIso2"],
                    )
                    coordinates_to_address[coordinates] = address
                    logger.debug(
                        f"Geo lookup {num_lookups} - {coordinates} -> {address.formattedAddress}"
                    )
                output_row = row
                output_row.update(address.__dict__)
                csv_writer.writerow(output_row)
                num_rows += 1
    logger.info(
        f"Performed {num_lookups} lookups. Wrote {num_rows} records to {output_file}"
    )


def semantic_history_extract_trips(
    source_file: str, output_file: str, region_delimiter: str
):
    num_trips = 0
    with open(source_file, "r") as source_csv:
        source_field_names = [field.name for field in dataclasses.fields(PlaceVisit)]
        source_field_names.extend(
            [field.name for field in dataclasses.fields(PlaceAddress)]
        )
        csv_reader = csv.DictReader(source_csv, fieldnames=source_field_names)
        next(csv_reader)
        with open(output_file, "w") as output_csv:
            output_field_names = [
                region_delimiter,
                "start_time",
                "end_time",
                "duration",
            ]
            csv_writer = csv.DictWriter(output_csv, fieldnames=output_field_names)
            csv_writer.writeheader()
            trip_region: str = None
            trip_start: datetime = None

            def write_trip(region, start_time, end_time):
                duration = end_time - start_time
                quantized_duration = timedelta(
                    days=duration.days, seconds=duration.seconds
                )
                output_row = {
                    region_delimiter: region,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": quantized_duration,
                }
                csv_writer.writerow(output_row)

            for row in csv_reader:
                visit = PlaceVisitWithAddress(**row)
                current_region = getattr(visit, region_delimiter)
                if trip_region != current_region:
                    if trip_region is not None:
                        # A trip completed
                        assert trip_start != None
                        trip_end = datetime.fromisoformat(visit.start_time)
                        write_trip(trip_region, trip_start, trip_end)
                        num_trips += 1
                    trip_region = current_region
                    trip_start = datetime.fromisoformat(visit.start_time)

            trip_end = datetime.fromisoformat(visit.start_time)
            write_trip(current_region, trip_start, trip_end)
            num_trips += 1

    logger.info(f"Extracted {num_trips} trips to {output_file}")


# Entry point
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Utilities for Google Semantic Location Data"
    )
    subparsers = parser.add_subparsers()

    # Extract sub-command
    parser_extractor = subparsers.add_parser(
        "extract", help="Extract place visits from Semantic Location History."
    )
    parser_extractor.add_argument(
        "source_dir",
        help="Source directory of exported Semantic Location History.",
    )
    parser_extractor.add_argument(
        "--output_file",
        default="place_visits.csv",
        help="Output CSV file containing places visited.",
    )

    def extract_command(args):
        semantic_history_extract_place_visits(args.source_dir, args.output_file)

    parser_extractor.set_defaults(func=extract_command)

    # Annotate sub-command
    parser_annotate = subparsers.add_parser(
        "annotate",
        help="Annotate extracted place visits with address and country data using Geo Lookup",
    )
    parser_annotate.add_argument(
        "source_file", help="CSV file of places visited output by extract command."
    )
    parser_annotate.add_argument("--geolookup_bing_api_key", required=True)
    parser_annotate.add_argument("--output_file", default="place_visits_annotated.csv")

    def annotate_command(args):
        semantic_history_annotate_place_visits(
            args.source_file,
            args.output_file,
            args.geolookup_bing_api_key,
        )

    parser_annotate.set_defaults(func=annotate_command)

    parser_annotate = subparsers.add_parser(
        "trips",
        help="Build trip history from annotated place visits using a regional delimiter",
    )
    parser_annotate.add_argument(
        "source_file", help="CSV file of annotated place visits."
    )
    parser_annotate.add_argument(
        "--region_delimiter",
        choices=["adminMajor, adminMinor, postalCode, countryCode"],
        default="countryCode",
    )
    parser_annotate.add_argument("--output_file", default="trips.csv")

    def trips_command(args):
        region_mapping = {
            "adminMajor": "adminDistrictMajor",
            "adminMinor": "adminDistrictMinor",
            "postalCode": "postalCode",
            "countryCode": "countryCode",
        }
        semantic_history_extract_trips(
            args.source_file,
            args.output_file,
            region_mapping[args.region_delimiter],
        )

    parser_annotate.set_defaults(func=trips_command)

    options = parser.parse_args()
    options.func(options)
