import logging
from pathlib import Path

import pandas as pd

from utils import load_json, load_jsonl, parse_date, load_yaml

# Set up logging
logger = logging.getLogger(__name__)


class TSVerDataLoader:

    def __init__(self, input_jsonl_path: str):
        self.tsver_claims_path = Path(input_jsonl_path)
        self.tsver_dir = self.tsver_claims_path.parent
        self.tsver_ts_dir = self.tsver_dir / "time_series"

        assert self.tsver_dir.exists(), f"TSVer path {self.tsver_dir} does not exist!"
        assert self.tsver_claims_path.exists(), f"Claims file not found at {self.tsver_claims_path}"
        assert self.tsver_ts_dir.exists(), f"Time series directory not found at {self.tsver_ts_dir}"
        assert (self.tsver_ts_dir / "metadata.json").exists(), f"Time series metadata file not found at {(self.tsver_ts_dir / 'metadata.json')}"
        assert (self.tsver_ts_dir / "country_codes.yaml").exists(), f"Country codes file not found at {(self.tsver_ts_dir / 'country_codes.yaml')}"
        assert (self.tsver_ts_dir / "csv").exists(), f"Time series csv directory not found at {(self.tsver_ts_dir / 'csv')}"

        self.country_codes = load_yaml(self.tsver_ts_dir / "country_codes.yaml")

    def get_claims(self) -> list[dict]:
        claims_data = load_jsonl(self.tsver_claims_path)
        logger.info(f"Found {len(claims_data)} TSVer claims.")
        return claims_data

    def get_country_name_from_code(self, code: str) -> str | None:
        code = code.removeprefix("country/")
        if code in self.country_codes:
            return self.country_codes[code][0]

        logger.warning(f"Country code '{code}' not found in country code mapping.")
        return None

    def get_country_code_from_name(self, name: str) -> str | None:
        for code, names in self.country_codes.items():
            if name in names:
                return code

        logger.warning(f"Country name '{name}' not found in country code mapping.")
        return None

    def get_tseries_metadata(self) -> dict[str, dict[str, str]]:
        metadata = load_json(self.tsver_ts_dir / "metadata.json")
        ts_metadata = {}
        for m in metadata:
            assert m['filename'] not in ts_metadata, f"Duplicate time series filename: {m['filename']}"
            ts_metadata[m['filename']] = {
                'title': m['title'],
                'description': m['description'],
                'unit': m['unit']
            }

        return ts_metadata

    def get_metadata_for_ts_fname(self, ts_fname: str) -> dict[str, str] | None:
        metadata = self.get_tseries_metadata()
        for m in metadata:
            if m.removesuffix(".csv") == ts_fname.removesuffix(".csv"):
                return metadata[m]

        logger.warning(f"Metadata not found for time series filename: {ts_fname}")
        return None

    def get_metadata_for_ts_title(self, ts_title: str, include_csv_fname: bool = False) -> dict[str, str] | None:
        metadata = self.get_tseries_metadata()
        for m in metadata:
            if metadata[m]['title'] == ts_title:
                if include_csv_fname:
                    metadata[m]['csv_fname'] = m
                return metadata[m]

        logger.warning(f"Metadata not found for time series title: {ts_title}")
        return None

    def _load_csv_tserie(self, series_fpath: str):
        series_fpath = Path(series_fpath)
        if not series_fpath.exists():
            raise Exception(f"File not found: {series_fpath}")

        return pd.read_csv(series_fpath, index_col=0)

    def _load_tserie(self, tserie_path: str, filter_countries: list[str] | None = None, filter_years: list[int] | None = None) -> pd.DataFrame:
        # Load tseries data
        tserie_data = self._load_csv_tserie(tserie_path)

        # Index
        tserie_data = tserie_data.copy()
        tserie_data.index.name = "Date"
        tserie_data.reset_index(inplace=True)

        # Melt
        tserie_data = tserie_data.melt(id_vars=[tserie_data.columns[0]], var_name='Country', value_name='Value')

        # Remove country/ prefix
        tserie_data['Country'] = tserie_data['Country'].str.removeprefix("country/")

        # Sort
        tserie_data['Parsed Date'] = tserie_data['Date'].apply(parse_date)
        tserie_data = tserie_data.sort_values(['Parsed Date', 'Country']).drop('Parsed Date', axis=1)

        # Filter countries
        if filter_countries:
            tserie_data = tserie_data[tserie_data['Country'].isin(filter_countries)]

        # Filter years
        if filter_years:
            tserie_data = tserie_data[tserie_data['Date'].isin(filter_years)]

        return tserie_data

    def get_tseries_data(self, ts_title: str, ts_ranges: list[dict[str, int]], ts_countries: list[str], input_data: dict | None = None) -> str | None:
        logger.debug(f"Getting data for time series: {ts_title}, time ranges: {ts_ranges}, countries: {ts_countries}")
        metadata = self.get_metadata_for_ts_title(ts_title, include_csv_fname=True)
        if metadata is None:
            return None

        # Relevant country codes
        relevant_country_codes = []
        for c in ts_countries:
            country_code = self.get_country_code_from_name(c)
            if country_code is not None:
                relevant_country_codes.append(country_code)
            else:
                logger.warning(f"Country '{c}' not found in country code mapping. Skipping.")

        # Relevant year
        relevant_years = set()
        for r in ts_ranges:
            relevant_years.update(range(r['from'], r['to'] + 1))
        relevant_years = sorted(relevant_years)

        # Load time series
        csv_path = self.tsver_ts_dir / "csv" / metadata['csv_fname']
        if not csv_path.exists():
            logger.warning(f"CSV file for time series {ts_title} not found at {csv_path}")
            return None

        # Load TS, while filtering countries and time ranges
        loaded_tserie = self._load_tserie(csv_path, filter_countries=relevant_country_codes, filter_years=relevant_years)
        if loaded_tserie.empty:
            # Try again with original country names values (in case of LLM predicting country names instead of codes)
            loaded_tserie = self._load_tserie(csv_path, filter_countries=ts_countries, filter_years=relevant_years)

        if loaded_tserie.empty:
            logger.warning(f"No time series data found!\n\tClaim: {input_data['Claim']}\n\tTS: '{ts_title}'\n\tCountries: {ts_countries}\n\tTime Ranges: {ts_ranges}")
            return None

        # To markdown
        loaded_tserie_markdown = loaded_tserie.to_markdown(index=False, floatfmt=".2f")

        return loaded_tserie_markdown
