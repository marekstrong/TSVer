import json
import yaml
from dateutil import parser
from pathlib import Path

import pandas as pd


def load_yaml(fpath: str) -> dict:
    fpath = Path(fpath)
    with fpath.open("r") as f:
        return yaml.safe_load(f)


def load_json(fpath: str) -> dict:
    fpath = Path(fpath)
    with fpath.open("r") as f:
        return json.load(f)


def load_jsonl(fpath: str) -> list[dict]:
    fpath = Path(fpath)

    data = []
    with fpath.open("r") as f:
        for line in f:
            line_d = json.loads(line)
            data.append(line_d)

    return data


def output_jsonl(json_data: list[dict], fpath: str) -> None:
    fpath = Path(fpath)

    assert type(json_data) == list
    with fpath.open("w") as fw:
        for d in json_data:
            out = f"{json.dumps(d)}\n"
            fw.write(out)


def parse_date(date_string: str) -> pd.Timestamp:
    try:
        return parser.parse(str(date_string))
    except Exception as e:
        return pd.NaT  # Return NaT (Not a Time) for unparseable dates