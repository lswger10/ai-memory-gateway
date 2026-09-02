import hashlib
import json
from pathlib import Path

from group_contracts_v11 import CONTRACT_VERSION, validate_room_event, verify_contract_bundle


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "contracts" / "group-room" / "v1.1"


def test_v11_bundle_matches_relay_bytes_and_hashes():
    assert hashlib.sha256((CONTRACT_ROOT / "group-room.schema.json").read_bytes()).hexdigest() == "dd692ba696e2e1cee2a2cddd7cb783bb2f6ddc4523f07763b530b9069d9f43ce"
    assert hashlib.sha256((CONTRACT_ROOT / "SHA256SUMS").read_bytes()).hexdigest() == "d07b52d2926f4fe6785bbc88263b113f8b468aeaefa27a3e6e9e83a072646ab0"
    assert len(verify_contract_bundle(CONTRACT_ROOT)) == 5


def test_v11_media_event_fixture_is_exactly_accepted():
    event = json.loads(
        (CONTRACT_ROOT / "fixtures" / "group-event-human-image.json").read_text("utf-8")
    )
    assert validate_room_event(event) == event
    assert event["contract_version"] == CONTRACT_VERSION
