from pathlib import Path


CONTRACT_ROOT = Path(__file__).resolve().parents[1] / "contracts" / "group-room" / "v1"


def test_gateway_validates_the_same_canonical_contract_fixtures():
    from group_contracts import validate_contract_bundle

    report = validate_contract_bundle(CONTRACT_ROOT)
    assert report.contract_version == "group-room.v1.0"
    assert report.hash_errors == ()
    assert report.fixture_count > 0


def test_gateway_contract_types_round_trip_stage_a_fixtures():
    from group_contracts import (
        ContextPackRequest,
        MemoryCandidateReceipt,
        OpaqueContextPack,
        PublicContextFacts,
    )

    cases = (
        ("context-pack-request.json", ContextPackRequest),
        ("context-facts-response-active.json", PublicContextFacts),
        ("context-pack-response-probe.json", OpaqueContextPack),
        ("memory-candidate-accepted.json", MemoryCandidateReceipt),
    )
    for filename, contract_type in cases:
        payload = __import__("json").loads(
            (CONTRACT_ROOT / "fixtures" / filename).read_text(encoding="utf-8")
        )
        assert contract_type.from_dict(payload).to_dict() == payload
