from archive_import import ArchiveRecord, inventory_archive


class SyntheticParser:
    format_id = "synthetic.v1"

    def detect(self, raw):
        return isinstance(raw, dict) and raw.get("format") == self.format_id

    def iter_records(self, raw, approved_mapping):
        for item in raw[approved_mapping["messages_field"]]:
            yield ArchiveRecord(
                source_system="synthetic",
                source_conversation_id=raw[approved_mapping["conversation_field"]],
                source_message_id=item[approved_mapping["message_id_field"]],
                raw_actor_label=item[approved_mapping["actor_field"]],
                raw_payload=item,
                raw_content=item[approved_mapping["content_field"]],
                raw_timestamp=item.get("created_at"),
            )


def test_unknown_export_shape_is_inventory_only_and_never_guessed():
    result = inventory_archive(
        {"unexpected": [{"speaker": "mystery", "body": "hello"}]},
        source_system="chatgpt",
        parsers=(),
    )
    assert result.detected_format is None
    assert result.records == ()
    assert result.requires_human_field_mapping is True
    assert "unexpected[].speaker" in result.field_paths
    assert result.suggested_actor_mapping == {}


def test_parser_contract_preserves_source_and_raw_actor_identity():
    parser = SyntheticParser()
    raw = {
        "format": "synthetic.v1",
        "conversation": "c-1",
        "messages": [{"id": "m-1", "actor": "raw-ai", "content": "hello"}],
    }
    mapping = {
        "conversation_field": "conversation",
        "messages_field": "messages",
        "message_id_field": "id",
        "actor_field": "actor",
        "content_field": "content",
    }
    record = next(parser.iter_records(raw, mapping))
    assert (
        record.source_system,
        record.source_conversation_id,
        record.source_message_id,
    ) == ("synthetic", "c-1", "m-1")
    assert record.raw_actor_label == "raw-ai"


def test_detected_format_still_requires_explicit_field_and_actor_mapping():
    result = inventory_archive(
        {"format": "synthetic.v1", "messages": []},
        source_system="synthetic",
        parsers=(SyntheticParser(),),
    )
    assert result.detected_format == "synthetic.v1"
    assert result.records == ()
    assert result.requires_human_field_mapping is True
