from provider_adapters import (
    AnthropicMessagesAdapter,
    OpenAIChatCompletionsAdapter,
    OpenAIResponsesAdapter,
)


def test_anthropic_usage_preserves_creation_and_read_tokens():
    usage = AnthropicMessagesAdapter().parse_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 70,
            "cache_read_input_tokens": 30,
        }
    )
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cache_creation_input_tokens == 70
    assert usage.cache_read_input_tokens == 30
    assert usage.cached_tokens is None


def test_openai_usage_preserves_cached_tokens():
    responses = OpenAIResponsesAdapter().parse_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 80},
        }
    )
    chat = OpenAIChatCompletionsAdapter().parse_usage(
        {
            "prompt_tokens": 90,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 60},
        }
    )
    assert responses.cached_tokens == 80
    assert chat.cached_tokens == 60


def test_absent_usage_fields_remain_none():
    usage = OpenAIResponsesAdapter().parse_usage({})
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.cache_creation_input_tokens is None
    assert usage.cache_read_input_tokens is None
    assert usage.cached_tokens is None
