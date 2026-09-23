"""Record/replay cache.

SHA-256 over the canonical JSON of: provider, requested model, resolved
model if known, full message list, tool schema hash, temperature, top_p,
seed, response_format, and cache_salt from config.

Changing the prompt changes the key, which is the point: replay tests
cannot mask a prompt edit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from toolproof.runner.cache import CacheMissError, ResponseCache, cache_key


def key_fields(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-4o",
        "resolved_model": "gpt-4o-2024-08-06",
        "messages": [{"role": "user", "content": "find me a job"}],
        "tool_schema_hash": "a" * 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "response_format": {"type": "json_object"},
        "cache_salt": "v1",
    }
    base.update(overrides)
    return base


class TestKeyStability:
    def test_the_same_inputs_give_the_same_key(self) -> None:
        assert cache_key(key_fields()) == cache_key(key_fields())

    def test_the_key_is_a_sha256_hex_digest(self) -> None:
        digest = cache_key(key_fields())
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)

    def test_key_ordering_does_not_matter(self) -> None:
        """Canonical JSON: a dict built in a different order is the same key."""
        forward = cache_key({"provider": "openai", "model": "gpt-4o"})
        backward = cache_key({"model": "gpt-4o", "provider": "openai"})
        assert forward == backward

    def test_unknown_extra_fields_do_not_change_the_key(self) -> None:
        """Only the documented fields participate, so an unrelated config edit
        does not invalidate every recorded response."""
        assert cache_key(key_fields(irrelevant="noise")) == cache_key(key_fields())


class TestKeySensitivity:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("provider", "anthropic"),
            ("model", "gpt-4o-mini"),
            ("resolved_model", "gpt-4o-2024-11-20"),
            ("tool_schema_hash", "b" * 64),
            ("temperature", 0.7),
            ("top_p", 0.9),
            ("seed", 43),
            ("cache_salt", "v2"),
        ],
    )
    def test_every_documented_field_changes_the_key(self, field: str, value: Any) -> None:
        assert cache_key(key_fields(**{field: value})) != cache_key(key_fields())

    def test_editing_a_prompt_changes_the_key(self) -> None:
        """The whole point: a replay test cannot mask a prompt edit."""
        edited = key_fields(messages=[{"role": "user", "content": "find me a job please"}])
        assert cache_key(edited) != cache_key(key_fields())

    def test_adding_a_message_changes_the_key(self) -> None:
        extended = key_fields(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "find me a job"},
            ]
        )
        assert cache_key(extended) != cache_key(key_fields())

    def test_a_changed_response_format_changes_the_key(self) -> None:
        assert cache_key(key_fields(response_format=None)) != cache_key(key_fields())

    def test_a_missing_resolved_model_is_distinct_from_a_known_one(self) -> None:
        assert cache_key(key_fields(resolved_model=None)) != cache_key(key_fields())

    def test_unicode_content_is_handled_consistently(self) -> None:
        hindi = key_fields(messages=[{"role": "user", "content": "मुझे नौकरी चाहिए"}])
        assert cache_key(hindi) == cache_key(hindi)
        assert cache_key(hindi) != cache_key(key_fields())


class TestRecordAndReplay:
    def test_a_recorded_response_replays(self, tmp_path: Path) -> None:
        recorder = ResponseCache(tmp_path, mode="record")
        recorder.put(key_fields(), {"text": "hi"}, usage={"input_tokens": 3})

        entry = ResponseCache(tmp_path, mode="replay").get(key_fields())
        assert entry is not None
        assert entry.response == {"text": "hi"}
        assert entry.usage == {"input_tokens": 3}

    def test_replay_of_an_unrecorded_key_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CacheMissError):
            ResponseCache(tmp_path, mode="replay").get(key_fields())

    def test_the_miss_error_names_the_key_and_the_mode(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, mode="replay")
        with pytest.raises(CacheMissError) as exc:
            cache.get(key_fields())
        message = str(exc.value)
        assert cache_key(key_fields()) in message
        assert "replay" in message.lower()

    def test_the_miss_error_explains_it_will_not_call_a_provider(self, tmp_path: Path) -> None:
        """A silent live call in replay mode would make a free run cost money
        and make a deterministic test nondeterministic."""
        with pytest.raises(CacheMissError, match="never"):
            ResponseCache(tmp_path, mode="replay").get(key_fields())

    def test_record_mode_returns_none_on_a_miss(self, tmp_path: Path) -> None:
        assert ResponseCache(tmp_path, mode="record").get(key_fields()) is None

    def test_record_mode_replays_a_hit_rather_than_re_recording(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, mode="record")
        cache.put(key_fields(), {"text": "first"})
        entry = cache.get(key_fields())
        assert entry is not None
        assert entry.response == {"text": "first"}

    def test_an_edited_prompt_misses_in_replay(self, tmp_path: Path) -> None:
        ResponseCache(tmp_path, mode="record").put(key_fields(), {"text": "hi"})
        edited = key_fields(messages=[{"role": "user", "content": "different"}])
        with pytest.raises(CacheMissError):
            ResponseCache(tmp_path, mode="replay").get(edited)

    def test_entries_survive_a_new_cache_instance(self, tmp_path: Path) -> None:
        ResponseCache(tmp_path, mode="record").put(key_fields(), {"text": "hi"})
        assert ResponseCache(tmp_path, mode="record").get(key_fields()) is not None

    def test_the_cache_directory_is_created(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "cache"
        ResponseCache(target, mode="record").put(key_fields(), {"text": "hi"})
        assert target.is_dir()

    def test_a_corrupt_entry_is_a_miss_in_record_mode(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, mode="record")
        cache.put(key_fields(), {"text": "hi"})
        path = cache.path_for(cache_key(key_fields()))
        path.write_text("{not json", encoding="utf-8")
        assert cache.get(key_fields()) is None

    def test_a_corrupt_entry_is_an_error_in_replay_mode(self, tmp_path: Path) -> None:
        """Replay must be exact, so a damaged entry cannot pass as a hit."""
        recorder = ResponseCache(tmp_path, mode="record")
        recorder.put(key_fields(), {"text": "hi"})
        recorder.path_for(cache_key(key_fields())).write_text("{bad", encoding="utf-8")

        with pytest.raises(CacheMissError, match="unreadable"):
            ResponseCache(tmp_path, mode="replay").get(key_fields())

    def test_live_mode_never_reads_or_writes(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, mode="live")
        cache.put(key_fields(), {"text": "hi"})
        assert cache.get(key_fields()) is None
        assert not any(tmp_path.glob("**/*.json"))

    def test_an_unknown_mode_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mode"):
            ResponseCache(tmp_path, mode="dry-run")  # type: ignore[arg-type]

    def test_statistics_count_hits_and_misses(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path, mode="record")
        cache.get(key_fields())
        cache.put(key_fields(), {"text": "hi"})
        cache.get(key_fields())
        assert cache.misses == 1
        assert cache.hits == 1
