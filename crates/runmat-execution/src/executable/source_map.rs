use serde::Deserialize;
use std::borrow::Cow;

use crate::{ContractError, Digest};

/// The v2 source-map component uses deterministic per-program ordinals and a
/// revision tied to exact text. This validation runs when an envelope is
/// imported as well as when one is built locally; a valid component hash
/// alone cannot authorize forged provenance inside its payload.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceMapV2<'a> {
    #[serde(borrow)]
    entries: Vec<SourceEntryV2<'a>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceEntryV2<'a> {
    source_id: usize,
    #[serde(borrow, rename = "owner_identity")]
    _owner_identity: Cow<'a, str>,
    #[serde(borrow, rename = "relative_path")]
    _relative_path: Cow<'a, str>,
    content_revision: Digest,
    #[serde(borrow, rename = "display_name")]
    _display_name: Cow<'a, str>,
    #[serde(borrow)]
    text: Cow<'a, str>,
}

pub(super) fn validate_source_map_v2(bytes: &[u8]) -> Result<(), ContractError> {
    let map: SourceMapV2<'_> = serde_json::from_slice(bytes)
        .map_err(|error| ContractError::invalid("executable.source_map", error.to_string()))?;
    if map.entries.is_empty() {
        return Err(ContractError::invalid(
            "executable.source_map",
            "source map has no primary source",
        ));
    }
    for (ordinal, entry) in map.entries.iter().enumerate() {
        if entry.source_id != ordinal {
            return Err(ContractError::invalid(
                "executable.source_map.source_id",
                "source ordinals must be dense, unique, and ordered from zero",
            ));
        }
        if entry.content_revision != Digest::sha256(entry.text.as_bytes()) {
            return Err(ContractError::invalid(
                "executable.source_map.content_revision",
                "source content revision does not match exact text",
            ));
        }
    }
    Ok(())
}
