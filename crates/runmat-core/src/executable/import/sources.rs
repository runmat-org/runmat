//! Primary source identity comes from the original bytecode source ID, not entry ordering.
use super::super::{ExecutableSource, ExecutableSourceMap, SourceMapEntry};
use runmat_execution::{ExecutableComponentKind as Kind, ExecutableUnitEnvelope};
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceMapWire {
    entries: Vec<SourceEntryWire>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceEntryWire {
    source_id: usize,
    owner_identity: String,
    relative_path: String,
    display_name: String,
    text: String,
}

pub(super) fn decode(
    envelope: &ExecutableUnitEnvelope,
    bytecode: &runmat_vm::Bytecode,
) -> Result<(ExecutableSource, ExecutableSourceMap), String> {
    if envelope.manifest.revisions.source_map_schema
        != super::super::portable::SOURCE_MAP_SCHEMA_VERSION
    {
        return Err("portable source-map schema is unsupported".into());
    }
    let primary = bytecode
        .source_id
        .ok_or("portable bytecode lacks its original primary source ID")?
        .0;
    let wire: SourceMapWire = super::decode(envelope, Kind::SourceMap)?;
    let mut ids = BTreeSet::new();
    let mut source = None;
    let mut entries = Vec::with_capacity(wire.entries.len());
    for entry in wire.entries {
        if !ids.insert(entry.source_id) {
            return Err("portable source map repeats a source ID".into());
        }
        if entry.source_id == primary {
            if entry.owner_identity != envelope.manifest.identity.root_package {
                return Err("portable primary source differs from its root package".into());
            }
            source = Some(ExecutableSource::new(
                &entry.owner_identity,
                &entry.relative_path,
                &entry.text,
            ));
        }
        entries.push(SourceMapEntry {
            source_id: entry.source_id,
            owner_identity: entry.owner_identity,
            relative_path: entry.relative_path,
            display_name: entry.display_name,
            full_path: None,
            text: entry.text,
        });
    }
    Ok((
        source.ok_or("portable primary source is absent from its source map")?,
        ExecutableSourceMap::new(entries),
    ))
}
