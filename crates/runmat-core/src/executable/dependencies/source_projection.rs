//! Compare independently packaged bodies in one authenticated source namespace.
use crate::ExecutableSourceMap;
use runmat_vm::{BytecodeSourceRef, FunctionBytecode};

pub(super) fn normalize_original(
    selected_map: &ExecutableSourceMap,
    original_map: &ExecutableSourceMap,
    selected: &FunctionBytecode,
    original: &mut FunctionBytecode,
) -> Result<(), String> {
    match (selected.source_id, original.source_id) {
        (None, None) => return Ok(()),
        (
            Some(BytecodeSourceRef::Program(selected)),
            Some(BytecodeSourceRef::Program(original_source)),
        ) => {
            let selected_record = record(selected_map, selected)
                .ok_or("foreign function source is absent from its selected source map")?;
            let original_record = record(original_map, original_source)
                .ok_or("original function source is absent from its owner source map")?;
            let mut projected_record = original_record.clone();
            projected_record.source_id = selected;
            if &projected_record != selected_record {
                return Err(
                    "foreign function source differs from its original authenticated record".into(),
                );
            }
        }
        _ => return Err("foreign function source domains disagree with immutable products".into()),
    }
    // Validate the old seal before changing its coordinate, then reseal the
    // exact same stream under the selected product's equivalent source record.
    // All stage facts remain in the subsequent full definition comparison.
    runmat_vm::rebind_session_source(original, selected.source_id)
}

fn record(
    map: &ExecutableSourceMap,
    ordinal: runmat_types::ProgramSourceId,
) -> Option<&crate::SourceMapEntry> {
    map.entries()
        .iter()
        .find(|record| record.source_id == ordinal)
}
