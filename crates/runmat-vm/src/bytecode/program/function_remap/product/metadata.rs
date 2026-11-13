//! Local-coordinate projection of decorated metadata. Portable point, region,
//! distributed and collective owners remain in their original revision.
use super::super::super::Bytecode;
use runmat_hir::FunctionId;
use std::collections::HashMap;

pub(super) fn remap(
    bytecode: &mut Bytecode,
    ids: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    bytecode.active_function = bytecode
        .active_function
        .map(|function| ids.get(&function).copied().unwrap_or(function));
    #[cfg(feature = "native-accel")]
    for group in &mut bytecode.fusion_metadata.mir_fusion_candidate_groups {
        if let Some(mapped) = ids.get(&group.function) {
            group.function = *mapped;
        }
    }
    Ok(())
}
