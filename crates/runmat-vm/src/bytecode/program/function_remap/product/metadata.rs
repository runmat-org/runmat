use std::collections::HashMap;

use runmat_hir::FunctionId;
use runmat_types::ProgramFunctionId;

use super::super::super::Bytecode;

pub(super) fn remap(
    bytecode: &mut Bytecode,
    ids: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    #[cfg(feature = "native-accel")]
    if bytecode
        .fusion_metadata
        .stage_projection
        .as_ref()
        .is_some_and(|stage| mapped_owner(stage.owner, ids).is_some())
    {
        return Err("selected entrypoint fusion owner cannot change after decoration".into());
    }
    bytecode.active_function = bytecode
        .active_function
        .map(|function| ids.get(&function).copied().unwrap_or(function));
    bytecode.resume_points = std::mem::take(&mut bytecode.resume_points)
        .into_iter()
        .map(|(mut point, pc)| {
            if let Some(owner) = mapped_owner(point.function, ids) {
                point.function = owner;
            }
            (point, pc)
        })
        .collect();
    for region in &mut bytecode.regions {
        if let Some(owner) = mapped_owner(region.id.function, ids) {
            region.rebind_owner(owner);
        }
    }
    for region in &mut bytecode.parfor_regions {
        if let Some(owner) = mapped_owner(region.contract.id.0.function, ids) {
            region.rebind_owner(owner);
        }
    }
    for region in &mut bytecode.spmd_regions {
        if let Some(owner) = mapped_owner(region.contract.id.0.function, ids) {
            region.rebind_owner(owner);
        }
    }
    for distributed in &mut bytecode.distributed_values {
        if let Some(owner) = mapped_owner(distributed.id.function, ids) {
            distributed.id.function = owner;
            match &mut distributed.owner {
                runmat_types::DistributedOwner::Client(function) => *function = owner,
                runmat_types::DistributedOwner::Region(region) => region.0.function = owner,
            }
        }
    }
    for collective in &mut bytecode.collective_contracts {
        if let Some(owner) = mapped_owner(collective.id.region.0.function, ids) {
            collective.id.region.0.function = owner;
        }
    }
    #[cfg(feature = "native-accel")]
    for group in &mut bytecode.fusion_metadata.mir_fusion_candidate_groups {
        if let Some(mapped) = ids.get(&group.function) {
            group.function = *mapped;
        }
    }
    Ok(())
}

fn mapped_owner(
    owner: ProgramFunctionId,
    ids: &HashMap<FunctionId, FunctionId>,
) -> Option<ProgramFunctionId> {
    let function = FunctionId(usize::try_from(owner.0).ok()?);
    ids.get(&function)
        .and_then(|mapped| u32::try_from(mapped.0).ok())
        .map(ProgramFunctionId)
}
