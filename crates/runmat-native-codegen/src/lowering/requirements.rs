use std::collections::BTreeSet;

use crate::{NativeCodegenError, NativeCodegenResult};

pub(super) fn validate_requirements(
    mir: &runmat_mir::MirAssembly,
    manifest: &runmat_execution::ExecutableUnitManifest,
) -> NativeCodegenResult<()> {
    let parfor = manifest
        .parallel
        .parfor_regions
        .iter()
        .map(|contract| contract.id)
        .collect::<BTreeSet<_>>();
    let spmd = manifest
        .parallel
        .spmd_regions
        .iter()
        .map(|contract| contract.id)
        .collect::<BTreeSet<_>>();
    for body in mir.bodies.values() {
        for block in &body.blocks {
            match &block.terminator.kind {
                runmat_mir::MirTerminatorKind::ParFor { region, .. }
                    if !parfor.contains(region) =>
                {
                    return Err(NativeCodegenError::new(
                        "native.lowering.parfor_contract",
                        "parfor terminator has no exact executable-manifest contract",
                    ));
                }
                runmat_mir::MirTerminatorKind::Spmd { region, .. } if !spmd.contains(region) => {
                    return Err(NativeCodegenError::new(
                        "native.lowering.spmd_contract",
                        "spmd terminator has no exact executable-manifest contract",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

pub(super) fn reject_predeclared_capabilities(
    mir: &runmat_mir::MirAssembly,
) -> NativeCodegenResult<()> {
    for (function, body) in &mir.bodies {
        let function = u32::try_from(function.0)
            .map(runmat_types::ProgramFunctionId)
            .map_err(|_| {
                NativeCodegenError::new(
                    "native.lowering.function_identity",
                    "MIR function identity exceeds the Native IR schema",
                )
            })?;
        for block in &body.blocks {
            for (position, statement) in block.statements.iter().enumerate() {
                let value = match &statement.kind {
                    runmat_mir::MirStmtKind::Assign { value, .. }
                    | runmat_mir::MirStmtKind::MultiAssign { value, .. }
                    | runmat_mir::MirStmtKind::Expr(value) => Some(value),
                    _ => None,
                };
                if let Some(value) = value {
                    reject_rvalue(function, block.id, position, value)?;
                }
            }
            match &block.terminator.kind {
                runmat_mir::MirTerminatorKind::For { iterable, .. }
                | runmat_mir::MirTerminatorKind::ParFor { iterable, .. } => {
                    reject_rvalue(function, block.id, block.statements.len(), iterable)?;
                }
                runmat_mir::MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
                    runmat_mir::parallel::MirSpmdHeader::Default => {}
                    runmat_mir::parallel::MirSpmdHeader::One(a) => {
                        reject_rvalue(function, block.id, block.statements.len(), a)?;
                    }
                    runmat_mir::parallel::MirSpmdHeader::Two(a, b) => {
                        reject_rvalue(function, block.id, block.statements.len(), a)?;
                        reject_rvalue(function, block.id, block.statements.len(), b)?;
                    }
                    runmat_mir::parallel::MirSpmdHeader::Three(a, b, c) => {
                        reject_rvalue(function, block.id, block.statements.len(), a)?;
                        reject_rvalue(function, block.id, block.statements.len(), b)?;
                        reject_rvalue(function, block.id, block.statements.len(), c)?;
                    }
                },
                _ => {}
            }
        }
    }
    Ok(())
}

fn reject_rvalue(
    function: runmat_types::ProgramFunctionId,
    block: runmat_mir::BasicBlockId,
    position: usize,
    value: &runmat_mir::MirRvalue,
) -> NativeCodegenResult<()> {
    let construct = runmat_mir::rvalue_construct_kind(value);
    if construct.native_lowering_class() == runmat_mir::NativeLoweringClass::CapabilityRejection {
        let block = u32::try_from(block.0).unwrap_or(u32::MAX);
        let position = u32::try_from(position).unwrap_or(u32::MAX);
        return Err(NativeCodegenError::new(
            "native.capability.distributed_core_pending",
            "distributed-value and collective Native IR require the R25 distributed core",
        )
        .at_point(runmat_types::ProgramPointId {
            function,
            block,
            position,
        })
        .for_construct(construct));
    }
    if let runmat_mir::MirRvalue::ShortCircuit { right_temps, .. } = value {
        for statement in right_temps {
            let nested = match &statement.kind {
                runmat_mir::MirStmtKind::Assign { value, .. }
                | runmat_mir::MirStmtKind::MultiAssign { value, .. }
                | runmat_mir::MirStmtKind::Expr(value) => Some(value),
                _ => None,
            };
            if let Some(nested) = nested {
                reject_rvalue(function, block, position, nested)?;
            }
        }
    }
    Ok(())
}
