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
                if let Some(value) = statement_value(&statement.kind) {
                    reject_rvalue(function, block.id, position, value)?;
                }
                reject_inventory(
                    function,
                    block.id,
                    position,
                    runmat_mir::statement_construct_inventory(&statement.kind),
                )?;
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

fn statement_value(statement: &runmat_mir::MirStmtKind) -> Option<&runmat_mir::MirRvalue> {
    match statement {
        runmat_mir::MirStmtKind::Assign { value, .. }
        | runmat_mir::MirStmtKind::MultiAssign { value, .. }
        | runmat_mir::MirStmtKind::SequenceAssign { value, .. }
        | runmat_mir::MirStmtKind::Expr(value) => Some(value),
        runmat_mir::MirStmtKind::PlaceMutation(_)
        | runmat_mir::MirStmtKind::CaptureSequence { .. }
        | runmat_mir::MirStmtKind::WorkspaceEffect { .. }
        | runmat_mir::MirStmtKind::EnvironmentEffect(_) => None,
    }
}

fn reject_rvalue(
    function: runmat_types::ProgramFunctionId,
    block: runmat_mir::BasicBlockId,
    position: usize,
    value: &runmat_mir::MirRvalue,
) -> NativeCodegenResult<()> {
    reject_inventory(
        function,
        block,
        position,
        runmat_mir::rvalue_construct_inventory(value),
    )
}

fn reject_inventory(
    function: runmat_types::ProgramFunctionId,
    block: runmat_mir::BasicBlockId,
    position: usize,
    inventory: impl IntoIterator<Item = runmat_mir::MirConstructKind>,
) -> NativeCodegenResult<()> {
    if let Some(construct) = inventory.into_iter().find(|construct| {
        construct.native_lowering_class() == runmat_mir::NativeLoweringClass::CapabilityRejection
    }) {
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
    Ok(())
}
