use crate::ir::{NativeMirSite, NativeSitePhase};
use crate::{NativeCodegenError, NativeCodegenResult};
use runmat_mir::{MirOperand, MirRvalue, MirStmtKind, MirTerminatorKind};
use runmat_types::{ProgramFunctionId, ProgramPointId};

/// Independently inventories canonical MIR sites before lowering. The Native IR
/// verifier compares this list with emitted sites so adding an unsupported MIR
/// arm cannot silently disappear behind the lowering implementation.
pub(super) fn expected_sites(
    function: ProgramFunctionId,
    body: &runmat_mir::MirBody,
) -> NativeCodegenResult<Vec<NativeMirSite>> {
    let mut sites = Vec::new();
    for block in &body.blocks {
        for (position, statement) in block.statements.iter().enumerate() {
            let point = point(function, block.id, position)?;
            if let Some(value) = statement_rvalue(&statement.kind) {
                sites.push(site(
                    point,
                    NativeSitePhase::Rvalue,
                    0,
                    runmat_mir::rvalue_construct_kind(value),
                ));
            }
            sites.push(site(
                point,
                NativeSitePhase::Statement,
                0,
                runmat_mir::statement_construct_kind(&statement.kind),
            ));
        }
        let point = point(function, block.id, block.statements.len())?;
        inventory_terminator_rvalues(point, &block.terminator.kind, &mut sites)?;
        sites.push(site(
            point,
            NativeSitePhase::Terminator,
            0,
            runmat_mir::terminator_construct_kind(&block.terminator.kind),
        ));
    }
    sites.sort();
    if sites.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(NativeCodegenError::new(
            "native.lowering.site_inventory",
            "canonical MIR produced duplicate Native IR sites",
        )
        .at_function(function));
    }
    Ok(sites)
}

fn statement_rvalue(statement: &MirStmtKind) -> Option<&MirRvalue> {
    match statement {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::Expr(value) => Some(value),
        MirStmtKind::PlaceMutation(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => None,
    }
}

fn inventory_terminator_rvalues(
    point: ProgramPointId,
    terminator: &MirTerminatorKind,
    sites: &mut Vec<NativeMirSite>,
) -> NativeCodegenResult<()> {
    let mut ordinal = 0_u32;
    let mut rvalue = |value: &MirRvalue| -> NativeCodegenResult<()> {
        sites.push(site(
            point,
            NativeSitePhase::TerminatorRvalue,
            ordinal,
            runmat_mir::rvalue_construct_kind(value),
        ));
        ordinal = ordinal.checked_add(1).ok_or_else(|| {
            NativeCodegenError::new(
                "native.lowering.terminator_ordinal",
                "terminator operation count exceeds the Native IR schema",
            )
            .at_point(point)
        })?;
        Ok(())
    };
    let mut operand = |operand: &MirOperand| -> NativeCodegenResult<()> {
        if !matches!(operand, MirOperand::Local(_)) {
            rvalue(&MirRvalue::Use(operand.clone()))?;
        }
        Ok(())
    };
    match terminator {
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Unreachable => {}
        MirTerminatorKind::Branch { cond, .. } => operand(cond)?,
        MirTerminatorKind::Switch { discr, cases, .. } => {
            operand(discr)?;
            for (case, _) in cases {
                operand(case)?;
            }
        }
        MirTerminatorKind::For { iterable, .. } => rvalue(iterable)?,
        MirTerminatorKind::ParFor {
            iterable,
            maximum_workers,
            ..
        } => {
            rvalue(iterable)?;
            if let Some(maximum_workers) = maximum_workers {
                rvalue(maximum_workers)?;
            }
        }
        MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
            runmat_mir::parallel::MirSpmdHeader::Default => {}
            runmat_mir::parallel::MirSpmdHeader::One(a) => rvalue(a)?,
            runmat_mir::parallel::MirSpmdHeader::Two(a, b) => {
                rvalue(a)?;
                rvalue(b)?;
            }
            runmat_mir::parallel::MirSpmdHeader::Three(a, b, c) => {
                rvalue(a)?;
                rvalue(b)?;
                rvalue(c)?;
            }
        },
        MirTerminatorKind::Return(values) => {
            for value in values {
                operand(value)?;
            }
        }
        MirTerminatorKind::Await { future, .. } => operand(future)?,
    }
    Ok(())
}

fn point(
    function: ProgramFunctionId,
    block: runmat_mir::BasicBlockId,
    position: usize,
) -> NativeCodegenResult<ProgramPointId> {
    Ok(ProgramPointId {
        function,
        block: u32::try_from(block.0).map_err(|_| {
            NativeCodegenError::new(
                "native.lowering.block_identity",
                "MIR block identity exceeds the Native IR schema",
            )
            .at_function(function)
        })?,
        position: u32::try_from(position).map_err(|_| {
            NativeCodegenError::new(
                "native.lowering.position",
                "MIR statement position exceeds the Native IR schema",
            )
            .at_function(function)
        })?,
    })
}

fn site(
    point: ProgramPointId,
    phase: NativeSitePhase,
    ordinal: u32,
    construct: runmat_mir::MirConstructKind,
) -> NativeMirSite {
    NativeMirSite {
        point,
        phase,
        ordinal,
        construct,
    }
}
