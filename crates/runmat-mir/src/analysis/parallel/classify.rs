use std::collections::{BTreeMap, BTreeSet};

use runmat_types::{
    CapabilityRequirement, CapabilitySet, LabCount, ParallelAccess, ParallelManifest,
    ParallelRandomnessPolicy, ParallelVariableContract, ParallelVariableRole, ParforContract,
    ProgramFunctionId, RegionValueId,
};

use crate::{
    MirBody, MirDiagnostic, MirDiagnosticSeverity, MirLocalId, MirLocalKind, MirRvalue,
    MirTerminatorKind,
};
use runmat_hir::FunctionId;

use super::super::AnalysisStore;
use super::{facts, legality, region};

struct ClassificationContext<'a> {
    body: &'a MirBody,
    blocks: &'a BTreeSet<crate::BasicBlockId>,
    function: ProgramFunctionId,
    store: &'a AnalysisStore,
    header: crate::BasicBlockId,
    header_position: usize,
    binding: MirLocalId,
    reads: &'a BTreeSet<MirLocalId>,
    writes: &'a BTreeSet<MirLocalId>,
}

pub(super) fn classify_body(
    body: &MirBody,
    function: ProgramFunctionId,
    store: &AnalysisStore,
    summaries: &BTreeMap<FunctionId, super::super::inference::FunctionSummary>,
    manifest: &mut ParallelManifest,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for header in &body.blocks {
        let MirTerminatorKind::ParFor {
            region: id,
            binding,
            iterable,
            maximum_workers,
            body_block,
            exit_block,
        } = &header.terminator.kind
        else {
            continue;
        };
        let blocks = region::body_blocks(body, header.id, *body_block, *exit_block);
        let mut legal =
            legality::validate_control_flow(body, &blocks, header.id, *exit_block, diagnostics);
        if region::contains_parallel_region(body, &blocks) {
            diagnostics.push(
                MirDiagnostic::new(
                    "RM-MIR0014",
                    MirDiagnosticSeverity::Error,
                    "a parfor body cannot contain another parallel region",
                    header.terminator.span,
                )
                .with_primary_label("nested parallel region exceeds the parent resource budget")
                .with_help("move the nested parallel work outside this parfor")
                .with_category("parfor-legality"),
            );
            legal = false;
        }
        let access = region::accesses(body, &blocks);
        let header_position = header.statements.len();
        let context = ClassificationContext {
            body,
            blocks: &blocks,
            function,
            store,
            header: header.id,
            header_position,
            binding: *binding,
            reads: &access.reads,
            writes: &access.writes,
        };
        let mut variables = access
            .reads
            .union(&access.writes)
            .copied()
            .chain(std::iter::once(*binding))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|local| variable_contract(&context, local))
            .collect::<Vec<_>>();
        variables.sort_by_key(|variable| variable.value);
        for variable in &variables {
            let Ok(local_index) = usize::try_from(variable.value.local) else {
                continue;
            };
            if matches!(
                variable.role,
                ParallelVariableRole::Private | ParallelVariableRole::Temporary
            ) && legality::read_before_iteration_assignment(
                body,
                &blocks,
                *body_block,
                *binding,
                MirLocalId(local_index),
            ) {
                let span = body
                    .locals
                    .get(local_index)
                    .map(|local| local.span)
                    .unwrap_or(header.terminator.span);
                diagnostics.push(
                    MirDiagnostic::new(
                        "RM-MIR0019",
                        MirDiagnosticSeverity::Error,
                        "parfor variable cannot be classified safely",
                        span,
                    )
                    .with_primary_label(
                        "this value is read before an iteration-local assignment and is not a valid broadcast, slice, or reduction",
                    )
                    .with_help(
                        "use one consistent sliced dimension, a supported reduction, or assign the temporary on every path before reading it",
                    )
                    .with_category("parfor-legality"),
                );
                legal = false;
            }
        }
        if !legal {
            continue;
        }
        let maximum_workers = maximum_workers
            .as_deref()
            .and_then(constant_positive_worker_count);
        let mut capabilities = CapabilitySet::default();
        let mut effects = runmat_types::EffectSet::default();
        for block in body
            .blocks
            .iter()
            .filter(|block| blocks.contains(&block.id))
        {
            for statement in &block.statements {
                let (statement_effects, statement_capabilities) =
                    super::super::inference::statement_contract(statement, summaries);
                effects.0.extend(statement_effects.0);
                capabilities.0.extend(statement_capabilities.0);
            }
            if matches!(block.terminator.kind, MirTerminatorKind::Await { .. }) {
                effects.0.insert(runmat_types::EffectKind::MaySuspend);
            }
        }
        capabilities
            .0
            .insert(CapabilityRequirement::ParallelRuntime);
        let Some(loop_variable) = region_value(function, *binding) else {
            continue;
        };
        manifest.parfor_regions.push(ParforContract {
            id: *id,
            loop_variable,
            iterable: facts::rvalue_fact(store, function, header.id, header_position, iterable),
            variables,
            maximum_workers,
            effects,
            capabilities,
            randomness: ParallelRandomnessPolicy::DeterministicSubstreams,
        });
    }
}

fn variable_contract(
    context: &ClassificationContext<'_>,
    local: MirLocalId,
) -> Option<ParallelVariableContract> {
    let value = region_value(context.function, local)?;
    let fact = facts::value_fact(
        context.store,
        context.function,
        context.header,
        context.header_position,
        local,
    );
    let read = context.reads.contains(&local);
    let write = context.writes.contains(&local);
    let role = if local == context.binding {
        ParallelVariableRole::Loop
    } else if read && !write {
        ParallelVariableRole::Broadcast
    } else if let Some(access) = super::patterns::sliced_access(
        context.body,
        context.blocks,
        context.function,
        context.binding,
        local,
    ) {
        ParallelVariableRole::Sliced { access }
    } else if let Some(operator) =
        super::patterns::reduction_operator(context.body, context.blocks, local)
    {
        ParallelVariableRole::Reduction { operator }
    } else {
        match context.body.locals.get(local.0).map(|local| &local.kind) {
            Some(MirLocalKind::Temporary) => ParallelVariableRole::Temporary,
            Some(MirLocalKind::Capture) => ParallelVariableRole::Private,
            _ => ParallelVariableRole::Private,
        }
    };
    let access = match (read, write) {
        (true, true) => ParallelAccess::ReadWrite,
        (false, true) => ParallelAccess::Write,
        _ => ParallelAccess::Read,
    };
    Some(ParallelVariableContract {
        value,
        role,
        access,
        transferable: facts::transferable(&fact),
        fact,
    })
}

fn region_value(function: ProgramFunctionId, local: MirLocalId) -> Option<RegionValueId> {
    Some(RegionValueId {
        function,
        local: u32::try_from(local.0).ok()?,
    })
}

fn constant_positive_worker_count(value: &MirRvalue) -> Option<LabCount> {
    let MirRvalue::Use(crate::MirOperand::Constant(constant)) = value else {
        return None;
    };
    let text = match constant {
        crate::MirConstant::Number(text) => text.as_str(),
        crate::MirConstant::IntegerLiteral(value) => return integer_literal_count(value),
        _ => return None,
    };
    let count = text.parse::<u32>().ok()?;
    (count > 0).then_some(LabCount(count))
}

fn integer_literal_count(value: &runmat_hir::IntegerLiteral) -> Option<LabCount> {
    use runmat_hir::IntegerLiteralClass;

    let bits = value.bits();
    let signed_positive = match value.class() {
        IntegerLiteralClass::Int8 => bits < (1 << 7),
        IntegerLiteralClass::Int16 => bits < (1 << 15),
        IntegerLiteralClass::Int32 => bits < (1 << 31),
        IntegerLiteralClass::Int64 => bits < (1 << 63),
        IntegerLiteralClass::UInt8
        | IntegerLiteralClass::UInt16
        | IntegerLiteralClass::UInt32
        | IntegerLiteralClass::UInt64 => true,
    };
    if !signed_positive {
        return None;
    }
    let count = u32::try_from(bits).ok()?;
    (count > 0).then_some(LabCount(count))
}
