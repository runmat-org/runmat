use std::collections::BTreeSet;

use runmat_types::{
    CapabilityRequirement, CapabilitySet, LabCount, ParallelAccess, ParallelManifest,
    ParallelRandomnessPolicy, ParallelVariableContract, ParallelVariableRole, ParforContract,
    ProgramFunctionId, RegionValueId,
};

use crate::{
    MirBody, MirDiagnostic, MirDiagnosticSeverity, MirIndexComponent, MirLocalId, MirLocalKind,
    MirOperand, MirPlace, MirRvalue, MirStmtKind, MirTerminatorKind,
};

use super::super::AnalysisStore;
use super::{facts, region};

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
        let maximum_workers = maximum_workers
            .as_deref()
            .and_then(constant_positive_worker_count);
        let mut capabilities = CapabilitySet::default();
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
    } else if let Some(dimensions) =
        sliced_dimensions(context.body, context.blocks, context.binding, local)
    {
        ParallelVariableRole::Sliced { dimensions }
    } else if let Some(operator) = reduction_operator(context.body, context.blocks, local) {
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

fn sliced_dimensions(
    body: &MirBody,
    blocks: &BTreeSet<crate::BasicBlockId>,
    binding: MirLocalId,
    local: MirLocalId,
) -> Option<Vec<u32>> {
    let mut dimensions = BTreeSet::new();
    let mut found = false;
    for statement in body
        .blocks
        .iter()
        .filter(|block| blocks.contains(&block.id))
        .flat_map(|block| &block.statements)
    {
        let places = match &statement.kind {
            MirStmtKind::Assign { place, .. } => vec![place],
            MirStmtKind::MultiAssign { targets, .. } => targets
                .targets
                .iter()
                .filter_map(|target| match target {
                    crate::MirOutputTarget::Place(place) => Some(place),
                    crate::MirOutputTarget::Discard => None,
                })
                .collect(),
            MirStmtKind::PlaceMutation(mutation) => vec![&mutation.place],
            _ => Vec::new(),
        };
        for place in places {
            if place_root_local(place) != Some(local) {
                continue;
            }
            found = true;
            let MirPlace::Index(_, indexing) = place else {
                return None;
            };
            let indexed = indexing
                .components
                .iter()
                .enumerate()
                .filter_map(|(dimension, component)| {
                    matches!(component, MirIndexComponent::Expr(MirOperand::Local(value)) if *value == binding)
                        .then_some(u32::try_from(dimension + 1).ok())
                        .flatten()
                })
                .collect::<Vec<_>>();
            if indexed.len() != 1 {
                return None;
            }
            dimensions.insert(indexed[0]);
        }
    }
    (found && dimensions.len() == 1).then(|| dimensions.into_iter().collect())
}

fn reduction_operator(
    body: &MirBody,
    blocks: &BTreeSet<crate::BasicBlockId>,
    local: MirLocalId,
) -> Option<runmat_types::OperatorKind> {
    let mut operator = None;
    let mut found = false;
    for statement in body
        .blocks
        .iter()
        .filter(|block| blocks.contains(&block.id))
        .flat_map(|block| &block.statements)
    {
        let MirStmtKind::Assign {
            place: MirPlace::Local(target),
            value: MirRvalue::Binary(left, candidate, right),
        } = &statement.kind
        else {
            if statement_writes_local(statement, local) {
                return None;
            }
            continue;
        };
        if *target != local {
            continue;
        }
        let reads_accumulator = matches!(left, MirOperand::Local(value) if *value == local)
            || (commutative_reduction(*candidate)
                && matches!(right, MirOperand::Local(value) if *value == local));
        if !reads_accumulator || !supported_reduction(*candidate) {
            return None;
        }
        found = true;
        if operator
            .replace(*candidate)
            .is_some_and(|prior| prior != *candidate)
        {
            return None;
        }
    }
    if found {
        operator
    } else {
        None
    }
}

fn supported_reduction(operator: runmat_types::OperatorKind) -> bool {
    matches!(
        operator,
        runmat_types::OperatorKind::Add
            | runmat_types::OperatorKind::Subtract
            | runmat_types::OperatorKind::MatrixMultiply
            | runmat_types::OperatorKind::ElementwiseMultiply
            | runmat_types::OperatorKind::ElementwiseAnd
            | runmat_types::OperatorKind::ElementwiseOr
    )
}

fn commutative_reduction(operator: runmat_types::OperatorKind) -> bool {
    !matches!(operator, runmat_types::OperatorKind::Subtract)
}

fn statement_writes_local(statement: &crate::MirStmt, local: MirLocalId) -> bool {
    let (_, writes) = super::super::regions::statement_uses_defs(statement);
    writes.contains(&local)
}

fn place_root_local(place: &MirPlace) -> Option<MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            place_root_local(base)
        }
        MirPlace::Binding(_) => None,
    }
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
