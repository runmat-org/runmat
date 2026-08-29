use std::collections::BTreeSet;

use crate::parallel::MirCollectiveOp;
use crate::{
    BasicBlockId, MirBody, MirCallArg, MirCallee, MirIndexComponent, MirIndexing, MirLocalId,
    MirOperand, MirOutputTarget, MirPlace, MirRvalue, MirStmt, MirStmtKind,
};

pub(super) fn sliced_access(
    body: &MirBody,
    blocks: &BTreeSet<BasicBlockId>,
    function: runmat_types::ProgramFunctionId,
    binding: MirLocalId,
    local: MirLocalId,
) -> Option<runmat_types::ParallelSliceAccess> {
    let context = SlicePatternContext {
        body,
        blocks,
        function,
        binding,
        local,
    };
    let mut access = None;
    let mut writes = 0usize;
    for block in body
        .blocks
        .iter()
        .filter(|block| blocks.contains(&block.id))
    {
        for statement in &block.statements {
            if !statement_preserves_slice_pattern(statement, &context, &mut access, &mut writes) {
                return None;
            }
        }
        let (uses, _) = super::super::regions::terminator_uses_defs(&block.terminator.kind);
        if uses.contains(&local) {
            return None;
        }
    }
    (writes > 0).then_some(access).flatten()
}

#[derive(Clone, Copy)]
struct SlicePatternContext<'a> {
    body: &'a MirBody,
    blocks: &'a BTreeSet<BasicBlockId>,
    function: runmat_types::ProgramFunctionId,
    binding: MirLocalId,
    local: MirLocalId,
}

pub(super) fn reduction_operator(
    body: &MirBody,
    blocks: &BTreeSet<BasicBlockId>,
    local: MirLocalId,
) -> Option<runmat_types::OperatorKind> {
    let mut operator = None;
    let mut writes = 0usize;
    for block in body
        .blocks
        .iter()
        .filter(|block| blocks.contains(&block.id))
    {
        for statement in &block.statements {
            let (uses, definitions) = super::super::regions::statement_uses_defs(statement);
            if !uses.contains(&local) && !definitions.contains(&local) {
                continue;
            }
            let MirStmtKind::Assign {
                place: MirPlace::Local(target),
                value: MirRvalue::Binary(left, candidate, right),
            } = &statement.kind
            else {
                return None;
            };
            if *target != local || !supported_reduction(*candidate) {
                return None;
            }
            let left_is_accumulator = matches!(left, MirOperand::Local(value) if *value == local);
            let right_is_accumulator = matches!(right, MirOperand::Local(value) if *value == local);
            let valid_position =
                left_is_accumulator || (commutative_reduction(*candidate) && right_is_accumulator);
            if !valid_position || left_is_accumulator == right_is_accumulator {
                return None;
            }
            writes += 1;
            if operator
                .replace(*candidate)
                .is_some_and(|prior| prior != *candidate)
            {
                return None;
            }
        }
        let (uses, definitions) =
            super::super::regions::terminator_uses_defs(&block.terminator.kind);
        if uses.contains(&local) || definitions.contains(&local) {
            return None;
        }
    }
    (writes > 0).then_some(operator).flatten()
}

fn statement_preserves_slice_pattern(
    statement: &MirStmt,
    context: &SlicePatternContext<'_>,
    access: &mut Option<runmat_types::ParallelSliceAccess>,
    writes: &mut usize,
) -> bool {
    match &statement.kind {
        MirStmtKind::Assign { place, value } => {
            place_preserves_slice_pattern(place, context, access, writes)
                && rvalue_preserves_slice_pattern(value, context, access)
        }
        MirStmtKind::MultiAssign { targets, value } => {
            targets.targets.iter().all(|target| match target {
                MirOutputTarget::Place(place) => {
                    place_preserves_slice_pattern(place, context, access, writes)
                }
                MirOutputTarget::Discard => true,
            }) && rvalue_preserves_slice_pattern(value, context, access)
        }
        MirStmtKind::Expr(value) => rvalue_preserves_slice_pattern(value, context, access),
        MirStmtKind::PlaceMutation(mutation) => {
            place_preserves_slice_pattern(&mutation.place, context, access, writes)
        }
        MirStmtKind::WorkspaceEffect { bindings, .. } => !bindings.contains(&context.local),
        MirStmtKind::EnvironmentEffect(_) => true,
    }
}

fn place_preserves_slice_pattern(
    place: &MirPlace,
    context: &SlicePatternContext<'_>,
    access: &mut Option<runmat_types::ParallelSliceAccess>,
    writes: &mut usize,
) -> bool {
    if place_root_local(place) == Some(context.local) {
        let MirPlace::Index(base, indexing) = place else {
            return false;
        };
        if !matches!(base.as_ref(), MirPlace::Local(value) if *value == context.local) {
            return false;
        }
        let Some(candidate) = slice_access(context, indexing) else {
            return false;
        };
        if !record_slice_access(access, candidate) {
            return false;
        }
        *writes += 1;
        true
    } else {
        !place_mentions_local(place, context.local)
    }
}

fn rvalue_preserves_slice_pattern(
    value: &MirRvalue,
    context: &SlicePatternContext<'_>,
    access: &mut Option<runmat_types::ParallelSliceAccess>,
) -> bool {
    let local = context.local;
    match value {
        MirRvalue::Use(value) | MirRvalue::Unary(_, value) | MirRvalue::Spawn(value) => {
            !operand_is_local(value, local)
        }
        MirRvalue::Binary(left, _, right) => {
            !operand_is_local(left, local) && !operand_is_local(right, local)
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            !operand_is_local(left, local)
                && right_temps.iter().all(|statement| {
                    let mut nested_writes = 0;
                    statement_preserves_slice_pattern(
                        statement,
                        context,
                        access,
                        &mut nested_writes,
                    ) && nested_writes == 0
                })
                && !operand_is_local(right, local)
        }
        MirRvalue::Range { start, step, end } => {
            !operand_is_local(start, local)
                && step
                    .as_ref()
                    .is_none_or(|value| !operand_is_local(value, local))
                && !operand_is_local(end, local)
        }
        MirRvalue::Call(call) => {
            !matches!(&call.callee, MirCallee::Dynamic(value) if operand_is_local(value, local))
                && call
                    .args
                    .iter()
                    .all(|argument| !call_argument_mentions_local(argument, local))
        }
        MirRvalue::Aggregate { elements, .. } => {
            elements.iter().all(|value| !operand_is_local(value, local))
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => fields
            .iter()
            .all(|(_, value)| !operand_is_local(value, local)),
        MirRvalue::Index { base, indexing } => {
            if operand_is_local(base, local) {
                let Some(candidate) = slice_access(context, indexing) else {
                    return false;
                };
                record_slice_access(access, candidate)
            } else {
                !indexing_mentions_local(indexing, local)
            }
        }
        MirRvalue::Member { base, .. } => !operand_is_local(base, local),
        MirRvalue::DynamicMember { base, member } => {
            !operand_is_local(base, local) && !operand_is_local(member, local)
        }
        MirRvalue::Future { args, .. } => args
            .iter()
            .all(|argument| !call_argument_mentions_local(argument, local)),
        MirRvalue::Distributed(operation) => operation
            .operands()
            .all(|operand| !operand_is_local(operand, local)),
        MirRvalue::Collective(operation) => {
            collective_input(operation).is_none_or(|input| !operand_is_local(input, local))
        }
        MirRvalue::WorkspaceFirstStaticProperty { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End => true,
    }
}

fn slice_access(
    context: &SlicePatternContext<'_>,
    indexing: &MirIndexing,
) -> Option<runmat_types::ParallelSliceAccess> {
    let indexed = indexing
        .components
        .iter()
        .enumerate()
        .filter_map(|(dimension, component)| {
            let MirIndexComponent::Expr(operand) = component else {
                return None;
            };
            match loop_index_use(
                context.body,
                context.blocks,
                context.function,
                operand,
                context.binding,
            ) {
                LoopIndexUse::Affine(offset) => Some(
                    u32::try_from(dimension + 1)
                        .ok()
                        .map(|dimension| (dimension, offset)),
                ),
                LoopIndexUse::Independent => None,
                LoopIndexUse::Unsupported => Some(None),
            }
        })
        .collect::<Option<Vec<_>>>()?;
    if indexed.len() != 1 || indexing_mentions_local(indexing, context.local) {
        return None;
    }
    let (dimension, offset) = indexed[0];
    let axis = if indexing.components.len() == 1 {
        runmat_types::ParallelSliceAxis::Linear
    } else {
        runmat_types::ParallelSliceAxis::Dimension(dimension)
    };
    Some(runmat_types::ParallelSliceAccess { axis, offset })
}

#[derive(Clone, Copy)]
enum LoopIndexUse {
    Independent,
    Affine(runmat_types::ParallelSliceOffset),
    Unsupported,
}

fn loop_index_use(
    body: &MirBody,
    blocks: &BTreeSet<BasicBlockId>,
    function: runmat_types::ProgramFunctionId,
    operand: &MirOperand,
    binding: MirLocalId,
) -> LoopIndexUse {
    let MirOperand::Local(local) = operand else {
        return LoopIndexUse::Independent;
    };
    if *local == binding {
        return LoopIndexUse::Affine(runmat_types::ParallelSliceOffset::None);
    }
    let Some(value) = unique_local_definition(body, blocks, *local) else {
        return LoopIndexUse::Independent;
    };
    let MirRvalue::Binary(left, operator, right) = value else {
        return if local_depends_on(body, blocks, *local, binding, &mut BTreeSet::new()) {
            LoopIndexUse::Unsupported
        } else {
            LoopIndexUse::Independent
        };
    };
    let offset = match operator {
        runmat_types::OperatorKind::Add if operand_is_local(left, binding) => {
            simple_offset_operand(body, blocks, function, right, binding)
                .map(runmat_types::ParallelSliceOffset::Add)
        }
        runmat_types::OperatorKind::Add if operand_is_local(right, binding) => {
            simple_offset_operand(body, blocks, function, left, binding)
                .map(runmat_types::ParallelSliceOffset::Add)
        }
        runmat_types::OperatorKind::Subtract if operand_is_local(left, binding) => {
            simple_offset_operand(body, blocks, function, right, binding)
                .map(runmat_types::ParallelSliceOffset::Subtract)
        }
        _ => None,
    };
    offset.map_or(LoopIndexUse::Unsupported, LoopIndexUse::Affine)
}

fn simple_offset_operand(
    body: &MirBody,
    blocks: &BTreeSet<BasicBlockId>,
    function: runmat_types::ProgramFunctionId,
    operand: &MirOperand,
    binding: MirLocalId,
) -> Option<runmat_types::ParallelSliceOffsetOperand> {
    match operand {
        MirOperand::Constant(constant) => {
            constant_index(constant).map(runmat_types::ParallelSliceOffsetOperand::Constant)
        }
        MirOperand::Local(local)
            if *local != binding && unique_local_definition(body, blocks, *local).is_none() =>
        {
            Some(runmat_types::ParallelSliceOffsetOperand::Broadcast(
                runmat_types::RegionValueId {
                    function,
                    local: u32::try_from(local.0).ok()?,
                },
            ))
        }
        MirOperand::Local(_) | MirOperand::FunctionHandle(_) => None,
    }
}

fn constant_index(constant: &crate::MirConstant) -> Option<runmat_types::ParallelIndexConstant> {
    match constant {
        crate::MirConstant::Number(value) => {
            let value = value.parse::<f64>().ok()?;
            if !value.is_finite() || value.fract() != 0.0 {
                return None;
            }
            if value >= 0.0 && value <= u64::MAX as f64 {
                Some(runmat_types::ParallelIndexConstant::Unsigned(value as u64))
            } else if value >= i64::MIN as f64 && value <= i64::MAX as f64 {
                Some(runmat_types::ParallelIndexConstant::Signed(value as i64))
            } else {
                None
            }
        }
        crate::MirConstant::IntegerLiteral(value) => {
            use runmat_hir::IntegerLiteralClass;
            let bits = value.bits();
            match value.class() {
                IntegerLiteralClass::Int8 => Some(runmat_types::ParallelIndexConstant::Signed(
                    i64::from(bits as i8),
                )),
                IntegerLiteralClass::Int16 => Some(runmat_types::ParallelIndexConstant::Signed(
                    i64::from(bits as i16),
                )),
                IntegerLiteralClass::Int32 => Some(runmat_types::ParallelIndexConstant::Signed(
                    i64::from(bits as i32),
                )),
                IntegerLiteralClass::Int64 => {
                    Some(runmat_types::ParallelIndexConstant::Signed(bits as i64))
                }
                IntegerLiteralClass::UInt8
                | IntegerLiteralClass::UInt16
                | IntegerLiteralClass::UInt32
                | IntegerLiteralClass::UInt64 => {
                    Some(runmat_types::ParallelIndexConstant::Unsigned(bits))
                }
            }
        }
        _ => None,
    }
}

fn unique_local_definition<'a>(
    body: &'a MirBody,
    blocks: &BTreeSet<BasicBlockId>,
    local: MirLocalId,
) -> Option<&'a MirRvalue> {
    let statement = unique_local_definition_statement(body, blocks, local)?;
    match &statement.kind {
        MirStmtKind::Assign { value, .. } => Some(value),
        _ => None,
    }
}

fn unique_local_definition_statement<'a>(
    body: &'a MirBody,
    blocks: &BTreeSet<BasicBlockId>,
    local: MirLocalId,
) -> Option<&'a MirStmt> {
    let mut definitions = body
        .blocks
        .iter()
        .filter(|block| blocks.contains(&block.id))
        .flat_map(|block| &block.statements)
        .filter(|statement| match &statement.kind {
            MirStmtKind::Assign {
                place: MirPlace::Local(target),
                ..
            } => *target == local,
            _ => false,
        });
    let definition = definitions.next()?;
    definitions.next().is_none().then_some(definition)
}

fn local_depends_on(
    body: &MirBody,
    blocks: &BTreeSet<BasicBlockId>,
    local: MirLocalId,
    target: MirLocalId,
    visited: &mut BTreeSet<MirLocalId>,
) -> bool {
    if local == target {
        return true;
    }
    if !visited.insert(local) {
        return false;
    }
    let Some(statement) = unique_local_definition_statement(body, blocks, local) else {
        return false;
    };
    let (uses, _) = super::super::regions::statement_uses_defs(statement);
    uses.into_iter()
        .any(|used| local_depends_on(body, blocks, used, target, visited))
}

fn record_slice_access(
    access: &mut Option<runmat_types::ParallelSliceAccess>,
    candidate: runmat_types::ParallelSliceAccess,
) -> bool {
    match access {
        Some(existing) => *existing == candidate,
        None => {
            *access = Some(candidate);
            true
        }
    }
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

fn place_mentions_local(place: &MirPlace, local: MirLocalId) -> bool {
    match place {
        MirPlace::Local(value) => *value == local,
        MirPlace::Binding(_) => false,
        MirPlace::Member(base, _) => place_mentions_local(base, local),
        MirPlace::DynamicMember(base, member) => {
            place_mentions_local(base, local) || operand_is_local(member, local)
        }
        MirPlace::Index(base, indexing) => {
            place_mentions_local(base, local) || indexing_mentions_local(indexing, local)
        }
    }
}

fn indexing_mentions_local(indexing: &MirIndexing, local: MirLocalId) -> bool {
    indexing.components.iter().any(|component| {
        matches!(component, MirIndexComponent::Expr(value) if operand_is_local(value, local))
    })
}

fn call_argument_mentions_local(argument: &MirCallArg, local: MirLocalId) -> bool {
    match argument {
        MirCallArg::Single(value) => operand_is_local(value, local),
        MirCallArg::Expansion { base, indices, .. } => {
            operand_is_local(base, local)
                || indices.iter().any(|value| operand_is_local(value, local))
        }
    }
}

fn collective_input(operation: &MirCollectiveOp) -> Option<&MirOperand> {
    operation.input()
}

fn operand_is_local(operand: &MirOperand, local: MirLocalId) -> bool {
    matches!(operand, MirOperand::Local(value) if *value == local)
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
    matches!(
        operator,
        runmat_types::OperatorKind::Add
            | runmat_types::OperatorKind::ElementwiseMultiply
            | runmat_types::OperatorKind::ElementwiseAnd
            | runmat_types::OperatorKind::ElementwiseOr
    )
}
