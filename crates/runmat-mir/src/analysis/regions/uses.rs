use std::collections::BTreeSet;

use runmat_types::{ProgramFunctionId, ProgramPointId};

use crate::parallel::MirSpmdHeader;
use crate::{
    BasicBlockId, MirBody, MirCallArg, MirCallee, MirIndexComponent, MirIndexing, MirLocalId,
    MirOperand, MirOutputTarget, MirPlace, MirRvalue, MirStmt, MirStmtKind, MirTerminatorKind,
};

pub(crate) type Locals = BTreeSet<MirLocalId>;

pub(crate) fn statement_uses_defs(statement: &MirStmt) -> (Locals, Locals) {
    let mut uses = Locals::new();
    let mut defs = Locals::new();
    match &statement.kind {
        MirStmtKind::Assign { place, value } => {
            rvalue_uses(value, &mut uses, &mut defs);
            place_uses_defs(place, &mut uses, &mut defs);
        }
        MirStmtKind::MultiAssign { targets, value } => {
            rvalue_uses(value, &mut uses, &mut defs);
            for target in &targets.targets {
                match target {
                    MirOutputTarget::Place(place) => place_uses_defs(place, &mut uses, &mut defs),
                    MirOutputTarget::Sequence(target) => {
                        sequence_target_uses_defs(target, &mut uses, &mut defs)
                    }
                    MirOutputTarget::Discard => {}
                }
            }
        }
        MirStmtKind::SequenceAssign { target, value } => {
            rvalue_uses(value, &mut uses, &mut defs);
            let base = match target {
                crate::MirSequenceTarget::Member { base, .. } => base,
                crate::MirSequenceTarget::DynamicMember { base, member } => {
                    operand_uses(member, &mut uses);
                    base
                }
                crate::MirSequenceTarget::CellContents { base, indexing } => {
                    indexing_uses(indexing, &mut uses);
                    base
                }
            };
            place_uses_defs(base, &mut uses, &mut defs);
        }
        MirStmtKind::CaptureSequence { source, .. } => {
            source.visit_outer_operands(|operand| operand_uses(operand, &mut uses));
        }
        MirStmtKind::Expr(value) => rvalue_uses(value, &mut uses, &mut defs),
        MirStmtKind::PlaceMutation(mutation) => {
            place_read_uses(&mutation.place, &mut uses);
            if let Some(root) = place_root_local(&mutation.place) {
                defs.insert(root);
            }
        }
        MirStmtKind::WorkspaceEffect { bindings, .. } => defs.extend(bindings.iter().copied()),
        MirStmtKind::EnvironmentEffect(_) => {}
    }
    (uses, defs)
}

fn sequence_target_uses_defs(
    target: &crate::MirSequenceTarget,
    uses: &mut Locals,
    defs: &mut Locals,
) {
    match target {
        crate::MirSequenceTarget::Member { base, .. } => place_uses_defs(base, uses, defs),
        crate::MirSequenceTarget::DynamicMember { base, member } => {
            operand_uses(member, uses);
            place_uses_defs(base, uses, defs);
        }
        crate::MirSequenceTarget::CellContents { base, indexing } => {
            indexing_uses(indexing, uses);
            place_uses_defs(base, uses, defs);
        }
    }
}

pub(crate) fn terminator_uses_defs(kind: &MirTerminatorKind) -> (Locals, Locals) {
    let mut uses = Locals::new();
    let mut defs = Locals::new();
    match kind {
        MirTerminatorKind::Goto(_) | MirTerminatorKind::TryCatch { .. } => {}
        MirTerminatorKind::Branch { cond, .. } => operand_uses(cond, &mut uses),
        MirTerminatorKind::Switch { discr, cases, .. } => {
            operand_uses(discr, &mut uses);
            for (case, _) in cases {
                operand_uses(case, &mut uses);
            }
        }
        MirTerminatorKind::For {
            binding, iterable, ..
        } => {
            rvalue_uses(iterable, &mut uses, &mut defs);
            defs.insert(*binding);
        }
        MirTerminatorKind::ParFor {
            binding,
            iterable,
            maximum_workers,
            ..
        } => {
            rvalue_uses(iterable, &mut uses, &mut defs);
            if let Some(maximum_workers) = maximum_workers {
                rvalue_uses(maximum_workers, &mut uses, &mut defs);
            }
            defs.insert(*binding);
        }
        MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
            MirSpmdHeader::Default => {}
            MirSpmdHeader::One(first) => rvalue_uses(first, &mut uses, &mut defs),
            MirSpmdHeader::Two(first, second) => {
                rvalue_uses(first, &mut uses, &mut defs);
                rvalue_uses(second, &mut uses, &mut defs);
            }
            MirSpmdHeader::Three(first, second, third) => {
                rvalue_uses(first, &mut uses, &mut defs);
                rvalue_uses(second, &mut uses, &mut defs);
                rvalue_uses(third, &mut uses, &mut defs);
            }
        },
        MirTerminatorKind::Return(outputs) => {
            for output in outputs {
                operand_uses(output, &mut uses);
            }
        }
        MirTerminatorKind::Await { future, result, .. } => {
            operand_uses(future, &mut uses);
            if let Some(result) = result {
                place_uses_defs(result, &mut uses, &mut defs);
            }
        }
        MirTerminatorKind::Unreachable => {}
    }
    (uses, defs)
}

pub(crate) fn successors(kind: &MirTerminatorKind) -> Vec<BasicBlockId> {
    let mut successors = match kind {
        MirTerminatorKind::Goto(target) => vec![*target],
        MirTerminatorKind::Branch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        MirTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, block)| *block)
            .chain(std::iter::once(*otherwise))
            .collect(),
        MirTerminatorKind::For {
            body_block,
            exit_block,
            ..
        }
        | MirTerminatorKind::ParFor {
            body_block,
            exit_block,
            ..
        }
        | MirTerminatorKind::Spmd {
            body_block,
            exit_block,
            ..
        } => vec![*body_block, *exit_block],
        MirTerminatorKind::TryCatch {
            try_block,
            catch_block,
            ..
        } => vec![*try_block, *catch_block],
        MirTerminatorKind::Await { resume, .. } => vec![*resume],
        MirTerminatorKind::Return(_) | MirTerminatorKind::Unreachable => Vec::new(),
    };
    successors.sort();
    successors.dedup();
    successors
}

pub(crate) fn terminator_edge_defs(kind: &MirTerminatorKind, successor: BasicBlockId) -> Locals {
    match kind {
        MirTerminatorKind::For {
            binding,
            body_block,
            ..
        }
        | MirTerminatorKind::ParFor {
            binding,
            body_block,
            ..
        } if *body_block == successor => BTreeSet::from([*binding]),
        MirTerminatorKind::Await { result, resume, .. } if *resume == successor => result
            .as_ref()
            .and_then(place_root_local)
            .into_iter()
            .collect(),
        _ => Locals::new(),
    }
}

pub(crate) fn consumer_sites(
    body: &MirBody,
    function: ProgramFunctionId,
) -> Vec<(ProgramPointId, Locals)> {
    let mut consumers = Vec::new();
    for block in &body.blocks {
        let Ok(block_id) = u32::try_from(block.id.0) else {
            continue;
        };
        for (position, statement) in block.statements.iter().enumerate() {
            let Ok(position) = u32::try_from(position) else {
                continue;
            };
            let (uses, _) = statement_uses_defs(statement);
            if !uses.is_empty() {
                consumers.push((
                    ProgramPointId {
                        function,
                        block: block_id,
                        position,
                    },
                    uses,
                ));
            }
        }
        let (uses, _) = terminator_uses_defs(&block.terminator.kind);
        if !uses.is_empty() {
            if let Ok(position) = u32::try_from(block.statements.len()) {
                consumers.push((
                    ProgramPointId {
                        function,
                        block: block_id,
                        position,
                    },
                    uses,
                ));
            }
        }
    }
    consumers.sort_by_key(|(point, _)| *point);
    consumers
}

fn rvalue_uses(value: &MirRvalue, uses: &mut Locals, defs: &mut Locals) {
    match value {
        MirRvalue::Use(value) | MirRvalue::Unary(_, value) | MirRvalue::Spawn(value) => {
            operand_uses(value, uses)
        }
        MirRvalue::Binary(left, _, right) => {
            operand_uses(left, uses);
            operand_uses(right, uses);
        }
        MirRvalue::ShortCircuit {
            left,
            right_temps,
            right,
            ..
        } => {
            operand_uses(left, uses);
            for statement in right_temps {
                let (nested_uses, nested_defs) = statement_uses_defs(statement);
                uses.extend(
                    nested_uses
                        .into_iter()
                        .filter(|local| !defs.contains(local)),
                );
                defs.extend(nested_defs);
            }
            if !matches!(right, MirOperand::Local(local) if defs.contains(local)) {
                operand_uses(right, uses);
            }
        }
        MirRvalue::Range { start, step, end } => {
            operand_uses(start, uses);
            if let Some(step) = step {
                operand_uses(step, uses);
            }
            operand_uses(end, uses);
        }
        MirRvalue::Call(call) => {
            if let MirCallee::Dynamic(callee) = &call.callee {
                operand_uses(callee, uses);
            }
            for argument in &call.args {
                match argument {
                    MirCallArg::Single(argument) => operand_uses(argument, uses),
                    MirCallArg::Expansion(source) => {
                        source.visit_outer_operands(|operand| operand_uses(operand, uses));
                    }
                    MirCallArg::CapturedSequence(_) => {}
                }
            }
        }
        MirRvalue::Aggregate { elements, .. } => {
            for element in elements {
                if let Some(operand) = element.operand() {
                    operand_uses(operand, uses);
                }
            }
        }
        MirRvalue::StructLiteral { fields } | MirRvalue::ObjectLiteral { fields, .. } => {
            for (_, value) in fields {
                operand_uses(value, uses);
            }
        }
        MirRvalue::Index { base, indexing } => {
            operand_uses(base, uses);
            indexing_uses(indexing, uses);
        }
        MirRvalue::SubscriptChain(chain) => {
            chain.visit_outer_operands(|operand| operand_uses(operand, uses));
        }
        MirRvalue::Member { base, .. } => operand_uses(base, uses),
        MirRvalue::DynamicMember { base, member, .. } => {
            operand_uses(base, uses);
            operand_uses(member, uses);
        }
        MirRvalue::Future { args, .. } => {
            for argument in args {
                match argument {
                    MirCallArg::Single(argument) => operand_uses(argument, uses),
                    MirCallArg::Expansion(source) => {
                        source.visit_outer_operands(|operand| operand_uses(operand, uses));
                    }
                    MirCallArg::CapturedSequence(_) => {}
                }
            }
        }
        MirRvalue::Distributed(operation) => {
            for operand in operation.operands() {
                operand_uses(operand, uses);
            }
        }
        MirRvalue::Collective(operation) => {
            for operand in operation.operands() {
                operand_uses(operand, uses);
            }
        }
        MirRvalue::WorkspaceFirstStaticProperty { .. }
        | MirRvalue::MetaClass(_)
        | MirRvalue::Colon
        | MirRvalue::End => {}
    }
}

fn place_uses_defs(place: &MirPlace, uses: &mut Locals, defs: &mut Locals) {
    match place {
        MirPlace::Local(local) => {
            defs.insert(*local);
        }
        MirPlace::Binding(_) => {}
        MirPlace::Member(base, _) => {
            place_read_uses(base, uses);
            if let Some(root) = place_root_local(base) {
                defs.insert(root);
            }
        }
        MirPlace::DynamicMember(base, member) => {
            place_read_uses(base, uses);
            operand_uses(member, uses);
            if let Some(root) = place_root_local(base) {
                defs.insert(root);
            }
        }
        MirPlace::Index(base, indexing) => {
            place_read_uses(base, uses);
            indexing_uses(indexing, uses);
            if let Some(root) = place_root_local(base) {
                defs.insert(root);
            }
        }
    }
}

fn place_read_uses(place: &MirPlace, uses: &mut Locals) {
    match place {
        MirPlace::Local(local) => {
            uses.insert(*local);
        }
        MirPlace::Binding(_) => {}
        MirPlace::Member(base, _) => place_read_uses(base, uses),
        MirPlace::DynamicMember(base, member) => {
            place_read_uses(base, uses);
            operand_uses(member, uses);
        }
        MirPlace::Index(base, indexing) => {
            place_read_uses(base, uses);
            indexing_uses(indexing, uses);
        }
    }
}

fn place_root_local(place: &MirPlace) -> Option<MirLocalId> {
    match place {
        MirPlace::Local(local) => Some(*local),
        MirPlace::Binding(_) => None,
        MirPlace::Member(base, _) | MirPlace::DynamicMember(base, _) | MirPlace::Index(base, _) => {
            place_root_local(base)
        }
    }
}

fn indexing_uses(indexing: &MirIndexing, uses: &mut Locals) {
    for component in &indexing.components {
        match component {
            MirIndexComponent::Expr(value) => operand_uses(value, uses),
            MirIndexComponent::ContextualExpr(region) => {
                region.visit_external_operands(|operand| operand_uses(operand, uses));
            }
            MirIndexComponent::Colon => {}
        }
    }
}

fn operand_uses(operand: &MirOperand, uses: &mut Locals) {
    if let MirOperand::Local(local) = operand {
        uses.insert(*local);
    }
}
