use std::collections::BTreeSet;

use crate::{MirLocalId, MirOperand, MirOutputTarget, MirPlace, MirStmtKind, MirTerminatorKind};

pub(super) fn visit_outer_statement_sequences(
    statement: &crate::MirStmt,
    visitor: &mut dyn FnMut(crate::MirSequenceLocalId),
) {
    let value = match &statement.kind {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => Some(value),
        MirStmtKind::CaptureSequence { .. }
        | MirStmtKind::PlaceMutation(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => None,
    };
    if let Some(value) = value {
        value.visit_sequence_locals(|sequence| visitor(*sequence));
    }
}

pub(super) fn visit_outer_statement_definitions(
    statement: &crate::MirStmt,
    visitor: &mut dyn FnMut(MirLocalId),
) {
    match &statement.kind {
        MirStmtKind::Assign {
            place: MirPlace::Local(local),
            ..
        } => visitor(*local),
        MirStmtKind::MultiAssign { targets, .. } => targets.targets.iter().for_each(|target| {
            if let MirOutputTarget::Place(MirPlace::Local(local)) = target {
                visitor(*local);
            }
        }),
        _ => {}
    }
}

pub(super) fn statement_outer_rvalue(statement: &crate::MirStmt) -> Option<&crate::MirRvalue> {
    match &statement.kind {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => Some(value),
        _ => None,
    }
}

pub(super) fn visit_terminator_regions(
    terminator: &MirTerminatorKind,
    visitor: &mut dyn FnMut(&crate::MirExpressionRegion),
) {
    match terminator {
        MirTerminatorKind::For { iterable, .. } => iterable.visit_expression_regions_dyn(visitor),
        MirTerminatorKind::ParFor {
            iterable,
            maximum_workers,
            ..
        } => {
            iterable.visit_expression_regions_dyn(visitor);
            if let Some(maximum_workers) = maximum_workers {
                maximum_workers.visit_expression_regions_dyn(visitor);
            }
        }
        MirTerminatorKind::Spmd { header, .. } => {
            use crate::parallel::MirSpmdHeader;
            match header.as_ref() {
                MirSpmdHeader::Default => {}
                MirSpmdHeader::One(first) => first.visit_expression_regions_dyn(visitor),
                MirSpmdHeader::Two(first, second) => {
                    first.visit_expression_regions_dyn(visitor);
                    second.visit_expression_regions_dyn(visitor);
                }
                MirSpmdHeader::Three(first, second, third) => {
                    first.visit_expression_regions_dyn(visitor);
                    second.visit_expression_regions_dyn(visitor);
                    third.visit_expression_regions_dyn(visitor);
                }
            }
        }
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::Branch { .. }
        | MirTerminatorKind::Switch { .. }
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Return(_)
        | MirTerminatorKind::Await { .. }
        | MirTerminatorKind::Unreachable => {}
    }
}

pub(super) fn terminator_region_local_escape(
    terminator: &MirTerminatorKind,
    region_locals: &BTreeSet<MirLocalId>,
) -> Option<(MirLocalId, bool)> {
    let mut escaped = None;
    let mut visit_operand = |operand: &MirOperand| {
        if let MirOperand::Local(local) = operand {
            if region_locals.contains(local) && escaped.is_none() {
                escaped = Some((*local, false));
            }
        }
    };
    match terminator {
        MirTerminatorKind::Branch { cond, .. } => visit_operand(cond),
        MirTerminatorKind::Switch { discr, cases, .. } => {
            visit_operand(discr);
            for (value, _) in cases {
                visit_operand(value);
            }
        }
        MirTerminatorKind::For {
            binding, iterable, ..
        } => {
            if region_locals.contains(binding) {
                return Some((*binding, true));
            }
            iterable.visit_outer_operands_dyn(&mut visit_operand);
        }
        MirTerminatorKind::ParFor {
            binding,
            iterable,
            maximum_workers,
            ..
        } => {
            if region_locals.contains(binding) {
                return Some((*binding, true));
            }
            iterable.visit_outer_operands_dyn(&mut visit_operand);
            if let Some(maximum_workers) = maximum_workers {
                maximum_workers.visit_outer_operands_dyn(&mut visit_operand);
            }
        }
        MirTerminatorKind::Spmd { header, .. } => {
            use crate::parallel::MirSpmdHeader;
            match header.as_ref() {
                MirSpmdHeader::Default => {}
                MirSpmdHeader::One(first) => first.visit_outer_operands_dyn(&mut visit_operand),
                MirSpmdHeader::Two(first, second) => {
                    first.visit_outer_operands_dyn(&mut visit_operand);
                    second.visit_outer_operands_dyn(&mut visit_operand);
                }
                MirSpmdHeader::Three(first, second, third) => {
                    first.visit_outer_operands_dyn(&mut visit_operand);
                    second.visit_outer_operands_dyn(&mut visit_operand);
                    third.visit_outer_operands_dyn(&mut visit_operand);
                }
            }
        }
        MirTerminatorKind::Return(values) => values.iter().for_each(&mut visit_operand),
        MirTerminatorKind::Await { future, result, .. } => {
            visit_operand(future);
            if let Some(MirPlace::Local(local)) = result {
                if region_locals.contains(local) {
                    return Some((*local, true));
                }
            }
        }
        MirTerminatorKind::Goto(_)
        | MirTerminatorKind::TryCatch { .. }
        | MirTerminatorKind::Unreachable => {}
    }
    escaped
}
