use super::MirExpressionStep;
use crate::MirOperand;
use std::collections::BTreeSet;

pub(super) fn validate(
    steps: &[MirExpressionStep],
    result: &MirOperand,
    require_contextual_result: bool,
) -> Result<(), String> {
    validate_with_result(steps, Some(result), require_contextual_result)
}

pub(super) fn validate_steps_only(steps: &[MirExpressionStep]) -> Result<(), String> {
    validate_with_result(steps, None, false)
}

fn validate_with_result(
    steps: &[MirExpressionStep],
    result: Option<&MirOperand>,
    require_contextual_result: bool,
) -> Result<(), String> {
    let all_locals = steps
        .iter()
        .filter_map(|step| match step {
            MirExpressionStep::Let { local, .. } => Some(*local),
            MirExpressionStep::CaptureSequence { .. } => None,
        })
        .collect::<BTreeSet<_>>();
    let all_sequences = steps
        .iter()
        .filter_map(|step| match step {
            MirExpressionStep::CaptureSequence { destination, .. } => Some(*destination),
            MirExpressionStep::Let { .. } => None,
        })
        .collect::<BTreeSet<_>>();
    let mut locals = BTreeSet::new();
    let mut sequences = BTreeSet::new();
    let mut contextual_locals = BTreeSet::new();
    let mut contextual_sequences = BTreeSet::new();
    let mut sequence_uses = std::collections::BTreeMap::new();
    for step in steps {
        let mut invalid_local = None;
        let mut invalid_sequence = None;
        match step {
            MirExpressionStep::Let { local, value, .. } => {
                reject_nested_fixed_preparation(value)?;
                value.visit_operands(|operand| {
                    if let MirOperand::Local(candidate) = operand {
                        if all_locals.contains(candidate) && !locals.contains(candidate) {
                            invalid_local = Some(*candidate);
                        }
                    }
                });
                value.visit_sequence_locals(|candidate| {
                    *sequence_uses.entry(*candidate).or_insert(0usize) += 1;
                    if all_sequences.contains(candidate) && !sequences.contains(candidate) {
                        invalid_sequence = Some(*candidate);
                    }
                });
                let mut depends_on_context = value.contains_contextual_end();
                value.visit_operands(|operand| {
                    if matches!(operand, MirOperand::Local(candidate) if contextual_locals.contains(candidate)) {
                        depends_on_context = true;
                    }
                });
                value.visit_sequence_locals(|candidate| {
                    if contextual_sequences.contains(candidate) {
                        depends_on_context = true;
                    }
                });
                if !locals.insert(*local) {
                    return Err(format!(
                        "expression region local {local:?} is defined more than once"
                    ));
                }
                if depends_on_context {
                    contextual_locals.insert(*local);
                }
            }
            MirExpressionStep::CaptureSequence {
                destination,
                source,
                ..
            } => {
                source.visit_operands(|operand| {
                    if let MirOperand::Local(candidate) = operand {
                        if all_locals.contains(candidate) && !locals.contains(candidate) {
                            invalid_local = Some(*candidate);
                        }
                    }
                });
                let mut depends_on_context = source.contains_contextual_end();
                source.visit_operands(|operand| {
                    if matches!(operand, MirOperand::Local(candidate) if contextual_locals.contains(candidate)) {
                        depends_on_context = true;
                    }
                });
                if !sequences.insert(*destination) {
                    return Err(format!(
                        "expression region sequence {destination:?} is defined more than once"
                    ));
                }
                if depends_on_context {
                    contextual_sequences.insert(*destination);
                }
            }
        }
        if let Some(local) = invalid_local {
            return Err(format!(
                "expression region local {local:?} is used before its definition"
            ));
        }
        if let Some(sequence) = invalid_sequence {
            return Err(format!(
                "expression region sequence {sequence:?} is used before its capture"
            ));
        }
    }
    for sequence in all_sequences {
        let uses = sequence_uses.get(&sequence).copied().unwrap_or(0);
        if uses != 1 {
            return Err(format!(
                "expression region sequence {sequence:?} is consumed {uses} times; expected exactly once"
            ));
        }
    }
    if let Some(MirOperand::Local(local)) = result {
        if all_locals.contains(local) && !locals.contains(local) {
            return Err(format!(
                "expression region result local {local:?} is not defined"
            ));
        }
    }
    let result_depends_on_context = match result {
        Some(MirOperand::Local(local)) => contextual_locals.contains(local),
        _ => false,
    };
    if require_contextual_result && !result_depends_on_context {
        return Err(
            "contextual expression result does not depend on the indexing context".to_owned(),
        );
    }
    Ok(())
}

pub(crate) fn reject_nested_fixed_preparation(value: &crate::MirRvalue) -> Result<(), String> {
    if let crate::MirRvalue::ShortCircuit { right_temps, .. } = value {
        for statement in right_temps {
            match &statement.kind {
                crate::MirStmtKind::PrepareFixedAssignment(_)
                | crate::MirStmtKind::PreparedAssign { .. } => {
                    return Err("fixed preparation roles require an explicit CFG statement".into());
                }
                crate::MirStmtKind::Assign { value, .. }
                | crate::MirStmtKind::MultiAssign { value, .. }
                | crate::MirStmtKind::SequenceAssign { value, .. }
                | crate::MirStmtKind::Expr(value) => reject_nested_fixed_preparation(value)?,
                _ => {}
            }
        }
    }
    Ok(())
}
