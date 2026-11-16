use runmat_types::{LiteralContext, LiteralValue, ScalarIndexBounds};

use crate::analysis::engine::FlowState;
use crate::{MirOperand, MirRvalue};

use super::value::{operand_literal, rvalue_literal};

pub(crate) fn index_selectors(
    indexing: &crate::MirIndexing,
    state: &FlowState,
) -> Vec<runmat_types::IndexSelectorFact> {
    crate::analysis::dataflow::index_selectors_with_bounds(
        indexing,
        &state.value_facts(),
        &|local| state.locals[local.0].index_bounds,
    )
}

pub(crate) fn literal_index_position(literal: &LiteralValue) -> Option<usize> {
    let index = if matches!(literal, LiteralValue::Integer { .. }) {
        usize::try_from(literal.integer_value_i128()?).ok()?
    } else {
        runmat_types::decode_shape_dimension(LiteralContext::numeric_from_literal(literal)?).ok()?
    };
    Some(index)
}

pub(crate) fn literal_index_bounds(literal: &LiteralValue) -> Option<ScalarIndexBounds> {
    let index = literal_index_position(literal)?;
    ScalarIndexBounds::new(index, index)
}

/// Positive integer element bounds. An array's bounds become scalar selector
/// evidence only after its shape proves that it contains one element.
pub(crate) fn rvalue_index_bounds(
    value: &MirRvalue,
    state: &FlowState,
) -> Option<ScalarIndexBounds> {
    match value {
        MirRvalue::Use(MirOperand::Local(local))
        | MirRvalue::Unary(runmat_types::OperatorKind::UnaryPlus, MirOperand::Local(local)) => {
            state.locals[local.0].index_bounds
        }
        MirRvalue::Range { start, step, end } => {
            let start = literal_index_bounds(&operand_literal(start, state))?.minimum();
            let end = literal_index_bounds(&operand_literal(end, state))?.maximum();
            let positive = match step {
                None => true,
                Some(operand) => {
                    let literal = operand_literal(operand, state);
                    if matches!(literal, LiteralValue::Integer { .. }) {
                        let integer = literal.integer_value_i128()?;
                        if integer == 0 {
                            return None;
                        }
                        integer > 0
                    } else {
                        let step = LiteralContext::numeric_from_literal(&literal)?;
                        if !step.is_finite() || step == 0.0 || step.fract() != 0.0 {
                            return None;
                        }
                        step > 0.0
                    }
                }
            };
            if (positive && start > end) || (!positive && start < end) {
                return None;
            }
            ScalarIndexBounds::new(start.min(end), start.max(end))
        }
        _ => literal_index_bounds(&rvalue_literal(value, state)),
    }
}

pub(crate) fn join_index_bounds(
    left: Option<ScalarIndexBounds>,
    right: Option<ScalarIndexBounds>,
    widen: bool,
) -> Option<ScalarIndexBounds> {
    match (left, right) {
        (Some(left), Some(right)) if left == right || !widen => Some(left.enclose(right)),
        _ => None,
    }
}
