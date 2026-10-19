use super::numeric_inference::{
    admitted_literal_controls, inferred_shape_fact, EmptyAxesPolicy, KnownControls,
    LiteralReductionControl,
};
use super::{MinComparisonMethod, MinControl};
use crate::ResolveContext;
use runmat_types::{LiteralValue, ShapeFact};

pub(crate) fn admitted_min_reduction_controls(
    argument_count: usize,
    context: &ResolveContext,
    option_start: usize,
) -> Option<KnownControls> {
    admitted_literal_controls(
        argument_count,
        context,
        min_literal_control,
        None,
        option_start,
        EmptyAxesPolicy::Default,
    )
}

pub(crate) fn inferred_min_shape(source: &ShapeFact, controls: &KnownControls) -> ShapeFact {
    inferred_shape_fact(source, controls)
}

pub(crate) fn admitted_min_pairwise_options(
    argument_count: usize,
    context: &ResolveContext,
) -> bool {
    let mut index = 2;
    while index < argument_count {
        if !matches!(
            context
                .literal_args
                .get(index)
                .and_then(MinControl::from_literal),
            Some(MinControl::ComparisonMethod)
        ) || context
            .literal_args
            .get(index + 1)
            .and_then(MinComparisonMethod::from_literal)
            .is_none()
        {
            return false;
        }
        index += 2;
    }
    true
}

fn min_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match MinControl::from_literal(literal)? {
        MinControl::All | MinControl::Linear => LiteralReductionControl::All,
        MinControl::Missing(_) => LiteralReductionControl::Missing,
        MinControl::ComparisonMethod => LiteralReductionControl::RequiresRuntimeValue,
    })
}
