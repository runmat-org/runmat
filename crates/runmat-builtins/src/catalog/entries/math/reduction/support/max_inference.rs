use super::super::max::{MaxComparisonMethod, MaxControl};
use super::numeric_inference::{
    admitted_literal_controls_with_axis_policy, inferred_shape_fact, AxisSelectionPolicy,
    EmptyAxesPolicy, KnownControls, LiteralReductionControl,
};
use crate::ResolveContext;
use runmat_types::{LiteralValue, ShapeFact};

pub(crate) fn admitted_max_reduction_controls(
    argument_count: usize,
    context: &ResolveContext,
    option_start: usize,
) -> Option<KnownControls> {
    admitted_literal_controls_with_axis_policy(
        argument_count,
        context,
        max_literal_control,
        None,
        option_start,
        EmptyAxesPolicy::Default,
        AxisSelectionPolicy::Once,
    )
}

pub(crate) fn inferred_max_shape(source: &ShapeFact, controls: &KnownControls) -> ShapeFact {
    inferred_shape_fact(source, controls)
}

pub(crate) fn inferred_empty_max_shape(
    source: &ShapeFact,
    controls: &KnownControls,
    native_integer: bool,
) -> ShapeFact {
    if controls.selects_all() && !native_integer {
        // Floating/complex host empty-all currently attempts a nonempty output
        // shape and can reject; do not invent a scalar or zero extent.
        ShapeFact::Unknown
    } else {
        // Empty integer extrema and all other successful empty host modes
        // preserve the source shape exactly.
        source.clone()
    }
}

pub(crate) fn admitted_max_pairwise_options(
    argument_count: usize,
    context: &ResolveContext,
) -> bool {
    let mut index = 2;
    while index < argument_count {
        match context
            .literal_args
            .get(index)
            .and_then(MaxControl::from_literal)
        {
            Some(MaxControl::Missing(_)) => index += 1,
            Some(MaxControl::ComparisonMethod)
                if context
                    .literal_args
                    .get(index + 1)
                    .and_then(MaxComparisonMethod::from_literal)
                    .is_some() =>
            {
                index += 2;
            }
            _ => return false,
        }
    }
    true
}

fn max_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match MaxControl::from_literal(literal)? {
        MaxControl::All | MaxControl::Linear => LiteralReductionControl::All,
        MaxControl::Missing(_) => LiteralReductionControl::Missing,
        MaxControl::ComparisonMethod => LiteralReductionControl::RequiresRuntimeValue,
    })
}
