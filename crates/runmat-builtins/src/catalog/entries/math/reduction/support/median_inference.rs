use super::numeric_inference::{
    admitted_literal_controls, inferred_shape_fact, EmptyAxesPolicy, KnownControls,
    LiteralReductionControl,
};
use super::MedianControl;
use crate::ResolveContext;
use runmat_types::{LiteralValue, ShapeFact};

/// Only literal forms whose entire ordered option grammar is known can
/// establish a median result. Weighted/value-dependent forms remain dynamic.
pub(crate) fn admitted_median_controls(
    argument_count: usize,
    context: &ResolveContext,
) -> Option<KnownControls> {
    admitted_literal_controls(
        argument_count,
        context,
        median_literal_control,
        None,
        1,
        EmptyAxesPolicy::Default,
    )
}

pub(crate) fn inferred_median_shape(source: &ShapeFact, controls: &KnownControls) -> ShapeFact {
    inferred_shape_fact(source, controls)
}

fn median_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match MedianControl::from_literal(literal)? {
        MedianControl::All => LiteralReductionControl::All,
        MedianControl::Missing(_) => LiteralReductionControl::Missing,
        MedianControl::Weights => LiteralReductionControl::RequiresRuntimeValue,
    })
}
