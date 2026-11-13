use super::numeric_inference::{LiteralReductionControl, LiteralReductionOutput};
use super::{MeanControl, MeanOutputClass};
use crate::ResolveContext;
use runmat_types::LiteralValue;
use runmat_types::ShapeFact;

pub(crate) fn admitted_mean_controls(
    argument_count: usize,
    context: &ResolveContext,
) -> Option<super::numeric_inference::KnownControls> {
    super::numeric_inference::admitted_literal_controls(
        argument_count,
        context,
        mean_literal_control,
        None,
        1,
        super::numeric_inference::EmptyAxesPolicy::Default,
    )
}

pub(crate) fn inferred_mean_shape(
    source: &ShapeFact,
    controls: &super::numeric_inference::KnownControls,
) -> ShapeFact {
    super::numeric_inference::inferred_shape_fact(source, controls)
}

fn mean_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match MeanControl::from_literal(literal)? {
        MeanControl::All => LiteralReductionControl::All,
        MeanControl::Missing(_) => LiteralReductionControl::Missing,
        MeanControl::Weights => LiteralReductionControl::NumericWeights,
        MeanControl::Output(class) => LiteralReductionControl::Output(match class {
            MeanOutputClass::Default => LiteralReductionOutput::Default,
            MeanOutputClass::Double => LiteralReductionOutput::Double,
            MeanOutputClass::Native => LiteralReductionOutput::Native,
        }),
    })
}
