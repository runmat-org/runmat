use super::numeric_inference::{LiteralReductionControl, LiteralReductionOutput};
use super::{SumControl, SumOutputClass};
use crate::ResolveContext;
use runmat_types::LiteralValue;
use runmat_types::ShapeFact;

pub(crate) fn admitted_sum_controls(
    argument_count: usize,
    context: &ResolveContext,
) -> Option<super::numeric_inference::KnownControls> {
    super::numeric_inference::admitted_literal_controls(
        argument_count,
        context,
        sum_literal_control,
        None,
        1,
        super::numeric_inference::EmptyAxesPolicy::Default,
    )
}

pub(crate) fn inferred_sum_shape(
    source: &ShapeFact,
    controls: &super::numeric_inference::KnownControls,
) -> ShapeFact {
    super::numeric_inference::inferred_shape_fact(source, controls)
}

fn sum_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match SumControl::from_literal(literal)? {
        SumControl::All => LiteralReductionControl::All,
        SumControl::Missing(_) => LiteralReductionControl::Missing,
        SumControl::Output(class) => LiteralReductionControl::Output(match class {
            SumOutputClass::Default => LiteralReductionOutput::Default,
            SumOutputClass::Double => LiteralReductionOutput::Double,
            SumOutputClass::Native => LiteralReductionOutput::Native,
        }),
    })
}
