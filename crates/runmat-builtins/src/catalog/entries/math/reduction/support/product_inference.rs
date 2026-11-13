use super::super::numerical_operation::{ProductControl, ProductOutputClass};
use super::numeric_inference::{
    admitted_literal_controls, inferred_shape_fact, EmptyAxesPolicy, KnownControls,
    LiteralReductionControl, LiteralReductionOutput,
};
use crate::ResolveContext;
use runmat_types::{LiteralValue, ShapeFact};

pub(crate) fn admitted_product_controls(
    count: usize,
    context: &ResolveContext,
) -> Option<KnownControls> {
    admitted_literal_controls(count, context, decode, None, 1, EmptyAxesPolicy::Default)
}

pub(crate) fn inferred_product_shape(source: &ShapeFact, controls: &KnownControls) -> ShapeFact {
    inferred_shape_fact(source, controls)
}

fn decode(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    let text = match literal {
        LiteralValue::String(text)
        | LiteralValue::Character(text)
        | LiteralValue::Keyword(text) => text,
        _ => return None,
    };
    Some(match ProductControl::from_public_option(text)? {
        ProductControl::All => LiteralReductionControl::All,
        ProductControl::Missing(_) => LiteralReductionControl::Missing,
        ProductControl::Output(class) => LiteralReductionControl::Output(match class {
            ProductOutputClass::Default => LiteralReductionOutput::Default,
            ProductOutputClass::Double => LiteralReductionOutput::Double,
            ProductOutputClass::Native => LiteralReductionOutput::Native,
        }),
    })
}
