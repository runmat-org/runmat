use super::dispersion_options::{std_normalization_literal, var_normalization_literal};
use super::numeric_inference::{
    admitted_literal_controls, inferred_shape_fact, EmptyAxesPolicy, KnownControls,
    LiteralReductionControl,
};
use super::DispersionControl;
use crate::ResolveContext;
use runmat_types::{LiteralValue, ShapeFact};

pub(crate) fn admitted_std_controls(
    argument_count: usize,
    context: &ResolveContext,
) -> Option<KnownControls> {
    admitted_literal_controls(
        argument_count,
        context,
        std_literal_control,
        Some(std_normalization_literal),
        1,
        EmptyAxesPolicy::Default,
    )
}

pub(crate) fn admitted_var_controls(
    argument_count: usize,
    context: &ResolveContext,
) -> Option<KnownControls> {
    admitted_literal_controls(
        argument_count,
        context,
        var_literal_control,
        Some(var_normalization_literal),
        1,
        EmptyAxesPolicy::Reject,
    )
}

pub(crate) fn inferred_dispersion_shape(source: &ShapeFact, controls: &KnownControls) -> ShapeFact {
    inferred_shape_fact(source, controls)
}

fn std_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match DispersionControl::from_std_literal(literal)? {
        DispersionControl::All => LiteralReductionControl::All,
        DispersionControl::Missing(_) => LiteralReductionControl::Missing,
    })
}

fn var_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match DispersionControl::from_var_literal(literal)? {
        DispersionControl::All => LiteralReductionControl::All,
        DispersionControl::Missing(_) => LiteralReductionControl::Missing,
    })
}
