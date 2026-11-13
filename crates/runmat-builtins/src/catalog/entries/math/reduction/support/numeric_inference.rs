use super::dispersion_options::NormalizationLiteral;
use crate::ResolveContext;
use runmat_types::LiteralValue;

/// Semantic control categories shared by numerical reduction inference. Each
/// owner maps its own public option identity here after decoding the literal.
pub(super) enum LiteralReductionControl {
    All,
    Missing,
    Output(LiteralReductionOutput),
    RequiresRuntimeValue,
    /// One following concrete real numeric weight payload, not an axis.
    NumericWeights,
}

#[derive(Clone, Copy)]
pub(super) enum EmptyAxesPolicy {
    Default,
    Reject,
}

#[derive(Clone, Copy)]
pub(super) enum AxisSelectionPolicy {
    ReplaceEmptyDefault,
    Once,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiteralReductionOutput {
    Default,
    Double,
    Native,
}

#[derive(Clone)]
enum KnownAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

pub(crate) struct KnownControls {
    axes: KnownAxes,
    pub(crate) output: Option<LiteralReductionOutput>,
}

impl KnownControls {
    pub(crate) fn selects_all(&self) -> bool {
        matches!(self.axes, KnownAxes::All)
    }
}

pub(crate) fn is_empty_placeholder(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Empty => true,
        LiteralValue::Vector(values) => values.is_empty(),
        LiteralValue::String(text) | LiteralValue::Character(text) => text.is_empty(),
        _ => false,
    }
}

/// A known zero in the selected reduction axis is invalid, including vecdim.
/// Unknown controls stay dynamic; a pairwise second operand is never an axis.
pub(crate) fn has_literal_zero_axis(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Vector(values) => values.iter().any(has_literal_zero_axis),
        other => ResolveContext::numeric_from_literal(other) == Some(0.0),
    }
}

pub(super) fn admitted_literal_controls(
    argument_count: usize,
    context: &ResolveContext,
    decode_control: fn(&LiteralValue) -> Option<LiteralReductionControl>,
    decode_normalization: Option<fn(&LiteralValue) -> NormalizationLiteral>,
    option_start: usize,
    empty_axes: EmptyAxesPolicy,
) -> Option<KnownControls> {
    admitted_literal_controls_with_axis_policy(
        argument_count,
        context,
        decode_control,
        decode_normalization,
        option_start,
        empty_axes,
        AxisSelectionPolicy::ReplaceEmptyDefault,
    )
}

pub(super) fn admitted_literal_controls_with_axis_policy(
    argument_count: usize,
    context: &ResolveContext,
    decode_control: fn(&LiteralValue) -> Option<LiteralReductionControl>,
    decode_normalization: Option<fn(&LiteralValue) -> NormalizationLiteral>,
    option_start: usize,
    empty_axes: EmptyAxesPolicy,
    axis_selection: AxisSelectionPolicy,
) -> Option<KnownControls> {
    let mut axes = KnownAxes::Default;
    let mut axes_set = false;
    let mut output_set = false;
    let mut output = None;
    let mut normalization_consumed = false;
    let mut weights_pending = false;
    let mut weights_seen = false;
    for index in option_start..argument_count {
        let literal = context.literal_args.get(index)?;
        if weights_pending {
            let known = match literal {
                LiteralValue::Vector(values) => {
                    !values.is_empty() && values.iter().all(known_weight)
                }
                value => known_weight(value),
            };
            if !known {
                return None;
            }
            weights_pending = false;
            continue;
        }
        if let Some(control) = decode_control(literal) {
            match control {
                LiteralReductionControl::All => {
                    if axes_set
                        && (matches!(axis_selection, AxisSelectionPolicy::Once)
                            || !matches!(axes, KnownAxes::Default))
                    {
                        return None;
                    }
                    axes = KnownAxes::All;
                    axes_set = true;
                }
                LiteralReductionControl::Missing => {}
                LiteralReductionControl::Output(class) => {
                    if output_set {
                        return None;
                    }
                    output = Some(class);
                    output_set = true;
                }
                LiteralReductionControl::RequiresRuntimeValue => return None,
                LiteralReductionControl::NumericWeights => {
                    if weights_seen {
                        return None;
                    }
                    weights_seen = true;
                    weights_pending = true;
                }
            }
            continue;
        }
        if let Some(decode_normalization) = decode_normalization {
            if !normalization_consumed {
                match decode_normalization(literal) {
                    NormalizationLiteral::Value | NormalizationLiteral::Placeholder => {
                        normalization_consumed = true;
                        continue;
                    }
                    NormalizationLiteral::Unmatched => {}
                    NormalizationLiteral::Invalid => return None,
                }
            }
        }
        let selection = axes_literal(literal, empty_axes)?;
        if axes_set
            && (matches!(axis_selection, AxisSelectionPolicy::Once)
                || !matches!(axes, KnownAxes::Default))
        {
            return None;
        }
        axes = selection;
        axes_set = true;
    }
    if weights_pending {
        return None;
    }
    Some(KnownControls { axes, output })
}

fn known_weight(literal: &LiteralValue) -> bool {
    ResolveContext::numeric_from_literal(literal)
        .is_some_and(|value| value.is_finite() && value >= 0.0)
}

fn axes_literal(literal: &LiteralValue, empty_axes: EmptyAxesPolicy) -> Option<KnownAxes> {
    match literal {
        LiteralValue::Empty => {
            matches!(empty_axes, EmptyAxesPolicy::Default).then_some(KnownAxes::Default)
        }
        LiteralValue::Vector(values) if values.is_empty() => {
            matches!(empty_axes, EmptyAxesPolicy::Default).then_some(KnownAxes::Default)
        }
        LiteralValue::Vector(values) => Some(KnownAxes::Vec(
            values
                .iter()
                .map(positive_dimension)
                .collect::<Option<Vec<_>>>()?,
        )),
        other => positive_dimension(other).map(KnownAxes::Dim),
    }
}

fn positive_dimension(literal: &LiteralValue) -> Option<usize> {
    if let LiteralValue::Integer { text, .. } = literal {
        return text
            .parse::<usize>()
            .ok()
            .filter(|dimension| *dimension > 0);
    }
    let number = ResolveContext::numeric_from_literal(literal)?;
    runmat_types::decode_shape_dimension(number)
        .ok()
        .filter(|dimension| *dimension > 0)
}

fn inferred_shape(source: &[Option<usize>], axes: KnownAxes) -> Option<Vec<Option<usize>>> {
    match axes {
        KnownAxes::All => Some(vec![Some(1), Some(1)]),
        KnownAxes::Dim(dimension) => reduce_known_dimensions(source, &[dimension]),
        KnownAxes::Vec(dimensions) => reduce_known_dimensions(source, &dimensions),
        KnownAxes::Default => {
            let mut shape = source.to_vec();
            for (index, dimension) in source.iter().enumerate() {
                match dimension {
                    Some(1) => continue,
                    Some(_) => {
                        shape[index] = Some(1);
                        return Some(shape);
                    }
                    None => return None,
                }
            }
            if shape.is_empty() {
                return Some(vec![Some(1), Some(1)]);
            }
            Some(shape)
        }
    }
}

fn reduce_known_dimensions(
    source: &[Option<usize>],
    dimensions: &[usize],
) -> Option<Vec<Option<usize>>> {
    let mut shape = source.to_vec();
    for dimension in dimensions {
        let index = dimension.checked_sub(1)?;
        if let Some(extent) = shape.get_mut(index) {
            *extent = Some(1);
        }
    }
    Some(shape)
}

pub(super) fn inferred_shape_fact(
    source: &runmat_types::ShapeFact,
    controls: &KnownControls,
) -> runmat_types::ShapeFact {
    use runmat_types::ShapeFact;
    if matches!(controls.axes, KnownAxes::All) {
        return ShapeFact::Scalar;
    }
    if matches!(source, ShapeFact::Ranked { .. }) {
        return source.clone();
    }
    let Some(dims) = source.known_dims() else {
        return ShapeFact::Unknown;
    };
    inferred_shape(&dims, controls.axes.clone()).map_or_else(
        || {
            source
                .rank()
                .map_or(ShapeFact::Unknown, |rank| ShapeFact::Ranked { rank })
        },
        ShapeFact::from,
    )
}

#[cfg(test)]
mod tests {
    use super::positive_dimension;
    use runmat_types::LiteralValue;

    #[test]
    fn literal_axes_share_shape_tolerance_and_platform_range() {
        assert_eq!(
            positive_dimension(&LiteralValue::Number(1.0 + f64::EPSILON)),
            Some(1)
        );
        assert_eq!(
            positive_dimension(&LiteralValue::Number(1.0 + 2.0 * f64::EPSILON)),
            None
        );
        assert_eq!(positive_dimension(&LiteralValue::Number(0.0)), None);
        let expected = (usize::BITS < 64).then_some(usize::MAX);
        assert_eq!(
            positive_dimension(&LiteralValue::Number(usize::MAX as f64)),
            expected
        );
    }
}
