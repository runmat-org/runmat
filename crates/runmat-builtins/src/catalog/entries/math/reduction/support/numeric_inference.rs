use super::dispersion_options::NormalizationLiteral;
use crate::{ResolveContext, Type};
use runmat_types::LiteralValue;

/// Semantic control categories shared by numerical reduction inference. Each
/// owner maps its own public option identity here after decoding the literal.
pub(super) enum LiteralReductionControl {
    All,
    Missing,
    Output { native: bool },
    RequiresRuntimeValue,
}

/// The scalar result boundary differs between owners. `mean` and `sum`
/// promote by default; `median` preserves exact integer and logical scalars.
/// `std` and `var` reject integer data rather than silently promoting it.
pub(super) enum ScalarResultClass {
    Promoted,
    Preserved,
    FloatingOnly,
}

#[derive(Clone)]
enum KnownAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

struct KnownControls {
    axes: KnownAxes,
    native_output: bool,
}

/// Prove only class and shape facts supported by a complete, ordered literal
/// grammar. Unknown controls, conflicting axes/classes, and value-dependent
/// options leave facts dynamic rather than guessing from any-position text.
pub(super) fn infer_numeric_reduction_type(
    args: &[Type],
    context: &ResolveContext,
    decode_control: fn(&LiteralValue) -> Option<LiteralReductionControl>,
    scalar_class: ScalarResultClass,
    decode_normalization: Option<fn(&LiteralValue) -> NormalizationLiteral>,
) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    let controls =
        admitted_literal_controls(args.len(), context, decode_control, decode_normalization);
    match input {
        Type::Tensor { shape } => Type::Tensor {
            shape: shape.as_ref().and_then(|dims| {
                controls
                    .as_ref()
                    .and_then(|controls| inferred_shape(dims, controls.axes.clone()))
            }),
        },
        Type::Logical { shape } if !matches!(scalar_class, ScalarResultClass::Preserved) => {
            Type::Tensor {
                shape: shape.as_ref().and_then(|dims| {
                    controls
                        .as_ref()
                        .and_then(|controls| inferred_shape(dims, controls.axes.clone()))
                }),
            }
        }
        // A logical median can collapse to Bool or retain LogicalArray; the
        // legacy Type cannot safely express both without an admitted extent.
        Type::Logical { .. } => Type::Unknown,
        Type::Int if matches!(scalar_class, ScalarResultClass::Preserved) && controls.is_some() => {
            Type::Int
        }
        Type::Bool
            if matches!(scalar_class, ScalarResultClass::Preserved) && controls.is_some() =>
        {
            Type::Bool
        }
        Type::Int if matches!(scalar_class, ScalarResultClass::FloatingOnly) => Type::Unknown,
        Type::Int
            if controls
                .as_ref()
                .is_some_and(|controls| controls.native_output) =>
        {
            Type::Int
        }
        Type::Num | Type::Int | Type::Bool if controls.is_some() => Type::Num,
        _ => Type::Unknown,
    }
}

fn admitted_literal_controls(
    argument_count: usize,
    context: &ResolveContext,
    decode_control: fn(&LiteralValue) -> Option<LiteralReductionControl>,
    decode_normalization: Option<fn(&LiteralValue) -> NormalizationLiteral>,
) -> Option<KnownControls> {
    let mut axes = KnownAxes::Default;
    let mut axes_set = false;
    let mut output_set = false;
    let mut native_output = false;
    let mut normalization_consumed = false;
    for index in 1..argument_count {
        let literal = context.literal_args.get(index)?;
        if let Some(control) = decode_control(literal) {
            match control {
                LiteralReductionControl::All => {
                    if axes_set && !matches!(axes, KnownAxes::Default) {
                        return None;
                    }
                    axes = KnownAxes::All;
                    axes_set = true;
                }
                LiteralReductionControl::Missing => {}
                LiteralReductionControl::Output { native } => {
                    if output_set {
                        return None;
                    }
                    native_output = native;
                    output_set = true;
                }
                LiteralReductionControl::RequiresRuntimeValue => return None,
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
        let selection = axes_literal(literal)?;
        if axes_set && !matches!(axes, KnownAxes::Default) {
            return None;
        }
        axes = selection;
        axes_set = true;
    }
    Some(KnownControls {
        axes,
        native_output,
    })
}

fn axes_literal(literal: &LiteralValue) -> Option<KnownAxes> {
    match literal {
        LiteralValue::Empty => Some(KnownAxes::Default),
        LiteralValue::Vector(values) if values.is_empty() => Some(KnownAxes::Default),
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
    if !number.is_finite() || number < 1.0 || number >= usize::MAX as f64 || number.fract() != 0.0 {
        return None;
    }
    let dimension = number as usize;
    (dimension as f64 == number).then_some(dimension)
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
        *shape.get_mut(index)? = Some(1);
    }
    Some(shape)
}
