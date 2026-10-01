use super::{MeanControl, MeanOutputClass};
use crate::{ResolveContext, Type};
use runmat_types::LiteralValue;

#[derive(Clone)]
enum KnownAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Clone)]
struct KnownControls {
    axes: KnownAxes,
    output: MeanOutputClass,
}

/// Conservative legacy type projection for `mean` until its complete catalog
/// entry is installed. Option words use the same typed identity as runtime
/// admission; any control requiring runtime values leaves the shape dynamic.
pub fn infer_mean_type(args: &[Type], context: &ResolveContext) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    let controls = admitted_literal_controls(args.len(), context);
    match input {
        Type::Tensor { shape } | Type::Logical { shape } => Type::Tensor {
            shape: shape.as_ref().and_then(|dims| {
                controls
                    .as_ref()
                    .and_then(|controls| inferred_shape(dims, controls.axes.clone()))
            }),
        },
        Type::Int
            if controls
                .as_ref()
                .is_some_and(|controls| controls.output == MeanOutputClass::Native) =>
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
) -> Option<KnownControls> {
    let mut axes = KnownAxes::Default;
    let mut axes_set = false;
    let mut output_set = false;
    let mut output = MeanOutputClass::Default;
    for index in 1..argument_count {
        let literal = context.literal_args.get(index)?;
        if let Some(control) = MeanControl::from_literal(literal) {
            match control {
                MeanControl::All => {
                    if axes_set && !matches!(axes, KnownAxes::Default) {
                        return None;
                    }
                    axes = KnownAxes::All;
                    axes_set = true;
                }
                MeanControl::Weights => return None,
                MeanControl::Missing(_) => {}
                MeanControl::Output(class) => {
                    if output_set {
                        return None;
                    }
                    output = class;
                    output_set = true;
                }
            }
            continue;
        }
        let selection = axes_literal(literal)?;
        if axes_set && !matches!(axes, KnownAxes::Default) {
            return None;
        }
        axes = selection;
        axes_set = true;
    }
    Some(KnownControls { axes, output })
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

#[cfg(test)]
mod tests {
    use super::infer_mean_type;
    use crate::{ResolveContext, Type};
    use runmat_types::LiteralValue;

    fn input() -> Type {
        Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        }
    }

    #[test]
    fn typed_literal_controls_preserve_known_mean_shapes() {
        let all = infer_mean_type(
            &[input(), Type::String, Type::String],
            &ResolveContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Character(" all ".into()),
                LiteralValue::String("omitmissing".into()),
            ]),
        );
        assert_eq!(
            all,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1)])
            }
        );

        let dimension = infer_mean_type(
            &[input(), Type::Num],
            &ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]),
        );
        assert_eq!(
            dimension,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    #[test]
    fn conflicting_and_dynamic_controls_do_not_prove_shape() {
        for literals in [
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("all".into()),
                LiteralValue::Number(2.0),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("native".into()),
                LiteralValue::String("double".into()),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("all".into()),
            ],
        ] {
            assert_eq!(
                infer_mean_type(
                    &[input(), Type::Unknown, Type::Unknown],
                    &ResolveContext::new(literals)
                ),
                Type::Tensor { shape: None }
            );
        }
    }

    #[test]
    fn weights_and_unknown_leading_dimension_remain_dynamic() {
        let weighted = infer_mean_type(
            &[input(), Type::String, Type::Tensor { shape: None }],
            &ResolveContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("Weights".into()),
                LiteralValue::Unknown,
            ]),
        );
        assert_eq!(weighted, Type::Tensor { shape: None });

        let uncertain_input = Type::Tensor {
            shape: Some(vec![None, Some(4)]),
        };
        assert_eq!(
            infer_mean_type(&[uncertain_input], &ResolveContext::new(Vec::new())),
            Type::Tensor { shape: None }
        );
    }

    #[test]
    fn native_integer_scalar_preserves_its_class_in_legacy_type_projection() {
        assert_eq!(
            infer_mean_type(
                &[Type::Int, Type::String],
                &ResolveContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::String("native".into()),
                ]),
            ),
            Type::Int
        );
        assert_eq!(
            infer_mean_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}
