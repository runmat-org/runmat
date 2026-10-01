use super::numeric_inference::{
    admitted_literal_controls, infer_numeric_reduction_type, inferred_shape_fact, EmptyAxesPolicy,
    KnownControls, LiteralReductionControl, ScalarResultClass,
};
use super::{MinComparisonMethod, MinControl};
use crate::{ResolveContext, Type};
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

/// Distinguish the pairwise and reduction call families before proving any
/// class or axis fact. A reduction requires its public empty placeholder.
pub fn infer_min_type(args: &[Type], context: &ResolveContext) -> Type {
    match args.len() {
        0 => Type::Unknown,
        1 => infer_reduction(args, context, 1),
        _ if context
            .literal_args
            .get(1)
            .is_some_and(is_empty_placeholder) =>
        {
            infer_reduction(args, context, 2)
        }
        _ => infer_pairwise(args, context),
    }
}

fn infer_reduction(args: &[Type], context: &ResolveContext, option_start: usize) -> Type {
    infer_numeric_reduction_type(
        args,
        context,
        min_literal_control,
        ScalarResultClass::PreserveInteger,
        None,
        option_start,
        EmptyAxesPolicy::Default,
    )
}

fn min_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match MinControl::from_literal(literal)? {
        MinControl::All | MinControl::Linear => LiteralReductionControl::All,
        MinControl::Missing(_) => LiteralReductionControl::Missing,
        MinControl::ComparisonMethod => LiteralReductionControl::RequiresRuntimeValue,
    })
}

pub(crate) fn is_empty_placeholder(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Empty => true,
        LiteralValue::Vector(values) => values.is_empty(),
        LiteralValue::String(text) | LiteralValue::Character(text) => text.is_empty(),
        _ => false,
    }
}

fn infer_pairwise(args: &[Type], context: &ResolveContext) -> Type {
    if !pairwise_options_admitted(args.len(), context) {
        return Type::Unknown;
    }
    match (args.first(), args.get(1)) {
        (Some(Type::Num), Some(Type::Num)) => Type::Num,
        (Some(Type::Int), Some(Type::Int)) => Type::Int,
        (Some(Type::Bool), Some(Type::Bool)) => Type::Num,
        // Without a known nonempty second literal, a tensor may be min's
        // reduction placeholder, so pairwise shape/class must stay dynamic.
        (Some(Type::Tensor { .. }), Some(Type::Tensor { .. }))
            if context
                .literal_args
                .get(1)
                .is_some_and(|literal| !is_empty_placeholder(literal)) =>
        {
            Type::tensor()
        }
        _ => Type::Unknown,
    }
}

fn pairwise_options_admitted(argument_count: usize, context: &ResolveContext) -> bool {
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

#[cfg(test)]
mod tests {
    use super::infer_min_type;
    use crate::{ResolveContext, Type};
    use runmat_types::LiteralValue;

    fn input() -> Type {
        Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        }
    }

    #[test]
    fn reduction_placeholder_and_axes_prove_only_the_reduction_form() {
        assert_eq!(
            infer_min_type(
                &[input(), Type::Tensor { shape: None }, Type::Num],
                &ResolveContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::Empty,
                    LiteralValue::Number(2.0),
                ]),
            ),
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
        assert_eq!(
            infer_min_type(
                &[input(), Type::Tensor { shape: None }, Type::String],
                &ResolveContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::Empty,
                    LiteralValue::String("linear".into()),
                ]),
            ),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1)])
            }
        );
    }

    #[test]
    fn pairwise_and_reduction_scalar_classes_are_not_conflated() {
        assert_eq!(
            infer_min_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Int
        );
        assert_eq!(
            infer_min_type(&[Type::Bool], &ResolveContext::new(Vec::new())),
            Type::Num
        );
        assert_eq!(
            infer_min_type(&[Type::Num, Type::Num], &ResolveContext::new(Vec::new())),
            Type::Num
        );
        assert_eq!(
            infer_min_type(&[Type::Int, Type::Int], &ResolveContext::new(Vec::new())),
            Type::Int
        );
    }

    #[test]
    fn invalid_pairwise_options_and_conflicting_reduction_axes_stay_dynamic() {
        assert_eq!(
            infer_min_type(
                &[Type::Num, Type::Num, Type::String],
                &ResolveContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::Number(2.0),
                    LiteralValue::String("omitnan".into()),
                ]),
            ),
            Type::Unknown
        );
        assert_eq!(
            infer_min_type(
                &[
                    input(),
                    Type::Tensor { shape: None },
                    Type::Num,
                    Type::String
                ],
                &ResolveContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::Empty,
                    LiteralValue::Number(2.0),
                    LiteralValue::String("all".into()),
                ]),
            ),
            Type::Tensor { shape: None }
        );
    }
}
