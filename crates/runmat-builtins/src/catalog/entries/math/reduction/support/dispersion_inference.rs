use super::dispersion_options::{
    std_normalization_literal, var_normalization_literal, NormalizationLiteral,
};
use super::numeric_inference::{
    admitted_literal_controls, infer_numeric_reduction_type, inferred_shape_fact, EmptyAxesPolicy,
    KnownControls, LiteralReductionControl, ScalarResultClass,
};
use super::DispersionControl;
use crate::{ResolveContext, Type};
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

/// Literal type projection shared by std and var, retaining their distinct
/// normalization and text grammars.
pub fn infer_std_type(args: &[Type], context: &ResolveContext) -> Type {
    infer_numeric_reduction_type(
        args,
        context,
        std_literal_control,
        ScalarResultClass::FloatingOnly,
        Some(std_normalization_literal as fn(&LiteralValue) -> NormalizationLiteral),
        1,
        EmptyAxesPolicy::Default,
    )
}

pub fn infer_var_type(args: &[Type], context: &ResolveContext) -> Type {
    infer_numeric_reduction_type(
        args,
        context,
        var_literal_control,
        ScalarResultClass::FloatingOnly,
        Some(var_normalization_literal as fn(&LiteralValue) -> NormalizationLiteral),
        1,
        EmptyAxesPolicy::Reject,
    )
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

#[cfg(test)]
mod tests {
    use super::{infer_std_type, infer_var_type};
    use crate::{ResolveContext, Type};
    use runmat_types::LiteralValue;

    fn input() -> Type {
        Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        }
    }

    #[test]
    fn std_normalization_and_axis_have_distinct_literal_roles() {
        let normalized_then_second = infer_std_type(
            &[input(), Type::Num, Type::Num],
            &ResolveContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(1.0),
                LiteralValue::Number(2.0),
            ]),
        );
        assert_eq!(
            normalized_then_second,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
        let second_only = infer_std_type(
            &[input(), Type::Num],
            &ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]),
        );
        assert_eq!(second_only, normalized_then_second);
    }

    #[test]
    fn var_requires_valid_normalization_before_axis() {
        let admitted = infer_var_type(
            &[input(), Type::Num, Type::Num],
            &ResolveContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(0.0),
                LiteralValue::Number(2.0),
            ]),
        );
        assert_eq!(
            admitted,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
        assert_eq!(
            infer_var_type(
                &[input(), Type::Num],
                &ResolveContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]),
            ),
            Type::Tensor { shape: None }
        );
    }

    #[test]
    fn std_and_var_disagree_on_empty_axis_after_normalization_placeholder() {
        let arguments = [
            input(),
            Type::Tensor {
                shape: Some(vec![Some(0), Some(0)]),
            },
            Type::Tensor {
                shape: Some(vec![Some(0), Some(0)]),
            },
        ];
        let literals = ResolveContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Empty,
            LiteralValue::Empty,
        ]);
        assert_eq!(
            infer_std_type(&arguments, &literals),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
        assert_eq!(
            infer_var_type(&arguments, &literals),
            Type::Tensor { shape: None }
        );
    }

    #[test]
    fn owner_specific_all_spelling_and_conflicts_are_conservative() {
        let spaced_all = vec![LiteralValue::Unknown, LiteralValue::String(" all ".into())];
        assert_eq!(
            infer_std_type(
                &[input(), Type::String],
                &ResolveContext::new(spaced_all.clone())
            ),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1)])
            }
        );
        assert_eq!(
            infer_var_type(&[input(), Type::String], &ResolveContext::new(spaced_all)),
            Type::Tensor { shape: None }
        );
        let conflict = vec![
            LiteralValue::Unknown,
            LiteralValue::Number(0.0),
            LiteralValue::Number(2.0),
            LiteralValue::String("all".into()),
        ];
        assert_eq!(
            infer_var_type(
                &[input(), Type::Num, Type::Num, Type::String],
                &ResolveContext::new(conflict),
            ),
            Type::Tensor { shape: None }
        );
    }

    #[test]
    fn integer_data_class_is_not_admitted_for_std_or_var() {
        assert_eq!(
            infer_std_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
        assert_eq!(
            infer_var_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }
}
