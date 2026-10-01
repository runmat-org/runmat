use super::numeric_inference::{infer_numeric_reduction_type, LiteralReductionControl};
use super::{MeanControl, MeanOutputClass};
use crate::{ResolveContext, Type};
use runmat_types::LiteralValue;

/// Conservative legacy type projection for `mean` until its complete catalog
/// entry is installed. Option words use the same typed identity as runtime
/// admission; any control requiring runtime values leaves the shape dynamic.
pub fn infer_mean_type(args: &[Type], context: &ResolveContext) -> Type {
    infer_numeric_reduction_type(args, context, mean_literal_control)
}

fn mean_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match MeanControl::from_literal(literal)? {
        MeanControl::All => LiteralReductionControl::All,
        MeanControl::Missing(_) => LiteralReductionControl::Missing,
        MeanControl::Weights => LiteralReductionControl::RequiresRuntimeValue,
        MeanControl::Output(class) => LiteralReductionControl::Output {
            native: class == MeanOutputClass::Native,
        },
    })
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

    #[test]
    fn repeated_mean_vecdim_literals_are_idempotent() {
        let inferred = infer_mean_type(
            &[
                input(),
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(2)]),
                },
            ],
            &ResolveContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(2.0)]),
            ]),
        );
        assert_eq!(
            inferred,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }
}
