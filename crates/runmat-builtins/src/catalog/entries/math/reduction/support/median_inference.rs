use super::numeric_inference::{
    infer_numeric_reduction_type, LiteralReductionControl, ScalarResultClass,
};
use super::MedianControl;
use crate::{ResolveContext, Type};
use runmat_types::LiteralValue;

/// Infer only admitted, literal `median` forms. Weighted and runtime-dependent
/// controls remain dynamic; integer scalar medians retain their exact class.
pub fn infer_median_type(args: &[Type], context: &ResolveContext) -> Type {
    infer_numeric_reduction_type(
        args,
        context,
        median_literal_control,
        ScalarResultClass::Preserved,
        None,
    )
}

fn median_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match MedianControl::from_literal(literal)? {
        MedianControl::All => LiteralReductionControl::All,
        MedianControl::Missing(_) => LiteralReductionControl::Missing,
        MedianControl::Weights => LiteralReductionControl::RequiresRuntimeValue,
    })
}

#[cfg(test)]
mod tests {
    use super::infer_median_type;
    use crate::{ResolveContext, Type};
    use runmat_types::LiteralValue;

    fn input() -> Type {
        Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        }
    }

    #[test]
    fn median_literals_prove_only_admitted_axis_shapes() {
        let along_second = infer_median_type(
            &[input(), Type::Num, Type::String],
            &ResolveContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Character("omitmissing".into()),
            ]),
        );
        assert_eq!(
            along_second,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );

        let all = infer_median_type(
            &[input(), Type::String],
            &ResolveContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("all".into()),
            ]),
        );
        assert_eq!(
            all,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1)])
            }
        );
    }

    #[test]
    fn median_conflicts_and_weights_leave_shape_dynamic() {
        let cases = [
            vec![
                LiteralValue::Number(1.0),
                LiteralValue::String("all".into()),
            ],
            vec![
                LiteralValue::String("all".into()),
                LiteralValue::Number(1.0),
            ],
            vec![
                LiteralValue::String("Weights".into()),
                LiteralValue::Unknown,
            ],
        ];
        for options in cases {
            let mut literals = vec![LiteralValue::Unknown];
            literals.extend(options);
            assert_eq!(
                infer_median_type(
                    &[input(), Type::Unknown, Type::Unknown],
                    &ResolveContext::new(literals),
                ),
                Type::Tensor { shape: None }
            );
        }
    }

    #[test]
    fn exact_scalar_medians_preserve_integer_and_logical_classes() {
        assert_eq!(
            infer_median_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Int
        );
        assert_eq!(
            infer_median_type(&[Type::Bool], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
        assert_eq!(
            infer_median_type(
                &[Type::Logical { shape: None }],
                &ResolveContext::new(Vec::new())
            ),
            Type::Unknown
        );
    }
}
