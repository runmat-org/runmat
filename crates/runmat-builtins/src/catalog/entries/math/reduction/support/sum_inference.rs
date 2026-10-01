use super::numeric_inference::{
    infer_numeric_reduction_type, LiteralReductionControl, LiteralReductionOutput,
    ScalarResultClass,
};
use super::{SumControl, SumOutputClass};
use crate::{ResolveContext, Type};
use runmat_types::LiteralValue;

/// A conservative type projection for the admitted literal `sum` grammar.
/// Runtime-dependent controls remain dynamic until value-level admission.
pub fn infer_sum_type(args: &[Type], context: &ResolveContext) -> Type {
    infer_numeric_reduction_type(
        args,
        context,
        sum_literal_control,
        ScalarResultClass::Promoted,
        None,
        1,
        super::numeric_inference::EmptyAxesPolicy::Default,
    )
}

fn sum_literal_control(literal: &LiteralValue) -> Option<LiteralReductionControl> {
    Some(match SumControl::from_literal(literal)? {
        SumControl::All => LiteralReductionControl::All,
        SumControl::Missing(_) => LiteralReductionControl::Missing,
        SumControl::Output(class) => LiteralReductionControl::Output(match class {
            SumOutputClass::Double => LiteralReductionOutput::Double,
            SumOutputClass::Native => LiteralReductionOutput::Native,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::infer_sum_type;
    use crate::{ResolveContext, Type};
    use runmat_types::LiteralValue;

    fn input() -> Type {
        Type::Tensor {
            shape: Some(vec![Some(3), Some(4)]),
        }
    }

    #[test]
    fn ordered_sum_literals_prove_reduction_shapes() {
        let along_second = infer_sum_type(
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

        let repeated_dimension = infer_sum_type(
            &[input(), Type::Tensor { shape: None }],
            &ResolveContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Vector(vec![LiteralValue::Number(2.0), LiteralValue::Number(2.0)]),
            ]),
        );
        assert_eq!(repeated_dimension, along_second);
    }

    #[test]
    fn invalid_or_ambiguous_sum_literals_do_not_prove_shapes() {
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
                LiteralValue::String("all".into()),
                LiteralValue::String("all".into()),
            ],
            vec![
                LiteralValue::String("native".into()),
                LiteralValue::String("double".into()),
            ],
            vec![LiteralValue::Unknown, LiteralValue::String("all".into())],
        ];
        for options in cases {
            let mut literals = vec![LiteralValue::Unknown];
            literals.extend(options);
            let arguments = vec![input(), Type::Unknown, Type::Unknown];
            assert_eq!(
                infer_sum_type(&arguments, &ResolveContext::new(literals)),
                Type::Tensor { shape: None }
            );
        }
    }

    #[test]
    fn native_sum_integer_scalar_retains_class_fact() {
        assert_eq!(
            infer_sum_type(
                &[Type::Int, Type::String],
                &ResolveContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::String("native".into()),
                ]),
            ),
            Type::Int
        );
        assert_eq!(
            infer_sum_type(&[Type::Int], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }
}
