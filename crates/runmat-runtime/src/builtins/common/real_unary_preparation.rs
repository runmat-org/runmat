//! Preparation of an admitted real floating operand with request-local domain evidence.
use crate::context::preparation::{
    execute_current_request, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::BuiltinResult;
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinPreparationFormId};
use runmat_types::{RealUnaryClassification, RealUnaryPredicate};
use runmat_value::{NumericDType, Tensor, Value};
use std::future::Future;

pub(crate) async fn execute<F, Fut>(
    entry: &'static BuiltinCatalogEntry,
    predicate: RealUnaryPredicate,
    value: Value,
    body: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    if !matches!(&value, Value::Tensor(tensor) if matches!(tensor.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
    {
        return body(value).await;
    }
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count: 1,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
    )?
    .with_numerical_domain(|operands| {
        let [Value::Tensor(tensor)] = operands else {
            unreachable!("admitted floating operand")
        };
        Ok(NumericalDomainEvidence::RealUnary {
            predicate,
            classification: classify(tensor, predicate),
        })
    })?;
    if !matches!(
        request.numerical_domain(),
        NumericalDomainEvidence::RealUnary {
            classification: RealUnaryClassification::Inside,
            ..
        }
    ) {
        return body(request.into_operands().remove(0)).await;
    }
    execute_current_request(request, |mut values| body(values.remove(0))).await
}

fn classify(tensor: &Tensor, predicate: RealUnaryPredicate) -> RealUnaryClassification {
    let mut result = RealUnaryClassification::Inside;
    for index in 0..tensor.len() {
        let value = crate::builtins::common::tensor::tensor_value_f64(tensor, index);
        if value.is_nan() {
            result = RealUnaryClassification::Indeterminate;
            continue;
        }
        let inside = match predicate {
            RealUnaryPredicate::Nonnegative => value >= 0.0,
            RealUnaryPredicate::AtLeastMinusOne => value >= -1.0,
            RealUnaryPredicate::AtLeastOne => value >= 1.0,
            RealUnaryPredicate::ClosedUnitInterval => (-1.0..=1.0).contains(&value),
        };
        if !inside {
            return RealUnaryClassification::Outside;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundaries_nan_and_single_values_are_classified_without_rounding() {
        for (values, predicate, expected) in [
            (
                vec![-0.0, f64::INFINITY],
                RealUnaryPredicate::Nonnegative,
                RealUnaryClassification::Inside,
            ),
            (
                vec![-1.0, 0.0],
                RealUnaryPredicate::AtLeastMinusOne,
                RealUnaryClassification::Inside,
            ),
            (
                vec![0.0, f64::NAN],
                RealUnaryPredicate::Nonnegative,
                RealUnaryClassification::Indeterminate,
            ),
            (
                vec![f64::NAN, -1e-300],
                RealUnaryPredicate::Nonnegative,
                RealUnaryClassification::Outside,
            ),
        ] {
            let tensor = Tensor::new(values, vec![1, 2]).unwrap();
            assert_eq!(classify(&tensor, predicate), expected);
        }
        let tensor = Tensor::from_f32(vec![-f32::MIN_POSITIVE], vec![1, 1]).unwrap();
        assert_eq!(
            classify(&tensor, RealUnaryPredicate::Nonnegative),
            RealUnaryClassification::Outside
        );
    }
}
