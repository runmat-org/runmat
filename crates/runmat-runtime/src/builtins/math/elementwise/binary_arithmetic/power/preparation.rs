//! Value-domain evidence belongs to this admitted pair, not a cached name.
#[cfg(test)]
use super::domain::PowerResultDomain;
use super::power_error_with_detail;
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::context::preparation::{
    execute_current_request, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::BuiltinResult;
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinPreparationFormId, POWER_CATALOG_ENTRY,
    POWER_ERROR_SIZE_MISMATCH,
};
use runmat_value::{NumericDType, Value};
use std::future::Future;

pub(super) async fn execute<F, Fut>(
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<Value>>,
{
    let admitted = input_count == 2 && [&left, &right].into_iter().all(|value| {
        matches!(value, Value::Tensor(t) if matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
    });
    if !admitted {
        return body(left, right).await;
    }
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: POWER_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            signature: 0,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![left, right],
    )?
    .with_numerical_domain(classify)?;
    execute_current_request(request, |mut values| {
        let right = values.remove(1);
        body(values.remove(0), right)
    })
    .await
}

fn classify(operands: &[Value]) -> BuiltinResult<NumericalDomainEvidence> {
    let [Value::Tensor(left), Value::Tensor(right)] = operands else {
        return Err(super::builtin_error(
            "power: admitted real pair is not a tensor pair",
        ));
    };
    let plan = BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|detail| power_error_with_detail(&POWER_ERROR_SIZE_MISMATCH, detail))?;
    let domain = super::domain::classify_pair(
        &plan,
        |index| crate::builtins::common::tensor::tensor_value_f64(left, index),
        |index| crate::builtins::common::tensor::tensor_value_f64(right, index),
    );
    Ok(NumericalDomainEvidence::RealPower(domain))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::Tensor;

    #[test]
    fn evidence_belongs_to_each_exact_pair_and_broadcast_layout() {
        let left = Value::Tensor(Tensor::new(vec![-4.0, 4.0], vec![1, 2]).unwrap());
        for (shape, expected) in [
            (vec![1, 2], PowerResultDomain::Real),
            (vec![2, 1], PowerResultDomain::Complex),
        ] {
            let right = Value::Tensor(Tensor::new(vec![2.0, 0.5], shape).unwrap());
            assert_eq!(
                classify(&[left.clone(), right]).unwrap(),
                NumericalDomainEvidence::RealPower(expected)
            );
        }
        let tiny = Value::Tensor(Tensor::from_f32(vec![-1e-30], vec![1, 1]).unwrap());
        let fraction = Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap());
        assert_eq!(
            classify(&[tiny, fraction]).unwrap(),
            NumericalDomainEvidence::RealPower(PowerResultDomain::Complex)
        );
    }
}
