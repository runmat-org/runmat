//! Value-domain evidence belongs to this admitted pair, not a cached name.
#[cfg(test)]
use super::domain::PowerResultDomain;
use super::power_error_with_detail;
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::context::preparation::{
    execute_request, AdmittedBuiltinPreparation, BuiltinPreparationAdmission,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinPreparationFormId, POWER_CATALOG_ENTRY,
    POWER_ERROR_SIZE_MISMATCH,
};
use runmat_value::{NumericDType, Value};
use std::future::Future;

pub(super) async fn execute<F, Fut>(
    runtime: Option<&RuntimeContext>,
    input_count: usize,
    left: Value,
    right: Value,
    body: F,
) -> BuiltinResult<RootedValue>
where
    F: FnOnce(Value, Value) -> Fut,
    Fut: Future<Output = BuiltinResult<RootedValue>>,
{
    let _inputs = RootedValueLease::for_values([&left, &right].into_iter())
        .map_err(|cause| cause.into_runtime_error())?;
    if input_count == 2 {
        if let Some(route) = crate::builtins::math::host_route::canonical(
            runtime,
            crate::context::VmFusionHostInstruction::ElementwiseBinary,
            [&left, &right].into_iter(),
        ) {
            // The canonical power body still owns domain/class arithmetic;
            // only operand promotion is bypassed for the selected host route.
            let evidence = classify_operands(&left, &right)?;
            let output = body(left, right).await?;
            route.confirm_numerical_host_branch(
                runtime.unwrap(),
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
                evidence,
            );
            return Ok(output);
        }
    }
    let resident = [&left, &right]
        .into_iter()
        .any(|value| matches!(value, Value::GpuTensor(_)));
    let admitted = input_count == 2
        && [&left, &right].into_iter().all(|value| match value {
            Value::GpuTensor(handle) => {
                runmat_accelerate_api::handle_provenance(handle)
                    == Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                    && crate::builtins::math::host_route::real_floating_shape(value).is_some()
            }
            Value::Tensor(t) => matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64),
            Value::Num(_) => resident,
            _ => false,
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
    execute_request(runtime, request, |mut values| async move {
        let route = crate::builtins::math::host_route::canonical(
            runtime,
            crate::context::VmFusionHostInstruction::ElementwiseBinary,
            values.iter(),
        );
        let evidence = route.as_ref().map(|_| classify(&values)).transpose()?;
        let right = values.remove(1);
        let output = body(values.remove(0), right).await?;
        if let (Some(route), Some(evidence)) = (route, evidence) {
            route.confirm_numerical_host_branch(
                runtime.unwrap(),
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
                evidence,
            );
        }
        Ok(output)
    })
    .await
}

fn classify(operands: &[Value]) -> BuiltinResult<NumericalDomainEvidence> {
    let [left, right] = operands else {
        return Err(super::builtin_error(
            "power: admitted real pair is not a tensor pair",
        ));
    };
    classify_operands(left, right)
}

fn classify_operands(left: &Value, right: &Value) -> BuiltinResult<NumericalDomainEvidence> {
    let left_shape = crate::builtins::math::host_route::real_floating_shape(left)
        .ok_or_else(|| super::builtin_error("power: nonnative admitted base"))?;
    let right_shape = crate::builtins::math::host_route::real_floating_shape(right)
        .ok_or_else(|| super::builtin_error("power: nonnative admitted exponent"))?;
    let plan = BroadcastPlan::new(left_shape, right_shape)
        .map_err(|detail| power_error_with_detail(&POWER_ERROR_SIZE_MISMATCH, detail))?;
    if [left, right]
        .into_iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(NumericalDomainEvidence::Unknown);
    }
    let value_at = |value: &Value, index| match value {
        Value::Tensor(tensor) => crate::builtins::common::tensor::tensor_value_f64(tensor, index),
        Value::Num(value) => *value,
        _ => unreachable!("admitted native floating pair"),
    };
    let domain = super::domain::classify_pair(
        &plan,
        |index| value_at(left, index),
        |index| value_at(right, index),
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
