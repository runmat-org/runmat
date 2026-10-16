//! Per-invocation broadcast and real-power value-domain evidence.

#[cfg(test)]
use super::super::domain::PowerResultDomain;
use super::super::{builtin_error, power_error_with_detail};
use super::*;
use crate::builtins::common::broadcast::BroadcastPlan;
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_builtins::POWER_ERROR_SIZE_MISMATCH;

pub(super) fn classify(operands: &[Value]) -> BuiltinResult<NumericalDomainEvidence> {
    let [left, right] = operands else {
        return Err(builtin_error(
            "power: admitted real pair is not a tensor pair",
        ));
    };
    classify_operands(left, right)
}

pub(super) fn classify_operands(
    left: &Value,
    right: &Value,
) -> BuiltinResult<NumericalDomainEvidence> {
    let left_shape = crate::builtins::math::host_route::real_floating_shape(left)
        .ok_or_else(|| builtin_error("power: nonnative admitted base"))?;
    let right_shape = crate::builtins::math::host_route::real_floating_shape(right)
        .ok_or_else(|| builtin_error("power: nonnative admitted exponent"))?;
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
    let domain = super::super::domain::classify_pair(
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
