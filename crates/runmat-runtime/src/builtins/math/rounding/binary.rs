use runmat_value::{Tensor, Value};

use crate::builtins::common::{binary::BinaryInputPlan, tensor};
use crate::BuiltinResult;

pub(super) struct DurationBinaryPlan {
    pub(super) left: Tensor,
    pub(super) right: Tensor,
    format: String,
}

pub(super) fn plan_duration(
    left: Value,
    right: Value,
    builtin: &str,
) -> Result<BinaryInputPlan<DurationBinaryPlan>, String> {
    let left_duration = crate::builtins::duration::is_duration_object(&left);
    let right_duration = crate::builtins::duration::is_duration_object(&right);
    if !left_duration && !right_duration {
        return Ok(BinaryInputPlan::Values(Box::new((left, right))));
    }
    let format = if left_duration {
        crate::builtins::duration::duration_format_from_value(&left)
    } else {
        crate::builtins::duration::duration_format_from_value(&right)
    };
    let left = if left_duration {
        crate::builtins::duration::duration_tensor_from_duration_value(&left)
            .map_err(|error| error.to_string())?
    } else {
        tensor::value_into_tensor_for(builtin, left).map_err(|error| error.to_string())?
    };
    let right = if right_duration {
        crate::builtins::duration::duration_tensor_from_duration_value(&right)
            .map_err(|error| error.to_string())?
    } else {
        tensor::value_into_tensor_for(builtin, right).map_err(|error| error.to_string())?
    };
    Ok(BinaryInputPlan::Structured(DurationBinaryPlan {
        left,
        right,
        format,
    }))
}

pub(super) fn finish_duration(
    plan: DurationBinaryPlan,
    value: Value,
    builtin: &str,
) -> BuiltinResult<Value> {
    let days = tensor::value_into_tensor_for(builtin, value)?;
    crate::builtins::duration::duration_object_from_days_tensor(days, plan.format)
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn duration_plan_accepts_numeric_day_counts_in_either_operand_position() {
        let duration = crate::builtins::duration::duration_object_from_days_tensor(
            Tensor::new(vec![1.25, 2.5], vec![1, 2]).unwrap(),
            "hh:mm:ss",
        )
        .expect("duration");
        for (left, right) in [
            (duration.clone(), Value::Num(1.0)),
            (Value::Num(1.0), duration.clone()),
        ] {
            let BinaryInputPlan::Structured(plan) =
                plan_duration(left, right, "test remainder").expect("duration plan")
            else {
                panic!("expected duration plan")
            };
            assert_eq!(plan.format, "hh:mm:ss");
            let output = finish_duration(
                plan,
                Value::Tensor(Tensor::new(vec![0.25, 0.5], vec![1, 2]).unwrap()),
                "test remainder",
            )
            .expect("duration output");
            assert!(crate::builtins::duration::is_duration_object(&output));
        }
    }

    #[test]
    fn binary_provider_outputs_require_exact_shape_type_owner_and_non_aliasing() {
        test_support::with_test_provider(|provider| {
            let left = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[-7, 7]),
                    shape: &[2, 1],
                })
                .expect("upload left");
            let right = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[4, -4]),
                    shape: &[2, 1],
                })
                .expect("upload right");
            let valid = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[-3, 3]),
                    shape: &[2, 1],
                })
                .expect("upload valid output");
            let contract = crate::builtins::common::gpu_helpers::BinaryGpuOutputContract {
                shape: left.shape.clone(),
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
                precision: None,
                integer: runmat_accelerate_api::handle_integer_type(&left),
                logical: false,
                alias: crate::builtins::common::gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
            };
            let Value::GpuTensor(validated) =
                crate::builtins::common::binary::validate_resident_output(
                    provider, &left, &right, valid, &contract,
                )
                .expect("valid provider output")
            else {
                panic!("validated output must remain resident")
            };
            assert_eq!(validated.shape, vec![2, 1]);

            let mut malformed = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[-3, 3]),
                    shape: &[2, 1],
                })
                .expect("upload malformed output");
            malformed.shape = vec![1, 2];
            let rejected = malformed.clone();
            assert_eq!(
                crate::builtins::common::binary::validate_resident_output(
                    provider, &left, &right, malformed, &contract,
                ),
                Err("provider returned a malformed binary output".to_string())
            );
            assert!(block_on(provider.download_integer(&rejected)).is_err());

            assert_eq!(
                crate::builtins::common::binary::validate_resident_output(
                    provider,
                    &left,
                    &right,
                    left.clone(),
                    &contract,
                ),
                Err("provider returned a malformed binary output".to_string())
            );
            assert!(block_on(provider.download_integer(&left)).is_ok());

            for handle in [&left, &right, &validated] {
                provider.free(handle).expect("free binary test handle");
            }
        });
    }
}
