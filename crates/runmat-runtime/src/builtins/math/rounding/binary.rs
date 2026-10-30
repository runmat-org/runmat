use runmat_value::{Tensor, Value};

use crate::builtins::common::{binary::BinaryInputPlan, tensor};
use crate::BuiltinResult;

pub(super) fn floating_pair_shape(
    left: &runmat_accelerate_api::GpuTensorHandle,
    right: &runmat_accelerate_api::GpuTensorHandle,
    builtin: &str,
) -> Result<Option<Vec<usize>>, String> {
    use runmat_accelerate_api::NumericElementType;
    if !matches!(
        left.descriptor().element_type,
        Some(NumericElementType::F32 | NumericElementType::F64)
    ) || left.descriptor().element_type != right.descriptor().element_type
        || runmat_accelerate_api::handle_is_logical(left)
        || runmat_accelerate_api::handle_is_logical(right)
    {
        return Ok(None);
    }
    let shape =
        crate::builtins::common::broadcast::broadcast_shapes(builtin, left.shape(), right.shape())?;
    if !shape.contains(&0)
        && shape
            .iter()
            .try_fold(1_usize, |n, d| n.checked_mul(*d))
            .is_none()
    {
        return Err("broadcast element count overflow".into());
    }
    Ok(Some(shape))
}

pub(super) fn admit_real_resident(
    value: &Value,
    builtin: &str,
    error: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        return Ok(());
    };
    let helpers =
        crate::builtins::common::gpu_helpers::expected_handle_numeric_element_type(handle);
    let valid = helpers.is_ok_and(|element| {
        crate::builtins::common::gpu_helpers::numeric_descriptor_matches(
            handle,
            element,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
    }) && (handle.shape().contains(&0)
        || handle
            .shape()
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .is_some());
    if !valid {
        return Err(super::unary::error_with_detail(
            builtin,
            error,
            "expected an owned real numeric resident input",
        ));
    }
    Ok(())
}

// Provider availability is an execution precondition, not source-language
// numeric admission. Check it only after pure operand-pair errors are resolved.
pub(super) fn require_resident_pair(
    lhs: &Value,
    rhs: &Value,
    builtin: &str,
    error: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> BuiltinResult<()> {
    crate::builtins::common::gpu_helpers::require_resident_compute_values(
        &[lhs, rhs],
        false,
        || {
            super::unary::error_with_detail(
                builtin,
                error,
                "expected an owned real numeric resident input",
            )
        },
    )
}

pub(super) async fn gather_real(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Tensor> {
    let _metadata = crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(handle);
    crate::builtins::common::gpu_helpers::gather_tensor_async(handle).await
}

pub(super) struct DurationBinaryPlan {
    pub(super) left: Tensor,
    pub(super) right: Tensor,
    format: String,
}

pub(super) fn plan_duration(
    left: Value,
    right: Value,
    builtin: &str,
) -> BuiltinResult<BinaryInputPlan<DurationBinaryPlan>> {
    let left_duration = crate::builtins::duration::is_duration_object(&left);
    let right_duration = crate::builtins::duration::is_duration_object(&right);
    if !left_duration && !right_duration {
        return Ok(BinaryInputPlan::Values(Box::new((left, right))));
    }
    let format = if left_duration {
        crate::builtins::duration::duration_format_from_value(&left)?
    } else {
        crate::builtins::duration::duration_format_from_value(&right)?
    };
    let left = if left_duration {
        crate::builtins::duration::duration_tensor_from_duration_value(&left)?
    } else {
        tensor::value_into_tensor_for(builtin, left)?
    };
    let right = if right_duration {
        crate::builtins::duration::duration_tensor_from_duration_value(&right)?
    } else {
        tensor::value_into_tensor_for(builtin, right)?
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
    crate::builtins::duration::duration_object_from_days_tensor(None, days, plan.format)
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
            None,
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
                shape: left.shape().to_vec(),
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
            assert_eq!(validated.shape(), &[2, 1]);

            let malformed = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[-3, 3]),
                    shape: &[1, 2],
                })
                .expect("upload malformed output");
            let rejected = malformed.allocation_lease().downgrade();
            let error = crate::builtins::common::binary::validate_resident_output(
                provider, &left, &right, malformed, &contract,
            )
            .unwrap_err();
            assert_eq!(
                error.message(),
                "provider returned a malformed binary output"
            );
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(rejected.acquire().is_none());

            let error = crate::builtins::common::binary::validate_resident_output(
                provider,
                &left,
                &right,
                left.clone(),
                &contract,
            )
            .unwrap_err();
            assert_eq!(
                error.message(),
                "provider returned a malformed binary output"
            );
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(block_on(provider.download_integer(&left)).is_ok());

            for handle in [left, right, validated] {
                let physical = handle.allocation_lease().downgrade();
                handle.release().expect("release binary test handle");
                assert!(physical.acquire().is_none());
            }
        });
    }
}
