use super::super::{builtin_error, power_host, BUILTIN_NAME};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, PowerResultDomain};
use runmat_value::Value;

pub(in super::super) async fn power_gpu_pair(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(&lhs).is_none()
        && runmat_accelerate_api::handle_integer_type(&rhs).is_none()
    {
        if let Some(provider) = common_owner(&lhs, &rhs) {
            if let Some(output) = try_real_pair(provider, &lhs, &rhs).await? {
                return Ok(output);
            }
        }
    }
    gather_pair_and_evaluate(lhs, rhs).await
}

pub(super) async fn try_real_pair(
    provider: &'static dyn AccelProvider,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    if !compatible_real_inputs(lhs, rhs) {
        return Ok(None);
    }
    let shape = crate::builtins::common::broadcast::broadcast_shapes(
        BUILTIN_NAME,
        lhs.shape(),
        rhs.shape(),
    )
    .map_err(builtin_error)?;
    if !super::admission::may_execute(provider, lhs, rhs)
        .map_err(|error| provider_error("execution admission", error))?
    {
        return Ok(None);
    }
    let domain = {
        let _left = gpu_helpers::ResidencyRestoreGuard::new(lhs);
        let _right = gpu_helpers::ResidencyRestoreGuard::new(rhs);
        match provider.classify_real_power(lhs, rhs).await {
            Ok(domain) => domain,
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
            Err(error) => return Err(provider_error("domain classification", error)),
        }
    };
    let result = {
        let _left = gpu_helpers::ResidencyRestoreGuard::new(lhs);
        let _right = gpu_helpers::ResidencyRestoreGuard::new(rhs);
        provider.real_power_in_domain(lhs, rhs, domain).await
    };
    match result {
        Ok(mut output) if valid_output(&output, lhs, rhs, provider, &shape, domain) => {
            gpu_helpers::propagate_output_provenance(&mut output, [lhs, rhs])?;
            let value = if domain == PowerResultDomain::Complex {
                gpu_helpers::complex_gpu_value(output)
            } else {
                gpu_helpers::resident_gpu_value(output)
            }?;
            Ok(Some(value))
        }
        Ok(output) => Err(gpu_helpers::reject_gpu_output(
            output,
            builtin_error("power: provider returned malformed domain-preserving output"),
        )),
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
        Err(error) => Err(provider_error("provider execution", error)),
    }
}

fn valid_output(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
    domain: PowerResultDomain,
) -> bool {
    use gpu_helpers::{BinaryGpuOutputContract, GpuOutputAliasPolicy};
    let storage = match domain {
        PowerResultDomain::Real => GpuTensorStorage::Real,
        PowerResultDomain::Complex => GpuTensorStorage::ComplexInterleaved,
    };
    output.descriptor().storage == Some(storage)
        && gpu_helpers::binary_gpu_output_matches(
            output,
            lhs,
            rhs,
            provider,
            &BinaryGpuOutputContract {
                shape: shape.to_vec(),
                storage,
                precision: runmat_accelerate_api::handle_precision(lhs),
                integer: None,
                logical: false,
                alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
            },
        )
}

fn common_owner(
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> Option<&'static dyn AccelProvider> {
    let left = gpu_helpers::exact_provider_for_handle(lhs)?;
    let right = gpu_helpers::exact_provider_for_handle(rhs)?;
    (std::ptr::eq(left, right) && lhs.device_id() == rhs.device_id()).then_some(left)
}

fn compatible_real_inputs(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    [lhs, rhs].into_iter().all(|handle| {
        handle.descriptor().storage == Some(GpuTensorStorage::Real)
            && !runmat_accelerate_api::handle_is_logical(handle)
            && runmat_accelerate_api::handle_integer_type(handle).is_none()
            && gpu_helpers::expected_handle_numeric_element_type(handle).is_ok()
    }) && runmat_accelerate_api::handle_precision(lhs).is_some()
        && runmat_accelerate_api::handle_precision(lhs)
            == runmat_accelerate_api::handle_precision(rhs)
}

fn provider_error(stage: &str, source: anyhow::Error) -> crate::RuntimeError {
    let mut error = builtin_error(format!("power: {stage} failed: {source}"));
    error.source = Some(source.into_boxed_dyn_error());
    error
}

async fn gather_pair_and_evaluate(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    let left = gpu_helpers::gather_tensor_async(&lhs)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::gather_tensor_async(&rhs)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    power_host(Value::Tensor(left), Value::Tensor(right))
}

#[cfg(test)]
mod owning_tests {
    use super::*;
    use runmat_accelerate_api::{test_support::TestAllocationOwner, NumericElementType};

    #[test]
    fn real_pair_admission_reads_immutable_class_and_storage() {
        let owner = TestAllocationOwner::new();
        let real = owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let complex = owner
            .allocate_zeroed(
                vec![2, 1],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let logical = owner.allocate_logical(vec![0, 1], vec![2, 1]).unwrap();
        assert!(compatible_real_inputs(&real.handle, &real.handle));
        assert!(!compatible_real_inputs(&real.handle, &complex.handle));
        assert!(!compatible_real_inputs(&logical.handle, &real.handle));
        assert_eq!(real.probe.retirements(), 0);
        assert_eq!(complex.probe.retirements(), 0);
        assert_eq!(logical.probe.retirements(), 0);
    }

    #[test]
    fn provider_failure_keeps_its_typed_source() {
        let error = provider_error(
            "domain classification",
            anyhow::Error::new(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "classifier transport closed",
            )),
        );
        let source = error.source.as_ref().unwrap();
        assert_eq!(
            source.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::BrokenPipe,
        );
    }
}
