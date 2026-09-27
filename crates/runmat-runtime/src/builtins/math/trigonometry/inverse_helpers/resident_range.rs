//! Range evidence for real inverse-trigonometric provider kernels.
use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(crate) enum RealDomain {
    ClosedUnitInterval,
    AtLeastOne,
}

fn error(builtin: &str, detail: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("{builtin}: {detail}"))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:Internal"))
        .build()
}

struct Reductions<'a> {
    input: &'a GpuTensorHandle,
    owned: Vec<GpuTensorHandle>,
}

impl Reductions<'_> {
    fn adopt(
        &mut self,
        provider: &'static dyn AccelProvider,
        output: GpuTensorHandle,
        builtin: &str,
    ) -> BuiltinResult<()> {
        let protected: Vec<_> = std::iter::once(self.input).chain(&self.owned).collect();
        let contract = gpu_helpers::BinaryGpuOutputContract {
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(self.input),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        };
        let valid = !protected
            .iter()
            .any(|handle| handle.shares_allocation_with(&output))
            && gpu_helpers::binary_gpu_output_matches(
                &output, self.input, self.input, provider, &contract,
            )
            && gpu_helpers::expected_handle_numeric_element_type(&output).is_ok_and(|element| {
                gpu_helpers::numeric_descriptor_matches(&output, element, GpuTensorStorage::Real)
            });
        if !valid {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                error(
                    builtin,
                    "provider returned malformed or aliased range reduction",
                ),
            ));
        }
        self.owned.push(output);
        Ok(())
    }
}

/// None means no conclusive real-domain proof, not successful provider execution.
/// Empty inputs, NaN reductions and explicitly unsupported hooks use the existing
/// host computation route. Other failures propagate; all accepted temporaries
/// remain owned across every suspension point and are reclaimed on abandonment.
pub(crate) async fn requires_complex(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    domain: RealDomain,
    builtin: &str,
) -> BuiltinResult<Option<bool>> {
    if input.shape().contains(&0) {
        return Ok(None);
    }
    let owner = gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| error(builtin, "GPU input has no owning provider"))?;
    if !std::ptr::eq(owner, provider)
        || !gpu_helpers::expected_handle_numeric_element_type(input).is_ok_and(|element| {
            gpu_helpers::numeric_descriptor_matches(input, element, GpuTensorStorage::Real)
        })
    {
        return Err(error(
            builtin,
            "range reduction requires an owned real numeric input",
        ));
    }
    let mut outputs = Reductions {
        input,
        owned: Vec::new(),
    };
    match provider.reduce_min(input).await {
        Ok(output) => outputs.adopt(provider, output, builtin)?,
        Err(failure) if gpu_helpers::provider_hook_is_unsupported(&failure) => return Ok(None),
        Err(failure) => return Err(provider_error(builtin, "reduce_min", failure)),
    };
    if matches!(domain, RealDomain::ClosedUnitInterval) {
        match provider.reduce_max(input).await {
            Ok(output) => {
                outputs.adopt(provider, output, builtin)?;
            }
            Err(failure) if gpu_helpers::provider_hook_is_unsupported(&failure) => return Ok(None),
            Err(failure) => return Err(provider_error(builtin, "reduce_max", failure)),
        }
    }
    let mut outside = false;
    let mut minimum = None;
    for output in &outputs.owned {
        let host = gpu_helpers::download_native_values_async(provider, output)
            .await
            .map_err(|failure| {
                crate::builtins::common::map_control_flow_with_builtin(failure, builtin)
            })?;
        if host.data.len() != 1 {
            return Err(error(
                builtin,
                "range reduction did not download one scalar",
            ));
        }
        let value = &host.data[0];
        if value.is_nan() {
            return Ok(None);
        }
        if matches!(domain, RealDomain::ClosedUnitInterval) {
            // NaN-omitting reductions can return their identity pair for an
            // all-NaN input. An inverted range is not evidence of complex data.
            let current = match value {
                runmat_value::NumericScalar::F64(value) => Some(*value),
                runmat_value::NumericScalar::F32(value) => Some(f64::from(*value)),
                _ => None,
            };
            if let (Some(low), Some(high)) = (minimum, current) {
                if low > high {
                    return Ok(None);
                }
            }
            minimum = current;
        }
        outside |= match domain {
            RealDomain::ClosedUnitInterval => value.is_outside_closed_unit_interval(0.0),
            RealDomain::AtLeastOne => value.is_less_than_one(),
        };
    }
    Ok(Some(outside))
}

fn provider_error(builtin: &str, operation: &str, source: anyhow::Error) -> RuntimeError {
    let mut failure = error(builtin, format!("{operation} failed: {source}"));
    failure = crate::runtime_error::with_provider_execution_source(failure, source.into());
    failure
}

#[cfg(test)]
mod lease_tests {
    use super::*;
    use runmat_accelerate_api::{test_support::TestAllocationOwner, NumericElementType};

    #[test]
    fn reduction_container_owns_temporary_but_not_input_or_exported_alias() {
        let owner = TestAllocationOwner::new();
        let input = owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let temporary = owner
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let alias = temporary.handle.clone();
        let reductions = Reductions {
            input: &input.handle,
            owned: vec![temporary.handle],
        };
        drop(reductions);
        assert_eq!(input.probe.retirements(), 0);
        assert_eq!(temporary.probe.retirements(), 0);
        alias.allocation_lease().validate_live().unwrap();
        drop(alias);
        assert_eq!(temporary.probe.retirements(), 1);
        input.handle.allocation_lease().validate_live().unwrap();
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}
