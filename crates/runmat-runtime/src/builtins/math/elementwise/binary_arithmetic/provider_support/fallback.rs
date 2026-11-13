//! Host evaluation is a domain decision, not a consequence of a missing hook.
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{BuiltinCatalogEntry, BuiltinIntegerClass};
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

mod integer;
use integer::{certify_integer_publication, integer_class_from_catalog, integer_materialization};

/// A pre-effect decline may leave automatically placed operands on the host.
/// An explicit operand needs a certified canonical materialization form and an
/// exact resident result; an unsupported provider hook alone proves neither.
pub(in crate::builtins::math::elementwise::binary_arithmetic) struct HostFallbackAdmission {
    borrowed: Vec<GpuTensorHandle>,
    restore_source: Option<GpuTensorHandle>,
    expected_integer: Option<(BuiltinIntegerClass, Vec<usize>)>,
    builtin: &'static str,
}

impl HostFallbackAdmission {
    pub(in crate::builtins::math::elementwise::binary_arithmetic) fn new(
        runtime: Option<&crate::context::RuntimeContext>,
        entry: &'static BuiltinCatalogEntry,
        left: &Value,
        right: &Value,
    ) -> BuiltinResult<Self> {
        // A provider decline is not authority to allocate a host result.
        // Selected finite host execution consumes its prepared output before
        // reaching this legacy continuation. Otherwise refuse before gather,
        // source copying or ordinary arithmetic can allocate unowned storage.
        if matches!(left, Value::GpuTensor(_)) || matches!(right, Value::GpuTensor(_)) {
            if let Some(runtime) = runtime {
                super::super::preparation::managed::require_unbounded_host_output(runtime)?;
            }
        }
        let builtin = entry.identity.name;
        let resident = [left, right]
            .into_iter()
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle),
                _ => None,
            })
            .collect::<Vec<_>>();
        let borrowed = resident
            .iter()
            .map(|handle| (*handle).clone())
            .collect::<Vec<_>>();
        let explicit = resident
            .iter()
            .any(|handle| runmat_accelerate_api::handle_is_explicit(handle));
        let certified_integer = integer_materialization(entry, left, right).map_err(|detail| {
            build_runtime_error(detail)
                .with_builtin(builtin)
                .with_gpu_gather_retry(GpuGatherRetry::Never)
                .build()
        })?;
        if explicit && certified_integer.is_none() {
            return Err(explicit_provider_required(builtin));
        }
        if explicit {
            for handle in &resident {
                handle.allocation_lease().validate_live().map_err(|error| {
                    build_runtime_error(format!("{builtin}: resident source lease is unavailable"))
                        .with_builtin(builtin)
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .with_source(error)
                        .build()
                })?;
            }
        }
        let restore_source = if explicit {
            gpu_helpers::select_resident_output_source(borrowed.iter().cloned(), builtin)?
        } else {
            None
        };
        if let (Some(source), Some((class, shape))) = (&restore_source, &certified_integer) {
            certify_integer_publication(source, *class, shape, builtin)?;
        }
        Ok(Self {
            borrowed,
            restore_source,
            expected_integer: if explicit { certified_integer } else { None },
            builtin,
        })
    }

    pub(in crate::builtins::math::elementwise::binary_arithmetic) fn finish(
        self,
        host: Value,
    ) -> BuiltinResult<Value> {
        let Some(source) = &self.restore_source else {
            return Ok(host);
        };
        let host = match host {
            value @ Value::Int(_) => Value::Tensor(
                tensor::value_into_tensor_for(self.builtin, value).map_err(|error| {
                    build_runtime_error(error)
                        .with_builtin(self.builtin)
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build()
                })?,
            ),
            other => other,
        };
        if let Some((class, shape)) = &self.expected_integer {
            let Value::Tensor(result) = &host else {
                return Err(explicit_provider_required(self.builtin));
            };
            if result.shape() != shape
                || result
                    .integer_storage()
                    .map(|storage| storage.integer_class())
                    != Some(integer_class_from_catalog(*class))
            {
                return Err(explicit_provider_required(self.builtin));
            }
        }
        let protected = self.borrowed.iter().collect::<Vec<_>>();
        let output = gpu_helpers::restore_class_preserving_value_with_operands(
            source,
            host,
            self.builtin,
            gpu_helpers::RestoreOperandContext::new(&protected, &protected),
        )?;
        if !matches!(output, Value::GpuTensor(_)) {
            return Err(explicit_provider_required(self.builtin));
        }
        Ok(output)
    }
}

fn explicit_provider_required(builtin: &'static str) -> crate::RuntimeError {
    build_runtime_error(format!(
        "{builtin}: explicit GPU input requires an admitted provider route"
    ))
    .with_builtin(builtin)
    .with_identifier("RunMat:gpu:ExplicitProviderRequired")
    .with_gpu_gather_retry(GpuGatherRetry::Never)
    .build()
}

#[cfg(test)]
#[path = "fallback_tests.rs"]
mod tests;
