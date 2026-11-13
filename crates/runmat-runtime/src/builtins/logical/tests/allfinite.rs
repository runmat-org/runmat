//! Scalar finite-value reduction.

use super::classification::valid_provider_truth_handle;
use super::metadata::validate_resident_numeric_metadata;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{
    BuiltinCatalogEntry, BuiltinErrorDescriptor, ALLFINITE_CATALOG_ENTRY, ALLFINITE_ERROR_INTERNAL,
    ALLFINITE_ERROR_INVALID_INPUT, ALLFINITE_ERROR_TOO_MANY_OUTPUTS,
    ALLFINITE_STRING_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[path = "allfinite/host.rs"]
mod host;
use host::{complex_tensor_all_finite, sparse_all_finite, tensor_all_finite};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::allfinite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "allfinite",
    op_kind: GpuOpKind::Custom("allfinite"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Unary {
            name: "logical_isfinite",
        },
        ProviderHook::Reduction { name: "reduce_all" },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may classify and reduce floating input on-device. The scalar logical result is returned on the host; typed unsupported hooks use one exact-owner input transfer.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::allfinite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "allfinite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Full-input scalar reduction that forms a fusion boundary.",
};

const BOUNDARY: AllFiniteBoundary = AllFiniteBoundary {
    entry: &ALLFINITE_CATALOG_ENTRY,
    invalid: &ALLFINITE_ERROR_INVALID_INPUT,
    internal: &ALLFINITE_ERROR_INTERNAL,
    too_many_outputs: &ALLFINITE_ERROR_TOO_MANY_OUTPUTS,
};

#[runtime_builtin(
    name = "allfinite",
    binding_variant = "default",
    builtin_path = "crate::builtins::logical::tests::allfinite"
)]
async fn allfinite_builtin(value: Value) -> BuiltinResult<Value> {
    BOUNDARY.execute(value).await
}

struct AllFiniteBoundary {
    entry: &'static BuiltinCatalogEntry,
    invalid: &'static BuiltinErrorDescriptor,
    internal: &'static BuiltinErrorDescriptor,
    too_many_outputs: &'static BuiltinErrorDescriptor,
}

impl AllFiniteBoundary {
    async fn execute(&self, value: Value) -> BuiltinResult<Value> {
        self.reject_excess_outputs()?;
        if matches!(value, Value::String(_) | Value::StringArray(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ALLFINITE_STRING_INPUT_EXTENSION,
                self.name(),
            )?;
        }
        match value {
            Value::GpuTensor(handle) => self.execute_resident(handle).await,
            host => self.execute_host(host),
        }
    }

    async fn execute_resident(&self, handle: GpuTensorHandle) -> BuiltinResult<Value> {
        validate_resident_numeric_metadata(&handle).map_err(|detail| self.internal(detail))?;
        if runmat_accelerate_api::handle_integer_type(&handle).is_some()
            || runmat_accelerate_api::handle_is_logical(&handle)
        {
            return Ok(Value::Bool(true));
        }
        let provider = gpu_helpers::exact_provider_for_handle(&handle)
            .ok_or_else(|| self.internal("no acceleration provider owns the input handle"))?;
        let _residency = runmat_accelerate_api::ResidencyRestoreGuard::new(&handle);
        let mask_result = provider.logical_isfinite(&handle);
        let mask = match mask_result {
            Ok(mask)
                if !gpu_helpers::same_gpu_handle(&handle, &mask)
                    && valid_provider_truth_handle(&mask, provider, handle.shape()) =>
            {
                mask
            }
            Ok(mask) => {
                return Err(gpu_helpers::reject_gpu_output(
                    mask,
                    self.internal("provider returned an invalid finite-value mask"),
                ));
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
                return self.gather_and_reduce(provider, &handle).await;
            }
            Err(error) => {
                return Err(self.owning_error(error.into()));
            }
        };

        let reduced_result = provider.reduce_all(&mask, false).await;
        let reduced = match reduced_result {
            Ok(reduced)
                if !gpu_helpers::same_gpu_handle(&handle, &reduced)
                    && !gpu_helpers::same_gpu_handle(&mask, &reduced)
                    && valid_provider_truth_handle(&reduced, provider, &[1, 1]) =>
            {
                reduced
            }
            Ok(reduced) => {
                return Err(gpu_helpers::reject_gpu_output(
                    reduced,
                    self.internal("provider returned an invalid logical reduction"),
                ));
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
                return self.gather_and_reduce(provider, &handle).await;
            }
            Err(error) => {
                return Err(self.owning_error(error.into()));
            }
        };

        let downloaded = gpu_helpers::download_truth_values_async(provider, &reduced).await;

        let downloaded = downloaded.map_err(|error| {
            self.owning_error(runmat_accelerate_api::ProviderExecutionError::from_source(
                error,
            ))
        })?;
        if downloaded.shape != [1, 1] || downloaded.data.len() != 1 {
            return Err(self.internal("provider reduction payload is not one logical scalar"));
        }
        Ok(Value::Bool(downloaded.data[0] != 0))
    }

    async fn gather_and_reduce(
        &self,
        provider: &dyn AccelProvider,
        handle: &GpuTensorHandle,
    ) -> BuiltinResult<Value> {
        let host = gpu_helpers::download_value_preserving_residency_async(provider, handle)
            .await
            .map_err(|error| {
                self.owning_error(runmat_accelerate_api::ProviderExecutionError::from_source(
                    error,
                ))
            })?;
        self.execute_host(host)
    }

    fn execute_host(&self, value: Value) -> BuiltinResult<Value> {
        let finite = match value {
            Value::Num(value) => value.is_finite(),
            Value::Int(_) | Value::Bool(_) => true,
            Value::Complex(real, imaginary) => real.is_finite() && imaginary.is_finite(),
            Value::Tensor(value) => tensor_all_finite(&value),
            Value::ComplexTensor(value) => complex_tensor_all_finite(&value),
            Value::SparseTensor(value) => sparse_all_finite(&value),
            Value::LogicalArray(_) | Value::CharArray(_) => true,
            Value::String(_) => false,
            Value::StringArray(value) => value.data().is_empty(),
            _ => return Err(self.error(self.invalid, "unsupported input representation")),
        };
        Ok(Value::Bool(finite))
    }

    fn reject_excess_outputs(&self) -> BuiltinResult<()> {
        if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
            return Err(self.error(self.too_many_outputs, "only one output is defined"));
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        self.entry.identity.name
    }

    fn owning_error(&self, error: runmat_accelerate_api::ProviderExecutionError) -> RuntimeError {
        let mut wrapped = self.internal(error.to_string());
        wrapped.source = Some(Box::new(error));
        wrapped
    }

    fn internal(&self, detail: impl std::fmt::Display) -> RuntimeError {
        self.error(self.internal, detail)
    }

    fn error(
        &self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl std::fmt::Display,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
            .with_builtin(self.name());
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

#[cfg(test)]
#[path = "allfinite/tests.rs"]
mod tests;
