//! Error, original context and physical admission for optimization model arrays.
use super::ScalarCooperation;
use crate::{context::RuntimeContext, BuiltinResult, RuntimeError};
use runmat_value::{Utf8Text, Value};
#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Policy {
    pub identity: runmat_builtins::BuiltinCatalogIdentity,
    pub invalid: &'static runmat_builtins::BuiltinErrorDescriptor,
}
impl Policy {
    pub fn roots<'a>(
        self,
        values: impl Iterator<Item = &'a Value> + Clone,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<crate::execution::RootedValueLease> {
        match runtime {
            Some(runtime) => {
                crate::execution::RootedValueLease::for_values_iterator_in_context(values, runtime)
            }
            None => {
                let scope = crate::context::PreparedUnselectedScope::prepare(None)?;
                let _guard = scope.enter();
                crate::execution::RootedValueLease::for_values(values)
                    .map_err(crate::execution::ExecutionServiceError::into_runtime_error)
            }
        }
    }

    pub fn reserve(self, source: std::collections::TryReserveError) -> RuntimeError {
        crate::build_runtime_error(format!(
            "{}: model metadata allocation failed",
            self.identity.name
        ))
        .with_identifier("RunMat:OptimModel:Allocation")
        .with_builtin(self.identity.name)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
    }
    pub fn text(self, value: &Value, runtime: Option<&RuntimeContext>) -> BuiltinResult<Utf8Text> {
        if !matches!(
            value,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            return Err(self.invalid("expected scalar text"));
        }
        if let Value::StringArray(array) = value {
            array
                .validate_layout()
                .map_err(|source| self.invalid(source))?;
        }
        crate::call::argument_text::convert(value, runtime)
    }

    pub fn invalid(self, message: impl std::fmt::Display) -> RuntimeError {
        let mut error = crate::build_runtime_error(format!("{}: {message}", self.identity.name))
            .with_builtin(self.identity.name)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
        if let Some(identifier) = self.invalid.identifier {
            error = error.with_identifier(identifier)
        }
        error.build()
    }
    pub fn gc(self, message: &'static str, source: runmat_gc::GcError) -> RuntimeError {
        crate::build_runtime_error(format!("{}: {message}", self.identity.name))
            .with_builtin(self.identity.name)
            .with_identifier("RunMat:OptimModel:GcFailure")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source)
            .build()
    }
    pub fn storage(self, source: runmat_value::ExactHostNumericAllocationError) -> RuntimeError {
        crate::build_runtime_error(format!(
            "{}: model storage admission failed",
            self.identity.name
        ))
        .with_builtin(self.identity.name)
        .with_identifier("RunMat:OptimModel:Allocation")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
    }
    pub async fn cooperation(
        self,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<ScalarCooperation> {
        let mut cooperation = ScalarCooperation::for_builtin(self.identity, runtime);
        cooperation.completed(0).await?;
        crate::context::host_allocation::require_unbounded_workspace(cooperation.runtime(),self.identity.name,"optimization graph/header/shape/index and evaluation backing has no complete finite physical certificate")?;
        Ok(cooperation)
    }
}
