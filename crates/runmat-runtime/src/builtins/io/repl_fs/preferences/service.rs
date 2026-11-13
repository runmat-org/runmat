use std::rc::Rc;

use runmat_value::{StructValue, Value};

use crate::context::{PreferenceServiceError, RuntimeContext, RuntimePreferenceService};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) fn required(
    runtime: Option<RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<(RuntimeContext, Rc<dyn RuntimePreferenceService>)> {
    let runtime = runtime.ok_or_else(|| unavailable(builtin))?;
    let service = runtime
        .service_ports()
        .require_preferences(format!("{builtin} application preferences"))
        .map_err(|source| source.into_runtime_error())?
        .clone();
    Ok((runtime, service))
}

fn unavailable(builtin: &str) -> RuntimeError {
    build_runtime_error(format!(
        "{builtin}: application preference service is unavailable"
    ))
    .with_builtin(builtin)
    .with_identifier("RunMat:RuntimeContext:CapabilityUnavailable")
    .build()
}

pub(super) fn service_error(builtin: &'static str, source: PreferenceServiceError) -> RuntimeError {
    match source {
        PreferenceServiceError::Execution(source) => source.into_runtime_error(),
        source => build_runtime_error(format!(
            "{builtin}: application preferences failed: {source}"
        ))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:Storage"))
        .with_source(source)
        .build(),
    }
}

/// The selected service is invoked only after public values have completed
/// preparation. A later failure cannot request whole-builtin gather replay:
/// the service may already have admitted a storage effect or imported DATA.
pub(super) fn service_error_after_preparation(
    builtin: &'static str,
    source: PreferenceServiceError,
) -> RuntimeError {
    let mut error = service_error(builtin, source);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

pub(super) fn values_struct(values: impl IntoIterator<Item = (String, Value)>) -> StructValue {
    let mut output = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in values {
        output.insert(name, value);
    }
    output.finish()
}
