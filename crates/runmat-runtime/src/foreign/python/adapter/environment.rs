//! Session-bound Python environment values and their public status fields.

use runmat_builtins::PYTHON_ENVIRONMENT_CLASS;
use runmat_python::{discover_python, PYTHON_ADAPTER_ID, PYTHON_ADAPTER_VERSION};
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::Value;

use super::{discovery_request, invalid_call, PythonRuntimeConfiguration};
use crate::foreign::{ForeignHandleRegistry, ForeignResourceMetadata};
use crate::RuntimeError;

pub(super) fn environment_metadata() -> ForeignResourceMetadata {
    ForeignResourceMetadata {
        type_identity: ForeignTypeIdentity {
            family: PYTHON_ADAPTER_ID.into(),
            name: PYTHON_ENVIRONMENT_CLASS.display_name().into(),
            version: PYTHON_ADAPTER_VERSION,
        },
        ownership: ForeignOwnership::Shared,
        affinity: ForeignAffinity::OriginProcess,
        lifetime: ForeignLifetime::Session,
    }
}

#[cfg(test)]
pub(super) fn validate_environment(
    handles: &ForeignHandleRegistry,
    host_identity: &str,
    arguments: &[Value],
) -> Result<(), RuntimeError> {
    let [Value::Object(environment)] = arguments else {
        return Err(invalid_call(
            "terminate expects the PythonEnvironment returned by pyenv",
        ));
    };
    validate_environment_object(handles, host_identity, environment)
}

pub(super) fn validate_rooted_environment(
    handles: &ForeignHandleRegistry,
    host_identity: &str,
    arguments: &crate::call::RootedFunctionArguments,
) -> Result<(), RuntimeError> {
    let Some(Value::Object(environment)) =
        (arguments.len() == 1).then(|| arguments.first()).flatten()
    else {
        return Err(invalid_call(
            "terminate expects the PythonEnvironment returned by pyenv",
        ));
    };
    validate_environment_object(handles, host_identity, environment)
}

fn validate_environment_object(
    handles: &ForeignHandleRegistry,
    host_identity: &str,
    environment: &runmat_value::ObjectInstance,
) -> Result<(), RuntimeError> {
    if !environment.is_class(PYTHON_ENVIRONMENT_CLASS) {
        return Err(invalid_call(
            "terminate expects the PythonEnvironment returned by pyenv",
        ));
    }
    let reference = environment
        .foreign_host_control()
        .ok_or_else(|| invalid_call("terminate expects the PythonEnvironment returned by pyenv"))?;
    if reference.host_identity != host_identity {
        return Err(invalid_call(
            "PythonEnvironment belongs to another Python host",
        ));
    }
    handles.resolve_host_control(reference, &environment_metadata())
}

pub(super) fn issued_python_status_value(
    runtime: Option<&crate::context::RuntimeContext>,
    handles: &ForeignHandleRegistry,
    host_identity: &str,
    configuration: &PythonRuntimeConfiguration,
    running: bool,
    terminated: bool,
    process_id: Option<u32>,
) -> Result<Value, RuntimeError> {
    let reference = handles.register_host_control(host_identity, environment_metadata())?;
    python_status_object(
        runtime,
        reference,
        configuration,
        running,
        terminated,
        process_id,
    )
}

fn python_status_object(
    runtime: Option<&crate::context::RuntimeContext>,
    reference: runmat_value::ForeignRef,
    configuration: &PythonRuntimeConfiguration,
    running: bool,
    terminated: bool,
    process_id: Option<u32>,
) -> Result<Value, RuntimeError> {
    let installation = discover_python(&discovery_request(configuration)).ok();
    let fields = [
        (
            "Version",
            Value::string(
                installation
                    .as_ref()
                    .map(|value| value.version.to_string())
                    .unwrap_or_default(),
            ),
        ),
        (
            "Executable",
            Value::string(
                installation
                    .as_ref()
                    .map(|value| value.executable.display().to_string())
                    .unwrap_or_default(),
            ),
        ),
        (
            "Library",
            Value::string(
                installation
                    .as_ref()
                    .map(|value| value.library.display().to_string())
                    .unwrap_or_default(),
            ),
        ),
        (
            "Home",
            Value::string(
                installation
                    .as_ref()
                    .map(|value| value.home.display().to_string())
                    .unwrap_or_default(),
            ),
        ),
        (
            "Status",
            Value::String(
                if running {
                    "Loaded"
                } else if terminated {
                    "Terminated"
                } else {
                    "NotLoaded"
                }
                .into(),
            ),
        ),
        (
            "ExecutionMode",
            Value::String(configuration.execution_mode.as_compatibility_name().into()),
        ),
        (
            "ProcessID",
            if let Some(process_id) = process_id {
                Value::Int(runmat_value::IntValue::U32(process_id))
            } else {
                Value::Num(f64::NAN)
            },
        ),
        (
            "ProcessName",
            Value::String(if running { "RunMat" } else { "" }.into()),
        ),
    ];
    crate::execution::object::construct_object_from_fields_with_native(
        runtime,
        PYTHON_ENVIRONMENT_CLASS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
        Some(crate::execution::object::ObjectNativeInitialization::ForeignHostControl(reference)),
    )
    // Status properties contain only text/scalars, never managed handles.
    .map(|status| status.value().clone())
}
