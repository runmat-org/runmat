//! Session-bound Python environment values and their public status fields.

use runmat_builtins::PYTHON_ENVIRONMENT_CLASS;
use runmat_python::{discover_python, PYTHON_ADAPTER_ID, PYTHON_ADAPTER_VERSION};
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{ObjectInstance, Value};

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
    handles: &ForeignHandleRegistry,
    host_identity: &str,
    configuration: &PythonRuntimeConfiguration,
    running: bool,
    terminated: bool,
    process_id: Option<u32>,
) -> Result<Value, RuntimeError> {
    let reference = handles.register_host_control(host_identity, environment_metadata())?;
    let mut environment = python_status_object(configuration, running, terminated, process_id);
    environment = environment.with_foreign_host_control(reference);
    Ok(Value::Object(environment))
}

fn python_status_object(
    configuration: &PythonRuntimeConfiguration,
    running: bool,
    terminated: bool,
    process_id: Option<u32>,
) -> ObjectInstance {
    let installation = discover_python(&discovery_request(configuration)).ok();
    let mut environment = ObjectInstance::new(PYTHON_ENVIRONMENT_CLASS.owned());
    environment.properties.insert(
        "Version".into(),
        Value::String(
            installation
                .as_ref()
                .map(|value| value.version.to_string())
                .unwrap_or_default(),
        ),
    );
    environment.properties.insert(
        "Executable".into(),
        Value::String(
            installation
                .as_ref()
                .map(|value| value.executable.display().to_string())
                .unwrap_or_default(),
        ),
    );
    environment.properties.insert(
        "Library".into(),
        Value::String(
            installation
                .as_ref()
                .map(|value| value.library.display().to_string())
                .unwrap_or_default(),
        ),
    );
    environment.properties.insert(
        "Home".into(),
        Value::String(
            installation
                .as_ref()
                .map(|value| value.home.display().to_string())
                .unwrap_or_default(),
        ),
    );
    environment.properties.insert(
        "Status".into(),
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
    );
    environment.properties.insert(
        "ExecutionMode".into(),
        Value::String(configuration.execution_mode.as_compatibility_name().into()),
    );
    environment.properties.insert(
        "ProcessID".into(),
        if let Some(process_id) = process_id {
            Value::Int(runmat_value::IntValue::U32(process_id))
        } else {
            Value::Num(f64::NAN)
        },
    );
    environment.properties.insert(
        "ProcessName".into(),
        Value::String(if running { "RunMat" } else { "" }.into()),
    );
    environment
}
