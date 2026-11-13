//! Python interoperability builtins backed by the session foreign runtime.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::execution::RootedValue;
use crate::{build_runtime_error, BuiltinResult};

#[derive(Clone, Copy)]
enum PythonBuiltinOperation {
    Status,
    Configure,
    Run,
    RunFile,
    Terminate,
}

impl PythonBuiltinOperation {
    #[cfg(not(target_arch = "wasm32"))]
    const fn symbol(self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Configure => "configure",
            Self::Run => "pyrun",
            Self::RunFile => "pyrunfile",
            Self::Terminate => "terminate",
        }
    }
}
pub use runmat_builtins::{
    PYARGS_DESCRIPTOR, PYENV_DESCRIPTOR, PYRUNFILE_DESCRIPTOR, PYRUN_DESCRIPTOR,
    PYTHON_ENVIRONMENT_CLASS, PYTHON_ERROR_HOST_UNAVAILABLE, PYTHON_ERROR_INVALID,
    PYTHON_ERROR_UNAVAILABLE, TERMINATE_DESCRIPTOR,
};

async fn invoke_python(
    context: Option<crate::context::RuntimeContext>,
    operation: PythonBuiltinOperation,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (operation, arguments);
        let _ = context;
        Err(python_error(
            &PYTHON_ERROR_UNAVAILABLE,
            PYTHON_ERROR_UNAVAILABLE.message,
        ))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let symbol = operation.symbol();
        let context = context.ok_or_else(|| {
            python_error(
                &PYTHON_ERROR_HOST_UNAVAILABLE,
                PYTHON_ERROR_HOST_UNAVAILABLE.message,
            )
        })?;
        let service = context
            .service_ports()
            .require_foreign(symbol)
            .map_err(|error| error.into_runtime_error())?
            .clone();
        let sequence = context
            .scope(service.invoke(
                context.clone(),
                crate::context::ForeignCall {
                    adapter: runmat_python::PYTHON_ADAPTER_ID.into(),
                    symbol: symbol.into(),
                    arguments: arguments.into_function_arguments(),
                    requested_outputs: crate::current_requested_outputs(),
                },
            ))
            .await
            .into_invoked_result()?;
        sequence.try_map_owned(
            |sequence| {
                Ok::<_, crate::RuntimeError>(
                    crate::call::arguments::project_legacy_builtin_value_abi(sequence),
                )
            },
            |error| error.into_runtime_error(),
        )
    }
}

#[runtime_builtin(
    name = "pyenv",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::python"
)]
async fn pyenv_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    if arguments.is_empty() {
        invoke_python(context, PythonBuiltinOperation::Status, arguments).await
    } else {
        invoke_python(context, PythonBuiltinOperation::Configure, arguments).await
    }
}

#[runtime_builtin(
    name = "pyargs",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::python"
)]
fn pyargs_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    if !arguments.len().is_multiple_of(2) {
        return Err(invalid_call("pyargs requires name-value pairs"));
    }
    for name in arguments.iter().step_by(2) {
        if !crate::builtins::common::validation::value_is_text_scalar(name) {
            return Err(invalid_call("pyargs names must be text"));
        }
        let empty = match name {
            Value::String(runmat_value::StringElement::Text(text)) => text.is_empty(),
            Value::StringArray(array) => array.data()[0]
                .as_str()
                .ok_or_else(|| invalid_call("pyargs names must be text"))?
                .is_empty(),
            Value::String(_) => return Err(invalid_call("pyargs names must be text")),
            Value::CharArray(array) => array.data().is_empty(),
            _ => unreachable!("validated text scalar"),
        };
        if empty {
            return Err(invalid_call("pyargs names must be non-empty"));
        }
    }
    let count = arguments.len() / 2;
    let (names, values) = match context.as_ref() {
        Some(runtime) => {
            let names = crate::execution::RootedValues::try_initialize_in_context(
                count,
                runtime,
                |index| {
                    pyargs_name(
                        arguments
                            .get(index * 2)
                            .expect("validated pyargs name position"),
                        Some(runtime),
                    )
                    .map(|text| Value::String(text.into()))
                },
            )?;
            let names =
                crate::execution::structure::cell::from_rooted_values(runtime, &[1, count], names)?;
            let values = crate::execution::structure::cell::from_borrowed_values(
                Some(runtime),
                &[1, count],
                arguments.iter().skip(1).step_by(2),
            )?;
            (names, values)
        }
        None => {
            let names = arguments
                .iter()
                .step_by(2)
                .map(|name| pyargs_name(name, None).map(|text| Value::String(text.into())))
                .collect::<Result<Vec<_>, _>>()?;
            (
                RootedValue::new(Value::Cell(
                    runmat_value::CellArray::new(names, 1, count).map_err(invalid_call)?,
                ))
                .map_err(|source| source.into_runtime_error())?,
                crate::execution::structure::cell::from_borrowed_values(
                    None,
                    &[1, count],
                    arguments.iter().skip(1).step_by(2),
                )?,
            )
        }
    };
    let fields = [
        ("Names", names.value().clone()),
        ("Values", values.value().clone()),
    ];
    crate::execution::object::construct_object_from_borrowed_fields(
        context.as_ref(),
        runmat_types::standard::PYTHON_ARGUMENTS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
}

fn pyargs_name(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> Result<runmat_value::Utf8Text, crate::RuntimeError> {
    match value {
        // pyargs historically accepts character rows with trailing singleton
        // dimensions. Validation above proves the effective row geometry.
        Value::CharArray(array) => crate::call::argument_text::copy_chars(array.data(), runtime),
        _ => crate::call::argument_text::convert(value, runtime),
    }
}

#[runtime_builtin(
    name = "pyrun",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::python"
)]
async fn pyrun_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    if arguments.is_empty() {
        return Err(invalid_call("pyrun requires Python source code"));
    }
    let arguments = run_arguments(arguments, context.as_ref(), "pyrun")?;
    invoke_python(context, PythonBuiltinOperation::Run, arguments).await
}

#[runtime_builtin(
    name = "pyrunfile",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::python"
)]
async fn pyrunfile_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    if arguments.is_empty() {
        return Err(invalid_call("pyrunfile requires a Python file"));
    }
    let arguments = run_arguments(arguments, context.as_ref(), "pyrunfile")?;
    invoke_python(context, PythonBuiltinOperation::RunFile, arguments).await
}

#[runtime_builtin(
    name = "terminate",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::interop::python"
)]
async fn terminate_builtin(
    context: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    if arguments.len() != 1
        || !matches!(
            arguments.first(),
            Some(Value::Object(environment))
                if environment.class_name().is(PYTHON_ENVIRONMENT_CLASS)
        )
    {
        return Err(invalid_call(
            "terminate expects the PythonEnvironment returned by pyenv",
        ));
    }
    // The foreign adapter validates the environment's private session ticket.
    // Keep the exact caller object; public class checking alone is insufficient.
    invoke_python(context, PythonBuiltinOperation::Terminate, arguments).await
}

fn run_arguments(
    arguments: crate::call::BuiltinArguments,
    context: Option<&crate::context::RuntimeContext>,
    builtin: &str,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let requested = crate::current_requested_outputs();
    if requested == 0 {
        let runtime = context.ok_or_else(|| {
            #[cfg(not(target_arch = "wasm32"))]
            {
                python_error(
                    &PYTHON_ERROR_HOST_UNAVAILABLE,
                    PYTHON_ERROR_HOST_UNAVAILABLE.message,
                )
            }
            #[cfg(target_arch = "wasm32")]
            {
                python_error(&PYTHON_ERROR_UNAVAILABLE, PYTHON_ERROR_UNAVAILABLE.message)
            }
        })?;
        let outputs = crate::execution::structure::cell::from_values(Some(runtime), &[0, 0], &[])?;
        return arguments.try_insert_in_context(1, outputs, runtime);
    }
    if arguments.len() < 2 {
        return Err(invalid_call(format!(
            "{builtin} requires Python output names when outputs are requested"
        )));
    }
    // Source, requested names, and remaining keyword pairs already have the
    // adapter's exact order. Preserve their original segmented owner.
    Ok(arguments)
}

fn invalid_call(message: impl Into<String>) -> crate::RuntimeError {
    python_error(&PYTHON_ERROR_INVALID, message)
}

fn python_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("python");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "python/tests.rs"]
mod tests;
