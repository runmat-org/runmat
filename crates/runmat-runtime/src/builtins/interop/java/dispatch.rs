//! Typed builtin-to-session dispatch and form-independent Java diagnostics.

#[cfg(not(target_arch = "wasm32"))]
use crate::sequence::ResolveValueSequence;
#[cfg(not(target_arch = "wasm32"))]
use runmat_builtins::JAVA_ERROR_HOST_UNAVAILABLE;
#[cfg(target_arch = "wasm32")]
use runmat_builtins::JAVA_ERROR_UNAVAILABLE;
use runmat_builtins::{BuiltinErrorDescriptor, JAVA_ERROR_INVALID};
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

#[derive(Clone, Copy)]
pub(super) enum JavaOperation {
    Construct,
    ConstructEdt,
    InvokeMember,
    InvokeMemberEdt,
    CallStatic,
    CallStaticEdt,
    NewArray,
    AddClasspath,
    RemoveClasspath,
    Classpath,
    SetClasspath,
    Status,
    Configure,
    #[cfg(not(target_arch = "wasm32"))]
    UseJava,
}

impl JavaOperation {
    #[cfg(not(target_arch = "wasm32"))]
    const fn symbol(self) -> &'static str {
        match self {
            Self::Construct => "construct",
            Self::ConstructEdt => "construct_edt",
            Self::InvokeMember => "invoke_member",
            Self::InvokeMemberEdt => "invoke_member_edt",
            Self::CallStatic => "call_static",
            Self::CallStaticEdt => "call_static_edt",
            Self::NewArray => "new_array",
            Self::AddClasspath => "add_classpath",
            Self::RemoveClasspath => "remove_classpath",
            Self::Classpath => "classpath",
            Self::SetClasspath => "set_classpath",
            Self::Status => "status",
            Self::Configure => "configure",
            Self::UseJava => "usejava",
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    const fn output_count(self) -> usize {
        match self {
            Self::AddClasspath | Self::RemoveClasspath | Self::SetClasspath => 0,
            _ => 1,
        }
    }
}

pub(super) async fn invoke_java(
    context: Option<crate::context::RuntimeContext>,
    operation: JavaOperation,
    arguments: Vec<Value>,
) -> BuiltinResult<Value> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (context, operation, arguments);
        Err(java_error(
            &JAVA_ERROR_UNAVAILABLE,
            JAVA_ERROR_UNAVAILABLE.message,
        ))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let symbol = operation.symbol();
        let output_count = operation.output_count();
        let context = context.ok_or_else(|| {
            java_error(
                &JAVA_ERROR_HOST_UNAVAILABLE,
                JAVA_ERROR_HOST_UNAVAILABLE.message,
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
                    adapter: super::JAVA_ADAPTER.into(),
                    symbol: symbol.into(),
                    arguments,
                    requested_outputs: output_count,
                },
            ))
            .await?;
        if output_count == 0 {
            if !sequence.is_empty() {
                return Err(invalid_call(format!(
                    "Java {symbol} returned an unexpected result value"
                )));
            }
            return Ok(Value::OutputList(Vec::new()));
        }
        let mut values = sequence.resolve(
            runmat_types::SequenceUse::RequireSingle,
            crate::sequence::SequenceResolutionContext::default(),
        )?;
        Ok(values.remove(0))
    }
}

pub(super) fn text_entries(arguments: Vec<Value>, builtin: &str) -> BuiltinResult<Vec<String>> {
    use crate::builtins::common::validation::value_text_scalar_string;

    let mut entries = Vec::new();
    for value in arguments {
        match value {
            Value::Cell(cell) => {
                for value in cell.data {
                    entries.push(
                        value_text_scalar_string(&value)
                            .filter(|text| !text.trim().is_empty())
                            .ok_or_else(|| {
                                invalid_call(format!("{builtin} entries must be non-empty text"))
                            })?,
                    );
                }
            }
            value => entries.push(
                value_text_scalar_string(&value)
                    .filter(|text| !text.trim().is_empty())
                    .ok_or_else(|| {
                        invalid_call(format!("{builtin} entries must be non-empty text"))
                    })?,
            ),
        }
    }
    if entries.is_empty() {
        return Err(invalid_call(format!(
            "{builtin} requires a classpath entry"
        )));
    }
    Ok(entries)
}

pub(super) fn invalid_call(message: impl Into<String>) -> crate::RuntimeError {
    java_error(&JAVA_ERROR_INVALID, message)
}

fn java_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("java");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}
