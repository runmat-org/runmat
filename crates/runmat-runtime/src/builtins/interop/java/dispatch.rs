//! Typed builtin-to-session dispatch and form-independent Java diagnostics.

#[cfg(not(target_arch = "wasm32"))]
use crate::sequence::ResolveValueSequence;
#[cfg(not(target_arch = "wasm32"))]
use runmat_builtins::JAVA_ERROR_HOST_UNAVAILABLE;
#[cfg(target_arch = "wasm32")]
use runmat_builtins::JAVA_ERROR_UNAVAILABLE;
use runmat_builtins::{BuiltinErrorDescriptor, JAVA_ERROR_INVALID};
use runmat_value::Value;

use crate::execution::RootedValue;
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
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    invoke_java_prepared(context, operation, |_| Ok(arguments)).await
}

/// Resolve the Java host before allocating a synthesized argument layout.
/// Public validation remains at the builtin boundary; exact text and final
/// slots are admitted only after the same host-availability decision that
/// precedes invocation.
pub(super) async fn invoke_java_prepared(
    context: Option<crate::context::RuntimeContext>,
    operation: JavaOperation,
    prepare: impl FnOnce(
        &crate::context::RuntimeContext,
    ) -> BuiltinResult<crate::call::BuiltinArguments>,
) -> BuiltinResult<RootedValue> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (context, operation, prepare);
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
        let arguments = prepare(&context)?.into_supplied_function_arguments(Some(&context))?;
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
            .await
            .into_invoked_result()?;
        sequence.try_map_owned(
            |sequence| {
                if output_count == 0 {
                    if !sequence.is_empty() {
                        return Err(invalid_call(format!(
                            "Java {symbol} returned an unexpected result value"
                        )));
                    }
                    return Ok(Value::output_list(Vec::new()));
                }
                let values = sequence.resolve(
                    runmat_types::SequenceUse::RequireSingle,
                    crate::sequence::SequenceResolutionContext::default(),
                )?;
                crate::sequence::into_first_output(values)?
                    .ok_or_else(|| invalid_call("Java callback returned no value"))
            },
            |error| error.into_runtime_error(),
        )
    }
}

pub(super) struct TextEntries {
    source: crate::call::BuiltinArguments,
    count: usize,
}

impl TextEntries {
    pub(super) fn len(&self) -> usize {
        self.count
    }

    pub(super) fn get(&self, index: usize) -> Option<&Value> {
        flattened_text_at(self.source.view(), index)
    }

    pub(super) fn entry_arguments(
        &self,
        index: usize,
        suffix: Option<&str>,
        runtime: &crate::context::RuntimeContext,
    ) -> BuiltinResult<crate::call::BuiltinArguments> {
        let count = 1 + usize::from(suffix.is_some());
        self.source
            .fork()
            .try_map_preserving_roots_in_context(count, runtime, |slot, source| {
                if slot == 0 {
                    let value = flattened_text_at(source.view(), index)
                        .expect("validated Java text entry index");
                    return crate::call::argument_text::convert(value, Some(runtime))
                        .map(|text| Value::String(text.into()));
                }
                crate::call::argument_text::copy_str(
                    suffix.expect("second Java argument has a suffix"),
                    Some(runtime),
                )
                .map(|text| Value::String(text.into()))
            })
    }

    pub(super) fn all_arguments(
        self,
        runtime: &crate::context::RuntimeContext,
    ) -> BuiltinResult<crate::call::BuiltinArguments> {
        let count = self.count;
        let mut outer = 0usize;
        let mut inner = 0usize;
        self.source
            .try_map_preserving_roots_in_context(count, runtime, move |_, source| {
                let value = next_flattened_text(source.view(), &mut outer, &mut inner)
                    .expect("validated Java text entry count");
                crate::call::argument_text::convert(value, Some(runtime))
                    .map(|text| Value::String(text.into()))
            })
    }
}

pub(super) fn text_entries(
    arguments: crate::call::BuiltinArguments,
    builtin: &str,
) -> BuiltinResult<TextEntries> {
    let mut count = 0usize;
    for value in arguments.iter() {
        let values = match value {
            Value::Cell(cell) => cell.data(),
            value => std::slice::from_ref(value),
        };
        for value in values {
            if !nonempty_text_scalar(value) {
                return Err(invalid_call(format!(
                    "{builtin} entries must be non-empty text"
                )));
            }
            count = count
                .checked_add(1)
                .ok_or_else(|| invalid_call(format!("{builtin} has too many entries")))?;
        }
    }
    if count == 0 {
        return Err(invalid_call(format!(
            "{builtin} requires a classpath entry"
        )));
    }
    Ok(TextEntries {
        source: arguments,
        count,
    })
}

pub(super) fn flattened_text_at(
    arguments: crate::call::FunctionArgumentView<'_>,
    mut index: usize,
) -> Option<&Value> {
    for value in arguments {
        let values = match value {
            Value::Cell(cell) => cell.data(),
            value => std::slice::from_ref(value),
        };
        if index < values.len() {
            return values.get(index);
        }
        index -= values.len();
    }
    None
}

fn next_flattened_text<'a>(
    arguments: crate::call::FunctionArgumentView<'a>,
    outer: &mut usize,
    inner: &mut usize,
) -> Option<&'a Value> {
    loop {
        let value = arguments.get(*outer)?;
        let values = match value {
            Value::Cell(cell) => cell.data(),
            value => std::slice::from_ref(value),
        };
        if let Some(value) = values.get(*inner) {
            *inner += 1;
            if *inner == values.len() {
                *outer += 1;
                *inner = 0;
            }
            return Some(value);
        }
        *outer += 1;
        *inner = 0;
    }
}

pub(super) fn text_scalar_eq(value: &Value, expected: &str, ignore_ascii_case: bool) -> bool {
    let compare = |actual: &str| {
        if ignore_ascii_case {
            actual.eq_ignore_ascii_case(expected)
        } else {
            actual == expected
        }
    };
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => compare(text.as_str()),
        Value::StringArray(strings)
            if strings.data().len() == 1
                && strings.rows == 1
                && strings.cols == 1
                && !strings.data()[0].is_missing() =>
        {
            compare(
                strings.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            )
        }
        Value::CharArray(chars)
            if chars.rows == 1
                && crate::builtins::common::shape::effective_rank(&chars.shape) == 2 =>
        {
            chars
                .data()
                .iter()
                .map(|unit| unit.raw())
                .eq(expected.encode_utf16())
                || (ignore_ascii_case
                    && chars
                        .data()
                        .iter()
                        .map(|unit| unit.raw())
                        .zip(expected.encode_utf16())
                        .all(|(left, right)| {
                            let lower = |unit: u16| {
                                if (65..=90).contains(&unit) {
                                    unit + 32
                                } else {
                                    unit
                                }
                            };
                            lower(left) == lower(right)
                        })
                    && chars.data().len() == expected.encode_utf16().count())
        }
        _ => false,
    }
}

fn nonempty_text_scalar(value: &Value) -> bool {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => !text.as_str().trim().is_empty(),
        Value::StringArray(strings)
            if strings.data().len() == 1
                && strings.rows == 1
                && strings.cols == 1
                && !strings.data()[0].is_missing() =>
        {
            !strings.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .trim()
                .is_empty()
        }
        Value::CharArray(chars)
            if chars.rows == 1
                && crate::builtins::common::shape::effective_rank(&chars.shape) == 2 =>
        {
            chars
                .data()
                .iter()
                .any(|character| !character.is_whitespace())
        }
        _ => false,
    }
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
