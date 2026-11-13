//! Pure call admission before interpreter discovery or startup.

use super::{invalid_call, ForeignRef, PythonValue, RuntimeError, Value};

pub(super) enum PreparedPythonCall {
    InvokeQualified {
        name: String,
        arguments: Vec<crate::call::RootedArgumentValue>,
    },
    GetMember {
        receiver: ForeignRef,
        name: String,
    },
    SetMember {
        receiver: ForeignRef,
        name: String,
        value: crate::call::RootedArgumentValue,
    },
    InvokeMember {
        receiver: ForeignRef,
        name: String,
        arguments: Vec<crate::call::RootedArgumentValue>,
    },
    GetItem {
        receiver: ForeignRef,
        index: PythonValue,
    },
    SetItem {
        receiver: ForeignRef,
        index: PythonValue,
        value: crate::call::RootedArgumentValue,
    },
    Iterate {
        receiver: ForeignRef,
    },
    Run {
        code: String,
        outputs: Vec<String>,
        inputs: Vec<(String, crate::call::RootedArgumentValue)>,
    },
    RunFile {
        path: String,
        arguments: Vec<String>,
        outputs: Vec<String>,
        inputs: Vec<(String, crate::call::RootedArgumentValue)>,
    },
}

pub(super) fn prepare_python_call(
    symbol: &str,
    arguments: crate::call::RootedFunctionArguments,
) -> Result<PreparedPythonCall, RuntimeError> {
    let mut arguments = arguments.try_into_values()?;
    let prepared = match symbol {
        "invoke_qualified" => PreparedPythonCall::InvokeQualified {
            name: rooted_string_argument(arguments.next(), "Python name")?,
            arguments: arguments.collect(),
        },
        "get_member" => PreparedPythonCall::GetMember {
            receiver: rooted_foreign_argument(arguments.next())?,
            name: rooted_string_argument(arguments.next(), "Python attribute")?,
        },
        "set_member" => PreparedPythonCall::SetMember {
            receiver: rooted_foreign_argument(arguments.next())?,
            name: rooted_string_argument(arguments.next(), "Python attribute")?,
            value: arguments
                .next()
                .ok_or_else(|| invalid_call("Python attribute assignment requires a value"))?,
        },
        "invoke_member" => PreparedPythonCall::InvokeMember {
            receiver: rooted_foreign_argument(arguments.next())?,
            name: rooted_string_argument(arguments.next(), "Python method")?,
            arguments: arguments.collect(),
        },
        "get_item" => PreparedPythonCall::GetItem {
            receiver: rooted_foreign_argument(arguments.next())?,
            index: rooted_python_indices(arguments.collect())?,
        },
        "set_item" => {
            let receiver = rooted_foreign_argument(arguments.next())?;
            let mut remaining = arguments.collect::<Vec<_>>();
            let value = remaining
                .pop()
                .ok_or_else(|| invalid_call("Python item assignment requires a value"))?;
            PreparedPythonCall::SetItem {
                receiver,
                index: rooted_python_indices(remaining)?,
                value,
            }
        }
        "iterate" => PreparedPythonCall::Iterate {
            receiver: rooted_foreign_argument(arguments.next())?,
        },
        "pyrun" => PreparedPythonCall::Run {
            code: rooted_string_argument(arguments.next(), "Python code")?,
            outputs: rooted_output_names(arguments.next())?,
            inputs: prepare_named_inputs(arguments.collect())?,
        },
        "pyrunfile" => {
            let path = rooted_string_argument(arguments.next(), "Python file")?;
            let outputs = rooted_output_names(arguments.next())?;
            let (script_arguments, inputs) = rooted_split_file_inputs(arguments.collect())?;
            PreparedPythonCall::RunFile {
                path,
                arguments: script_arguments,
                outputs,
                inputs: prepare_named_inputs(inputs)?,
            }
        }
        operation => {
            return Err(invalid_call(format!(
                "unknown Python adapter operation {operation}"
            )))
        }
    };
    Ok(prepared)
}

fn prepare_named_inputs(
    arguments: Vec<crate::call::RootedArgumentValue>,
) -> Result<Vec<(String, crate::call::RootedArgumentValue)>, RuntimeError> {
    if !arguments.len().is_multiple_of(2) {
        return Err(invalid_call("Python named inputs require name-value pairs"));
    }
    let mut inputs = Vec::with_capacity(arguments.len() / 2);
    let mut arguments = arguments.into_iter();
    while let Some(name) = arguments.next() {
        let name = rooted_string_argument(Some(name), "Python input name")?;
        let value = arguments.next().expect("even argument count");
        inputs.push((name, value));
    }
    Ok(inputs)
}

fn rooted_string_argument(
    value: Option<crate::call::RootedArgumentValue>,
    label: &str,
) -> Result<String, RuntimeError> {
    value
        .as_ref()
        .and_then(|value| {
            crate::builtins::common::validation::value_text_scalar_string(value.value())
        })
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| invalid_call(format!("{label} must be non-empty text")))
}

fn rooted_foreign_argument(
    value: Option<crate::call::RootedArgumentValue>,
) -> Result<ForeignRef, RuntimeError> {
    match value.as_ref().map(crate::call::RootedArgumentValue::value) {
        Some(Value::Foreign(reference)) => Ok(reference.clone()),
        _ => Err(invalid_call("Python receiver must be a Python object")),
    }
}

fn rooted_python_indices(
    values: Vec<crate::call::RootedArgumentValue>,
) -> Result<PythonValue, RuntimeError> {
    let values = values
        .iter()
        .map(|value| super::python_index_ref(Some(value.value())))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() == 1 {
        Ok(values.into_iter().next().expect("one index"))
    } else {
        Ok(PythonValue::Tuple(values))
    }
}

fn rooted_output_names(
    value: Option<crate::call::RootedArgumentValue>,
) -> Result<Vec<String>, RuntimeError> {
    super::output_names_ref(value.as_ref().map(crate::call::RootedArgumentValue::value))
}

fn rooted_split_file_inputs(
    mut arguments: Vec<crate::call::RootedArgumentValue>,
) -> Result<(Vec<String>, Vec<crate::call::RootedArgumentValue>), RuntimeError> {
    if arguments.len() >= 2 {
        if let Some(position) = arguments.iter().position(|argument| {
            crate::builtins::common::validation::value_text_scalar_string(argument.value())
                .is_some_and(|value| value.eq_ignore_ascii_case("Arguments"))
        }) {
            let Some(Value::Cell(values)) = arguments.get(position + 1).map(|value| value.value())
            else {
                return Err(invalid_call(
                    "pyrunfile Arguments must be a cell array of text",
                ));
            };
            let script_arguments = values
                .data()
                .iter()
                .map(|value| {
                    crate::builtins::common::validation::value_text_scalar_string(value)
                        .ok_or_else(|| invalid_call("pyrunfile Arguments must contain text"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            // Removing these two control owners cannot invalidate
            // `script_arguments`: every String above is an independent Python
            // command value. All non-control Values retain their original
            // exact/captured owners in this collection.
            arguments.drain(position..=position + 1);
            return Ok((script_arguments, arguments));
        }
    }
    Ok((Vec::new(), arguments))
}
