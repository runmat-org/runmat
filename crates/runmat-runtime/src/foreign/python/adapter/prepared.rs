//! Pure call admission before interpreter discovery or startup.

use super::{
    foreign_argument, invalid_call, output_names, python_indices, split_file_inputs,
    split_item_assignment_arguments, string_argument, ForeignRef, PythonValue, RuntimeError, Value,
};

pub(super) enum PreparedPythonCall {
    InvokeQualified {
        name: String,
        arguments: Vec<Value>,
    },
    GetMember {
        receiver: ForeignRef,
        name: String,
    },
    SetMember {
        receiver: ForeignRef,
        name: String,
        value: Value,
    },
    InvokeMember {
        receiver: ForeignRef,
        name: String,
        arguments: Vec<Value>,
    },
    GetItem {
        receiver: ForeignRef,
        index: PythonValue,
    },
    SetItem {
        receiver: ForeignRef,
        index: PythonValue,
        value: Value,
    },
    Iterate {
        receiver: ForeignRef,
    },
    Run {
        code: String,
        outputs: Vec<String>,
        inputs: Vec<(String, Value)>,
    },
    RunFile {
        path: String,
        arguments: Vec<String>,
        outputs: Vec<String>,
        inputs: Vec<(String, Value)>,
    },
}

pub(super) fn prepare_python_call(
    symbol: &str,
    arguments: Vec<Value>,
) -> Result<PreparedPythonCall, RuntimeError> {
    let mut arguments = arguments.into_iter();
    let prepared = match symbol {
        "invoke_qualified" => PreparedPythonCall::InvokeQualified {
            name: string_argument(arguments.next(), "Python name")?,
            arguments: arguments.collect(),
        },
        "get_member" => PreparedPythonCall::GetMember {
            receiver: foreign_argument(arguments.next())?,
            name: string_argument(arguments.next(), "Python attribute")?,
        },
        "set_member" => PreparedPythonCall::SetMember {
            receiver: foreign_argument(arguments.next())?,
            name: string_argument(arguments.next(), "Python attribute")?,
            value: arguments
                .next()
                .ok_or_else(|| invalid_call("Python attribute assignment requires a value"))?,
        },
        "invoke_member" => PreparedPythonCall::InvokeMember {
            receiver: foreign_argument(arguments.next())?,
            name: string_argument(arguments.next(), "Python method")?,
            arguments: arguments.collect(),
        },
        "get_item" => PreparedPythonCall::GetItem {
            receiver: foreign_argument(arguments.next())?,
            index: python_indices(arguments.collect())?,
        },
        "set_item" => {
            let receiver = foreign_argument(arguments.next())?;
            let (indices, value) = split_item_assignment_arguments(arguments.collect())?;
            PreparedPythonCall::SetItem {
                receiver,
                index: python_indices(indices)?,
                value,
            }
        }
        "iterate" => PreparedPythonCall::Iterate {
            receiver: foreign_argument(arguments.next())?,
        },
        "pyrun" => PreparedPythonCall::Run {
            code: string_argument(arguments.next(), "Python code")?,
            outputs: output_names(arguments.next())?,
            inputs: prepare_named_inputs(arguments.collect())?,
        },
        "pyrunfile" => {
            let path = string_argument(arguments.next(), "Python file")?;
            let outputs = output_names(arguments.next())?;
            let (script_arguments, inputs) = split_file_inputs(arguments.collect())?;
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

fn prepare_named_inputs(arguments: Vec<Value>) -> Result<Vec<(String, Value)>, RuntimeError> {
    if !arguments.len().is_multiple_of(2) {
        return Err(invalid_call("Python named inputs require name-value pairs"));
    }
    let mut inputs = Vec::with_capacity(arguments.len() / 2);
    let mut arguments = arguments.into_iter();
    while let Some(name) = arguments.next() {
        let name = string_argument(Some(name), "Python input name")?;
        let value = arguments.next().expect("even argument count");
        inputs.push((name, value));
    }
    Ok(inputs)
}
