use runmat_value::{CharArray, Value};

use crate::builtins::common::path_state::{
    current_path_string, set_path_string, PATH_LIST_SEPARATOR,
};
use crate::{runtime_descriptor_error, BuiltinResult};

const BUILTIN_NAME: &str = "path";

pub(super) async fn run(args: Vec<Value>) -> BuiltinResult<Value> {
    match args.len() {
        0 => Ok(path_value()),
        1 => replace_one(args.into_iter().next().expect("one path argument")).await,
        2 => {
            let mut args = args.into_iter();
            let left = args.next().expect("first path argument");
            let right = args.next().expect("second path argument");
            replace_two(left, right).await
        }
        _ => Err(runtime_descriptor_error(
            BUILTIN_NAME,
            &runmat_builtins::PATH_ERROR_TOO_MANY_INPUTS,
        )),
    }
}

fn path_value() -> Value {
    Value::CharArray(CharArray::new_row(&current_path_string()))
}

async fn replace_one(value: Value) -> BuiltinResult<Value> {
    let replacement = super::input::text(value).await?;
    replace(replacement)
}

async fn replace_two(left: Value, right: Value) -> BuiltinResult<Value> {
    let left = super::input::text(left).await?;
    let right = super::input::text(right).await?;
    replace(join(&left, &right))
}

fn replace(replacement: String) -> BuiltinResult<Value> {
    let previous = current_path_string();
    set_path_string(&replacement);
    Ok(Value::CharArray(CharArray::new_row(&previous)))
}

fn join(left: &str, right: &str) -> String {
    match (left.is_empty(), right.is_empty()) {
        (true, true) => String::new(),
        (false, true) => left.to_owned(),
        (true, false) => right.to_owned(),
        (false, false) => format!("{left}{PATH_LIST_SEPARATOR}{right}"),
    }
}
