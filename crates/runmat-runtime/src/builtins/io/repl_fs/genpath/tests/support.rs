use std::convert::TryFrom;
use std::path::Path;

use runmat_value::Value;

pub(super) fn call(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(super::super::execute::run(args))
}

pub(super) fn text(value: Value) -> String {
    String::try_from(&value).expect("character row")
}

pub(super) fn segments(value: Value) -> Vec<String> {
    let text = text(value);
    if text.is_empty() {
        Vec::new()
    } else {
        text.split(crate::builtins::common::path_state::PATH_LIST_SEPARATOR)
            .map(str::to_owned)
            .collect()
    }
}

pub(super) fn canonical(path: &Path) -> String {
    super::super::root::canonical(path)
}
