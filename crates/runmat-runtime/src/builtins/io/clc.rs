//! MATLAB-compatible `clc` builtin for clearing the host-visible console.

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, console, BuiltinResult};

#[runtime_builtin(
    name = "clc",
    category = "io",
    summary = "Clear the Command Window display.",
    keywords = "clc,console,clear screen",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::clc"
)]
async fn clc_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(build_runtime_error("clc: expected no input arguments")
            .with_builtin("clc")
            .build());
    }

    console::record_clear_screen();
    Ok(empty_return_value())
}

fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}
