mod dissection;
mod host;
mod provider;

use futures::executor::block_on;
use runmat_value::Value;

use crate::BuiltinResult;

fn call(value: Value) -> BuiltinResult<Value> {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    block_on(super::log2_builtin(value))
}

fn call_with_outputs(value: Value, count: usize) -> BuiltinResult<Value> {
    let _outputs = crate::output_count::push_output_count(Some(count));
    call(value)
}

fn call_matlab(value: Value) -> BuiltinResult<Value> {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    block_on(super::log2_builtin(value))
}
