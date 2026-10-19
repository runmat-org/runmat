mod admission;
mod containers;
mod numeric;

use super::*;
use futures::executor::block_on;
use runmat_value::Value;

fn run(value: Value, dims: Vec<Value>) -> Value {
    block_on(num2cell_builtin(value, dims)).expect("num2cell succeeds")
}
