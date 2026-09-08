mod dimensions;
mod extensions;
mod providers;

use futures::executor::block_on;
use runmat_value::Value;

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(super::cell_builtin(args))
}

fn output(args: Vec<Value>, shape: &[usize]) -> runmat_value::CellArray {
    let Value::Cell(cell) = run(args).expect("cell succeeds") else {
        panic!("expected cell array");
    };
    assert_eq!(cell.shape, shape);
    cell
}
