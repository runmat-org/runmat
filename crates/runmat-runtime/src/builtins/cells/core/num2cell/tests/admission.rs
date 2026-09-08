use super::super::num2cell_builtin;
use futures::executor::block_on;
use runmat_value::{Tensor, Value};

#[test]
fn rejects_matrix_duplicate_and_out_of_range_dimensions() {
    let input = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    for dims in [
        Tensor::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).unwrap(),
        Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap(),
        Tensor::new(vec![3.0], vec![1, 1]).unwrap(),
    ] {
        assert!(block_on(num2cell_builtin(input(), vec![Value::Tensor(dims)])).is_err());
    }
}
