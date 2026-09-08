use runmat_value::{SparseTensor, Value};

use super::super::error;

pub(super) fn convert(array: SparseTensor, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    if array.is_complex() {
        let dense = array.to_dense_complex().map_err(error::internal)?;
        return super::complex::convert(dense, dimensions);
    }
    if array.is_logical() {
        let dense = array.to_dense_logical().map_err(error::internal)?;
        return super::values::logical(dense, dimensions);
    }
    let dense = array.to_dense().map_err(error::internal)?;
    super::numeric::convert(dense, dimensions)
}
