//! Shared square native sparse selection; triangular fast path precedes this.
mod owned;
mod rectangular;
use crate::{context::RuntimeContext, BuiltinResult};
pub(super) use rectangular::solve as solve_singular;
pub(in super::super) use rectangular::try_solve as try_rectangular;
use runmat_value::Value;

fn rhs_columns(shape: &[usize], order: usize) -> Option<usize> {
    (shape.first().copied() == Some(order) && shape.iter().skip(2).all(|&dim| dim == 1))
        .then(|| shape.get(1).copied().unwrap_or(1))
}

pub(in super::super) async fn try_solve(
    lhs: &Value,
    rhs: &Value,
    context: &RuntimeContext,
) -> BuiltinResult<Option<Value>> {
    let Value::SparseTensor(sparse) = lhs else {
        return Ok(None);
    };
    let order = sparse.rows;
    if order == 0 || order != sparse.cols {
        return Ok(None);
    }
    let columns = match rhs {
        Value::Tensor(tensor) => rhs_columns(tensor.shape(), order).filter(|_| {
            (sparse.as_f32_slice().is_some() && tensor.as_f32_slice().is_some())
                || (sparse.as_f64_slice().is_some() && tensor.as_f64_slice().is_some())
        }),
        Value::ComplexTensor(tensor) => rhs_columns(tensor.shape(), order).filter(|_| {
            (sparse.as_complex_f32_slice().is_some() && tensor.as_f32_slice().is_some())
                || (sparse.as_complex_f64_slice().is_some() && tensor.as_f64_slice().is_some())
        }),
        Value::Num(_) if order == 1 && sparse.as_f64_slice().is_some() => Some(1),
        Value::Complex(_, _) if order == 1 && sparse.as_complex_f64_slice().is_some() => Some(1),
        _ => None,
    };
    let Some(columns) = columns else {
        return Ok(None);
    };

    owned::solve(sparse, rhs, columns, context).await.map(Some)
}
