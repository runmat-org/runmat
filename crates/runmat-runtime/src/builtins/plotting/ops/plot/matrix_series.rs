//! Native host series expansion for the existing plot orientation rules.
use super::{plot_err, BuiltinResult, Value, BUILTIN_NAME};
use crate::context::RuntimeContext;
use runmat_value::Tensor;

fn dims(tensor: &Tensor) -> (usize, usize) {
    (
        tensor.shape().first().copied().unwrap_or(1),
        tensor.shape().get(1).copied().unwrap_or(1),
    )
}

fn is_matrix(tensor: &Tensor) -> bool {
    let (rows, columns) = dims(tensor);
    rows > 1 && columns > 1 && tensor.shape().len() <= 2
}

fn select(tensor: &Tensor, indices: Vec<usize>) -> BuiltinResult<Value> {
    let len = indices.len();
    super::super::op_common::line_inputs::select_native(
        tensor,
        &indices,
        vec![len, 1],
        BUILTIN_NAME,
    )
    .map(Value::Tensor)
}

fn column(tensor: &Tensor, column: usize) -> BuiltinResult<Value> {
    let (rows, _) = dims(tensor);
    select(tensor, (column * rows..(column + 1) * rows).collect())
}

fn row(tensor: &Tensor, row: usize) -> BuiltinResult<Value> {
    let (rows, columns) = dims(tensor);
    select(
        tensor,
        (0..columns).map(|column| column * rows + row).collect(),
    )
}

fn vector(tensor: &Tensor) -> BuiltinResult<Value> {
    select(tensor, (0..tensor.len()).collect())
}

pub(super) fn split_matrix_series(
    x: Value,
    y: Value,
    implicit_x: bool,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<(Value, Value)>> {
    let (Value::Tensor(xt), Value::Tensor(yt)) = (&x, &y) else {
        return Ok(vec![(x, y)]);
    };
    if !is_matrix(xt) && !is_matrix(yt) {
        return Ok(vec![(x, y)]);
    }
    // Matrix expansion allocates output coordinates and control indices. There
    // is no complete finite owner for this path; do not borrow an Unbounded one.
    let work = super::super::coordinate_admission::CoordinateWork::admit(runtime, BUILTIN_NAME)?;
    xt.validate_shape().map_err(plot_err)?;
    yt.validate_shape().map_err(plot_err)?;
    let mismatch = || plot_err("X and Y inputs must have the same number of elements");
    let mut pairs = Vec::new();
    if is_matrix(yt) {
        let (rows, columns) = dims(yt);
        if implicit_x {
            let xs = Tensor::new(
                (1..=rows).map(|index| index as f64).collect(),
                vec![rows, 1],
            )
            .map_err(plot_err)?;
            for index in 0..columns {
                work.check()?;
                pairs.push((Value::Tensor(xs.clone()), column(yt, index)?));
            }
        } else if is_matrix(xt) {
            if dims(xt) != (rows, columns) {
                return Err(mismatch());
            }
            for index in 0..columns {
                work.check()?;
                pairs.push((column(xt, index)?, column(yt, index)?));
            }
        } else if xt.len() == rows {
            for index in 0..columns {
                work.check()?;
                pairs.push((vector(xt)?, column(yt, index)?));
            }
        } else if xt.len() == columns {
            for index in 0..rows {
                work.check()?;
                pairs.push((vector(xt)?, row(yt, index)?));
            }
        } else {
            return Err(mismatch());
        }
    } else {
        let (rows, columns) = dims(xt);
        if yt.len() == rows {
            for index in 0..columns {
                work.check()?;
                pairs.push((column(xt, index)?, vector(yt)?));
            }
        } else if yt.len() == columns {
            for index in 0..rows {
                work.check()?;
                pairs.push((row(xt, index)?, vector(yt)?));
            }
        } else {
            return Err(mismatch());
        }
    }
    work.finish(Ok(pairs))
}
