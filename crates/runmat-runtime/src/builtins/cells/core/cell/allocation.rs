use runmat_value::Value;

use super::{arguments::ParsedCell, error, prototype};

pub(super) fn build(parsed: ParsedCell) -> crate::BuiltinResult<Value> {
    let shape = normalize_shape(parsed.shape);
    let total = shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| error::invalid_size("requested size exceeds platform limits"))?;
    if total == 0 {
        return crate::make_cell_with_shape(Vec::new(), shape).map_err(error::internal);
    }
    let empty = prototype::empty_value(parsed.prototype.as_ref())?;
    let mut values = Vec::with_capacity(total);
    values.resize(total, empty);
    crate::make_cell_with_shape(values, shape).map_err(error::internal)
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    match shape.len() {
        0 => vec![0, 0],
        1 => vec![shape[0], shape[0]],
        _ => shape,
    }
}
