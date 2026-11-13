//! Exact character publication preserves the scalar collector's column-major order.
use super::*;
use runmat_value::{CharArray, SharedShape, Utf16CodeUnit};

pub(super) fn character(
    source: &Managed<'_>,
    shape: &[usize],
) -> Result<Value, UniformScalarError> {
    let normalized = if shape.is_empty() {
        &[1usize, 1][..]
    } else {
        shape
    };
    if normalized.len() > 2 {
        return Err(UniformScalarError::CHARACTER_RANK);
    }
    let rows = normalized.first().copied().unwrap_or(1);
    let columns = normalized.get(1).copied().unwrap_or(1);
    let count = rows
        .checked_mul(columns)
        .ok_or(UniformScalarError::SIZE_OVERFLOW)?;
    if count != source.values.initialized().len() {
        return Err(UniformScalarError::CHARACTER_COUNT);
    }
    // CharArray's historical normalization stores two dimensions even when
    // the supplied shape has rank one. Fill final row-major backing directly.
    let shape = [rows, columns];
    let bytes = ExactArrayStorage::<Utf16CodeUnit>::requested_bytes(count).and_then(|bytes| {
        bytes
            .checked_add(SharedShape::requested_bytes_for_rank(2)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    });
    let receipt = admit(source, bytes, 1 + usize::from(count != 0))?;
    let permit = receipt
        .output_permit(source.runtime)
        .map_err(|e| allocation(source.runtime, e))?;
    let data = ExactArrayStorage::try_begin(count, &permit);
    checkpoint(source, &receipt)?;
    let mut data = data.map_err(|e| allocation(source.runtime, e.into()))?;
    for row in 0..rows {
        for column in 0..columns {
            checkpoint(source, &receipt)?;
            let ClassifiedValue::Char(value) = source.values.initialized()[row + column * rows]
            else {
                return Err(invalid());
            };
            data.try_push(value).map_err(|_| invalid())?;
        }
    }
    let shape = SharedShape::new_admitted_from_slice(&shape, &permit);
    checkpoint(source, &receipt)?;
    let shape = shape.map_err(|e| allocation(source.runtime, e.into()))?;
    let data = data.try_finish().map_err(|_| invalid())?;
    let value =
        CharArray::from_admitted(data, shape).map_err(UniformScalarError::materialization)?;
    checkpoint(source, &receipt)?;
    Ok(Value::CharArray(value))
}
