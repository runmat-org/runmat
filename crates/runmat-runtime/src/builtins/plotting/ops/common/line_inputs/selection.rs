//! Borrowed native coordinate selection; series topology stays with its caller.
use crate::builtins::plotting::plotting_error;
use crate::BuiltinResult;
use runmat_value::{IntegerStorage, NumericStorage, Tensor};

pub(crate) fn select_native(
    tensor: &Tensor,
    indices: &[usize],
    shape: Vec<usize>,
    builtin: &'static str,
) -> BuiltinResult<Tensor> {
    macro_rules! selected {
        ($values:expr, $variant:ident) => {
            NumericStorage::$variant(gather($values, indices, builtin)?)
        };
    }
    let storage = if let Some(values) = tensor.as_f64_slice() {
        selected!(values, F64)
    } else if let Some(values) = tensor.as_f32_slice() {
        selected!(values, F32)
    } else {
        match tensor
            .integer_storage()
            .ok_or_else(|| plotting_error(builtin, "native coordinate storage unavailable"))?
        {
            IntegerStorage::I8(values) => selected!(values, I8),
            IntegerStorage::I16(values) => selected!(values, I16),
            IntegerStorage::I32(values) => selected!(values, I32),
            IntegerStorage::I64(values) => selected!(values, I64),
            IntegerStorage::U8(values) => selected!(values, U8),
            IntegerStorage::U16(values) => selected!(values, U16),
            IntegerStorage::U32(values) => selected!(values, U32),
            IntegerStorage::U64(values) => selected!(values, U64),
        }
    };
    Tensor::from_numeric_storage(storage, shape).map_err(|error| plotting_error(builtin, error))
}

fn gather<T: Copy>(
    values: &[T],
    indices: &[usize],
    builtin: &'static str,
) -> BuiltinResult<Vec<T>> {
    indices
        .iter()
        .map(|index| {
            values.get(*index).copied().ok_or_else(|| {
                plotting_error(builtin, "coordinate storage index exceeds input length")
            })
        })
        .collect()
}
