//! One explicit numerical operation boundary for host and provider parameters.
//! The canonical layer payload remains unchanged; same-class tensors retain
//! their Arc-backed storage instead of materializing another floating copy.

use runmat_value::{NumericDType, NumericStorage, Tensor};

use crate::builtins::deep_learning::deep_learning_error;
use crate::BuiltinResult;

pub(in crate::builtins::deep_learning) fn project_owned(
    tensor: Tensor,
    dtype: NumericDType,
    function: &'static str,
) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == dtype {
        return Ok(tensor);
    }
    let shape = tensor.shape().to_vec();
    let storage = match dtype {
        NumericDType::F32 => NumericStorage::F32(
            tensor
                .into_numeric_storage()
                .map_err(|error| deep_learning_error(function, error))?
                .materialize_f32(),
        ),
        NumericDType::F64 => NumericStorage::F64(tensor.materialize_f64()),
        _ => {
            return Err(deep_learning_error(
                function,
                format!("{function}: network execution requires single or double precision"),
            ))
        }
    };
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|error| deep_learning_error(function, error))
}
