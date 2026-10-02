//! Consume a validated native integer readback without repeating a transfer.

use runmat_value::Tensor;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::validation::PreparedIntegerPayload;

pub(crate) fn prepared_real_integer_tensor(
    payload: Option<PreparedIntegerPayload>,
) -> Result<Option<Tensor>, String> {
    match payload {
        None => Ok(None),
        Some(PreparedIntegerPayload::Real(gathered)) => Tensor::new_integer(
            gpu_helpers::integer_storage_from_owned(gathered.data),
            gathered.shape,
        )
        .map(Some),
        Some(PreparedIntegerPayload::Complex(_)) => {
            Err("complex integer samples require owning arithmetic rejection".into())
        }
    }
}
