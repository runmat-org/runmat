use runmat_value::{NumericStorage, Tensor, Value};

use super::super::{error, plan::PartitionPlan};

pub(super) fn convert(tensor: Tensor, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    let plan = PartitionPlan::new(&tensor.shape, dimensions)?;
    let storage = tensor.into_numeric_storage().map_err(error::internal)?;
    let cells = plan
        .groups()
        .map(|indices| block(&storage, &indices?, &plan.slice_shape))
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    super::output(&plan, cells)
}

fn block(
    storage: &NumericStorage,
    indices: &[usize],
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    let selected = storage.gather(indices).map_err(error::internal)?;
    let single = matches!(selected, NumericStorage::F32(_));
    let tensor = Tensor::from_numeric_storage(selected, shape.to_vec()).map_err(error::internal)?;
    Ok(if single {
        Value::Tensor(tensor)
    } else {
        crate::builtins::common::tensor::tensor_into_value(tensor)
    })
}
