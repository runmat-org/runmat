use runmat_value::{ComplexStorage, ComplexTensor, Value};

use super::super::{error, plan::PartitionPlan};

pub(super) fn convert(tensor: ComplexTensor, dimensions: &[usize]) -> crate::BuiltinResult<Value> {
    let plan = PartitionPlan::new(&tensor.shape, dimensions)?;
    let storage = tensor.into_complex_storage();
    let cells = plan
        .groups()
        .map(|indices| block(&storage, &indices?, &plan.slice_shape))
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    super::output(&plan, cells)
}

fn block(
    storage: &ComplexStorage,
    indices: &[usize],
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    let selected = storage.gather(indices).map_err(error::internal)?;
    if let ComplexStorage::F64(values) = &selected {
        if values.len() == 1 {
            let value = values
                .first()
                .ok_or_else(|| error::internal("complex scalar storage is empty"))?;
            return Ok(Value::Complex(value.0, value.1));
        }
    }
    ComplexTensor::from_complex_storage(selected, shape.to_vec())
        .map(Value::ComplexTensor)
        .map_err(error::internal)
}
