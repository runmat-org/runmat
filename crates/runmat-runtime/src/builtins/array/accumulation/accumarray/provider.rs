use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::error;

pub(super) struct MaterializedRequest {
    pub subscripts: Value,
    pub data: Value,
    pub options: Vec<Value>,
    pub roots: crate::execution::RootedValueLease,
}

pub(super) async fn materialize(
    subscripts: Value,
    data: Value,
    options: Vec<Value>,
) -> BuiltinResult<MaterializedRequest> {
    let _inputs_root = crate::execution::RootedValueLease::for_values(
        [&subscripts, &data].into_iter().chain(options.iter()),
    )
    .map_err(|error| error.into_runtime_error())?;
    reject_resident_integer(&data, "input data")?;
    if let Some(fill) = options.get(2) {
        reject_resident_integer(fill, "fill value")?;
    }
    let subscripts = gather_if_needed_async(&subscripts).await?;
    let data = gather_if_needed_async(&data).await?;
    let options = gather_all(options).await?;
    let roots = crate::execution::RootedValueLease::for_values(
        [&subscripts, &data].into_iter().chain(options.iter()),
    )
    .map_err(|error| error.into_runtime_error())?;
    Ok(MaterializedRequest {
        subscripts,
        data,
        options,
        roots,
    })
}

fn reject_resident_integer(value: &Value, role: &str) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
    {
        return Err(error::invalid(format!(
            "accumarray: GPU {role} must be logical, single, or double"
        )));
    }
    Ok(())
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut materialized = Vec::with_capacity(values.len());
    for value in values {
        materialized.push(gather_if_needed_async(&value).await?);
    }
    Ok(materialized)
}
