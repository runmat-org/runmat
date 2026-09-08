use crate::builtins::common::broadcast;
use crate::{gather_if_needed_async, BuiltinResult};
use runmat_builtins::ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION;
use runmat_value::Value;

use super::error::arrayfun_flow;
use super::input::{ArrayData, ArrayInput};
use super::BUILTIN_NAME;

pub(super) struct InputPlan {
    pub(super) inputs: Vec<ArrayInput>,
    pub(super) output_shape: Vec<usize>,
}

impl InputPlan {
    pub(super) async fn prepare(
        inputs: Vec<Value>,
        has_provider_input: bool,
    ) -> BuiltinResult<Self> {
        let mut prepared = Vec::with_capacity(inputs.len());
        for value in inputs {
            validate_input(&value)?;
            let data = ArrayData::from_value(gather_if_needed_async(&value).await?)?;
            let shape = data.shape_vec();
            let strides = broadcast::compute_strides(&shape);
            prepared.push(ArrayInput {
                data,
                shape,
                strides,
            });
        }
        let output_shape = if has_provider_input {
            provider_shape(&prepared)?
        } else {
            host_shape(&prepared)?
        };
        Ok(Self {
            inputs: prepared,
            output_shape,
        })
    }
}

fn validate_input(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Cell(_) => Err(arrayfun_flow(
            "arrayfun: cell inputs are not supported (use cellfun instead)",
        )),
        Value::Struct(_) => Err(arrayfun_flow("arrayfun: struct inputs are not supported")),
        _ => Ok(()),
    }
}

fn provider_shape(inputs: &[ArrayInput]) -> BuiltinResult<Vec<usize>> {
    let first = inputs
        .first()
        .map(|input| input.shape.clone())
        .unwrap_or_default();
    inputs.iter().skip(1).try_fold(first, |shape, input| {
        broadcast::broadcast_shapes(BUILTIN_NAME, &shape, &input.shape).map_err(arrayfun_flow)
    })
}

fn host_shape(inputs: &[ArrayInput]) -> BuiltinResult<Vec<usize>> {
    let first = inputs
        .first()
        .map(|input| input.shape.as_slice())
        .unwrap_or_default();
    let non_scalar = inputs
        .iter()
        .filter(|input| input.len() != 1)
        .map(|input| input.shape.as_slice())
        .collect::<Vec<_>>();
    let target = non_scalar.first().copied().unwrap_or(first);
    if non_scalar.iter().skip(1).any(|shape| *shape != target) {
        return Err(arrayfun_flow(
            "arrayfun: host input does not match the size of the first array",
        ));
    }
    if inputs.iter().any(|input| input.shape != target) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(target.to_vec())
}
