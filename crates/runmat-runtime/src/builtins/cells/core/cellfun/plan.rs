use crate::BuiltinResult;
use runmat_value::{CellArray, Value};

use super::error;
use super::options::Invocation;

pub(super) struct Plan {
    pub(super) callable: super::callback::Callable,
    pub(super) cells: Vec<CellArray>,
    pub(super) extra_arguments: Vec<Value>,
    pub(super) uniform_output: bool,
    pub(super) error_handler: Option<super::callback::Callable>,
    pub(super) shape: Vec<usize>,
    pub(super) element_count: usize,
}

impl Plan {
    pub(super) fn build(invocation: Invocation) -> BuiltinResult<Self> {
        let (cells, extra_arguments) = partition_arguments(invocation.arguments)?;
        let shape = common_shape(&cells)?;
        let element_count = element_count(&shape)?;
        Ok(Self {
            callable: invocation.callable,
            cells,
            extra_arguments,
            uniform_output: invocation.uniform_output,
            error_handler: invocation.error_handler,
            shape,
            element_count,
        })
    }
}

fn partition_arguments(arguments: Vec<Value>) -> BuiltinResult<(Vec<CellArray>, Vec<Value>)> {
    let mut cells = Vec::new();
    let mut extra = Vec::new();
    let mut constants_started = false;
    for argument in arguments {
        match argument {
            Value::Cell(_) if constants_started => {
                return Err(error::invalid(
                    "cellfun: cell array inputs must precede extra arguments",
                ))
            }
            Value::Cell(cell) => cells.push(cell),
            value => {
                constants_started = true;
                extra.push(value);
            }
        }
    }
    if cells.is_empty() {
        return Err(error::invalid(
            "cellfun: expected at least one cell array input",
        ));
    }
    Ok((cells, extra))
}

fn common_shape(cells: &[CellArray]) -> BuiltinResult<Vec<usize>> {
    let shape = cells
        .first()
        .map(|cell| cell.shape.clone())
        .unwrap_or_default();
    if let Some(index) = cells.iter().position(|cell| cell.shape != shape) {
        return Err(error::invalid(format!(
            "cellfun: cell array input {} does not match the size of the first input",
            index + 1
        )));
    }
    Ok(shape)
}

fn element_count(shape: &[usize]) -> BuiltinResult<usize> {
    if shape.is_empty() {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| error::invalid("cellfun: cell array size exceeds platform limits"))
    })
}
