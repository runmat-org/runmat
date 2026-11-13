mod character;
mod complex;
mod containers;
mod numeric;
mod sparse;
mod values;

use runmat_value::{CellArray, Value};

use super::{error, input, plan::PartitionPlan};

pub(super) fn convert_managed_strings(
    array: runmat_value::StringArray,
    dimensions: Option<&Value>,
    runtime: &crate::context::RuntimeContext,
) -> crate::BuiltinResult<Value> {
    values::managed_strings::convert(array, dimensions, runtime)
}

pub(super) fn convert(
    value: Value,
    dimensions: &[usize],
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<Value> {
    match input::classify(value) {
        input::Input::Numeric(array) => numeric::convert(array, dimensions),
        input::Input::Sparse(array) => sparse::convert(array, dimensions),
        input::Input::Complex(array) => complex::convert(array, dimensions),
        input::Input::Logical(array) => values::logical(array, dimensions),
        input::Input::String(array) => values::strings(array, dimensions, runtime),
        input::Input::Character(array) => character::convert(array, dimensions),
        input::Input::Symbolic(array) => values::symbolic(array, dimensions),
        input::Input::Cell(array) => containers::cells(array, dimensions),
        input::Input::Object(array) => containers::objects(array, dimensions),
        input::Input::Scalar(value) if dimensions.is_empty() => {
            CellArray::from_column_major(vec![value], vec![1, 1])
                .map(Value::Cell)
                .map_err(error::internal)
        }
        input::Input::Scalar(_) => Err(error::invalid_input(
            "grouped dimensions require an array input",
        )),
        input::Input::Unsupported => Err(error::invalid_input("unsupported input value")),
    }
}

pub(super) fn output(plan: &PartitionPlan, cells: Vec<Value>) -> crate::BuiltinResult<Value> {
    // Numeric and other rank-zero scalar arrays own one element, but a cell
    // shape of [] denotes zero cells. Preserve that distinction at the cell
    // construction boundary without changing the partition plan's indexing.
    let shape = if plan.output_shape.is_empty() && cells.len() == 1 {
        vec![1, 1]
    } else {
        plan.output_shape.clone()
    };
    CellArray::new_with_shape(cells, shape)
        .map(Value::Cell)
        .map_err(error::internal)
}
