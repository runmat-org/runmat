//! Raw UTF-16 selection/scatter in the original page-wise physical layout.
use super::scalar::{checked_shape_len, index_out_of_bounds, shape_error};
use crate::{
    context::RuntimeContext,
    indexing::plan::{admission::Owner, IndexPlan},
    runtime_error::semantic_error,
    RuntimeError,
};
use runmat_value::{CharArray, SharedShape, Utf16CodeUnit, Value};

pub(super) fn read(
    context: Option<&RuntimeContext>,
    source: &CharArray,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        source
            .column_major_value_at(index as usize)
            .ok_or_else(index_out_of_bounds)?;
    }
    let shape = plan.row_major_output_shape_in_context(context)?;
    validate_length(plan.indices.len(), &shape)?;
    construct(
        context,
        shape,
        |physical, rows, cols| {
            let ordinal = CharArray::column_major_position(rows, cols, physical)
                .ok_or_else(index_out_of_bounds)?;
            source
                .column_major_value_at(plan.indices[ordinal] as usize)
                .ok_or_else(index_out_of_bounds)
        },
        |_, _, _| Ok(()),
    )
    .map(Value::CharArray)
}

enum Units<'a> {
    Array { value: &'a CharArray, index: usize },
    Text(std::str::EncodeUtf16<'a>),
}
impl Iterator for Units<'_> {
    type Item = Utf16CodeUnit;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Array { value, index } => {
                let result = value.column_major_value_at(*index);
                *index += 1;
                result
            }
            Self::Text(value) => value.next().map(Utf16CodeUnit::from_raw),
        }
    }
}
fn units(value: &Value) -> Result<Units<'_>, RuntimeError> {
    match value {
        Value::CharArray(value) => Ok(Units::Array { value, index: 0 }),
        Value::String(value) => value
            .as_str()
            .map(|text| Units::Text(text.encode_utf16()))
            .ok_or_else(|| {
                semantic_error(
                    "AssignmentTypeMismatch",
                    "missing string cannot be assigned to a character array",
                )
            }),
        _ => Err(semantic_error(
            "AssignmentTypeMismatch",
            "character assignment requires character or text data",
        )),
    }
}

fn unit_count(value: &Value, owner: &Owner<'_>) -> Result<usize, RuntimeError> {
    if let Value::CharArray(value) = value {
        return Ok(value.data().len());
    }
    let mut count = 0usize;
    for _ in units(value)? {
        if count.is_multiple_of(4096) {
            owner.check()?;
        }
        count = count.checked_add(1).ok_or_else(|| {
            semantic_error("ShapeOverflow", "character RHS exceeds platform limits")
        })?;
    }
    owner.check()?;
    Ok(count)
}

pub(super) fn assign(
    context: Option<&RuntimeContext>,
    mut value: CharArray,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = Owner::new(context)?;
    let rhs_len = unit_count(rhs, &owner)?;
    if rhs_len != 1 && rhs_len != plan.indices.len() {
        return Err(semantic_error(
            "AssignmentShapeMismatch",
            "character assignment size does not match the indexed selection",
        ));
    }
    let count = checked_shape_len(&plan.base_shape)?;
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        if index as usize >= count {
            return Err(index_out_of_bounds());
        }
    }
    let shape = plan.row_major_base_shape_in_context(context)?;
    validate_length(count, &shape)?;
    if value.shape == shape {
        let (rows, cols) = (value.rows, value.cols);
        if let Some(bytes) = value.try_data_mut_unique() {
            scatter(&owner, bytes, rows, cols, plan, rhs, rhs_len)?;
            owner.check()?;
            return Ok(Value::CharArray(value));
        }
    }
    construct(
        context,
        shape,
        |physical, rows, cols| {
            let logical = CharArray::column_major_position(rows, cols, physical)
                .ok_or_else(index_out_of_bounds)?;
            Ok(value
                .column_major_value_at(logical)
                .unwrap_or(Utf16CodeUnit::from_raw(u16::from(b' '))))
        },
        |bytes, rows, cols| scatter(&owner, bytes, rows, cols, plan, rhs, rhs_len),
    )
    .map(Value::CharArray)
}

fn scatter(
    owner: &Owner<'_>,
    bytes: &mut [Utf16CodeUnit],
    rows: usize,
    cols: usize,
    plan: &IndexPlan,
    rhs: &Value,
    rhs_len: usize,
) -> Result<(), RuntimeError> {
    let mut units = units(rhs)?;
    let scalar = if rhs_len == 1 { units.next() } else { None };
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        let physical = CharArray::row_major_position(rows, cols, index as usize)
            .ok_or_else(index_out_of_bounds)?;
        *bytes.get_mut(physical).ok_or_else(index_out_of_bounds)? = scalar
            .or_else(|| units.next())
            .ok_or_else(index_out_of_bounds)?;
    }
    Ok(())
}

fn validate_length(length: usize, shape: &[usize]) -> Result<(), RuntimeError> {
    let expected = CharArray::code_unit_count_for_shape(shape)
        .ok_or_else(|| shape_error("Char data shape exceeds platform limits"))?;
    if length != expected {
        return Err(shape_error(format!(
            "Char data length {length} doesn't match shape {shape:?} ({expected} elements)"
        )));
    }
    Ok(())
}

fn construct(
    context: Option<&RuntimeContext>,
    shape: SharedShape,
    mut initialize: impl FnMut(usize, usize, usize) -> Result<Utf16CodeUnit, RuntimeError>,
    finish: impl FnOnce(&mut [Utf16CodeUnit], usize, usize) -> Result<(), RuntimeError>,
) -> Result<CharArray, RuntimeError> {
    let owner = Owner::new(context)?;
    let shape = super::primitive_output::shape(context, &shape)?;
    let count = CharArray::code_unit_count_for_shape(&shape)
        .ok_or_else(|| shape_error("Char data shape exceeds platform limits"))?;
    let (rows, cols) = (shape[0], shape[1]);
    let mut data =
        crate::context::host_allocation::initialize_host_array(context, count, |index| {
            if index % 4096 == 0 {
                owner.check()?;
            }
            initialize(index, rows, cols)
        })?;
    finish(
        data.try_as_mut_slice_unique()
            .expect("new character backing is unique"),
        rows,
        cols,
    )?;
    owner.check()?;
    let output = if data.is_exact() && shape.is_admitted() {
        CharArray::from_admitted(data, shape)
    } else {
        CharArray::from_shared_storage(data, shape)
    }
    .map_err(shape_error)?;
    owner.check()?;
    Ok(output)
}
