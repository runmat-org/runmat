use crate::builtins::common::tensor::tensor_element_len;
use crate::indexing::selectors::IndexScalar;
use crate::runtime_error::semantic_error as mex;
use crate::RuntimeError;
use runmat_value::{CellArray, NumericScalar, Tensor, Value};

mod selection;

use selection::{expand_cell_subscripts, linear_index_from_subscripts, row_major_pos_from_linear};

const CELL_END_PLUS_TAG_MASK: u64 = 0xffff_ffff_0000_0000;
const CELL_END_PLUS_TAG_VALUE: u64 = 0x7ff8_c311_0000_0000;
const CELL_END_PLUS_OFFSET_MASK: u64 = 0x0000_0000_ffff_ffff;

fn map_cell_shape_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("ShapeMismatch", format!("{context}: {err}"))
}

fn writable_cell(
    mut cell: CellArray,
    context: Option<&crate::context::RuntimeContext>,
) -> Result<(CellArray, Option<crate::execution::RootedValueLease>), RuntimeError> {
    if cell.data_owner().is_unique() {
        return Ok((cell, None));
    }
    if !cell.data_owner().is_exact()
        && crate::context::host_allocation::HostAllocationReceipt::policy(context)
            .map_err(|source| crate::object::indexing::allocation_error(context, source))?
            == crate::context::host_allocation::HostAllocationPolicy::Unbounded
    {
        cell.make_data_unique_unbounded()
            .map_err(|error| map_cell_shape_error("cell write owner", error))?;
        return Ok((cell, None));
    }
    let (cell, roots) = crate::execution::structure::replacement::cell(&cell, context)?;
    Ok((cell, Some(roots)))
}

fn writable_cell_data(cell: &mut CellArray) -> Result<&mut [Value], RuntimeError> {
    cell.try_data_mut_unique().ok_or_else(|| {
        mex(
            "RunMat:Cell:SharedOwner",
            "cell replacement owner is unexpectedly shared",
        )
    })
}

fn borrowed_cell<'a>(
    context: Option<&crate::context::RuntimeContext>,
    shape: &[usize],
    values: impl Iterator<Item = &'a Value> + Clone,
) -> Result<(CellArray, crate::execution::RootedValueLease), RuntimeError> {
    let rooted = crate::execution::structure::cell::from_borrowed_values(context, shape, values)?;
    let (value, roots) = rooted.into_value_and_lease();
    let Value::Cell(cell) = value else {
        unreachable!("cell constructor returned a non-cell value")
    };
    Ok((cell, roots))
}

fn empty_numeric_cell_value(
    context: Option<&crate::context::RuntimeContext>,
) -> Result<Value, RuntimeError> {
    use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
    let error = |source| crate::object::indexing::allocation_error(context, source);
    let output =
        HostAllocationPlan::retained_tensor_bytes(runmat_value::NumericDType::F64, &[0, 0])
            .map_err(error)?;
    if let Some(receipt) =
        HostAllocationReceipt::admit(context, HostAllocationPlan::new(0, output).map_err(error)?)
            .map_err(error)?
    {
        let runtime = context.expect("managed filler has an explicit context");
        let value = receipt
            .exact_output_tensor(runtime, runmat_value::NumericDType::F64, &[0, 0])
            .map_err(error)?
            .into_tensor()
            .map(Value::Tensor)
            .map_err(|source| error(source.into()))?;
        receipt.checkpoint(runtime).map_err(error)?;
        return Ok(value);
    }
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|e| map_cell_shape_error("cell growth empty filler", e))
}

fn exact_index_from_f64(value: f64) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return None;
    }
    if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return None;
    }
    Some(rounded as i64)
}

fn parse_positive_cell_index(index: IndexScalar) -> Result<usize, RuntimeError> {
    index
        .positive_usize()
        .ok_or_else(|| mex("CellIndexOutOfBounds", "Cell index out of bounds"))
}

fn index_scalar_from_numeric(value: NumericScalar) -> Result<IndexScalar, RuntimeError> {
    match value {
        NumericScalar::F64(value) => exact_index_from_f64(value).map(IndexScalar::Signed),
        NumericScalar::F32(value) => {
            exact_index_from_f64(f64::from(value)).map(IndexScalar::Signed)
        }
        value => value
            .into_int_value()
            .map(|value| IndexScalar::from_int(&value)),
    }
    .ok_or_else(|| mex("CellIndexType", "Unsupported cell index type"))
}

fn parse_cell_index_value(value: &Value) -> Result<usize, RuntimeError> {
    let index = match value {
        Value::Num(n) => IndexScalar::Signed(
            exact_index_from_f64(*n)
                .ok_or_else(|| mex("CellIndexType", "Unsupported cell index type"))?,
        ),
        Value::Int(i) => IndexScalar::from_int(i),
        Value::Tensor(t)
            if tensor_element_len(t) == 1 && t.shape().iter().product::<usize>() == 1 =>
        {
            index_scalar_from_numeric(
                t.numeric_value_at(0)
                    .expect("validated scalar numeric tensor storage"),
            )?
        }
        other => {
            let n: f64 = other
                .try_into()
                .map_err(|_| mex("CellIndexType", "Unsupported cell index type"))?;
            IndexScalar::Signed(
                exact_index_from_f64(n)
                    .ok_or_else(|| mex("CellIndexType", "Unsupported cell index type"))?,
            )
        }
    };
    parse_positive_cell_index(index)
}

fn parse_cell_index_value_for_len(value: &Value, len: usize) -> Result<usize, RuntimeError> {
    match value {
        Value::Num(n) => {
            if let Some(idx) = resolve_cell_end_relative_index(*n, len)? {
                return Ok(idx);
            }
            if n.is_nan() {
                return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
            }
        }
        Value::Tensor(t) if t.len() == 1 && t.shape().iter().product::<usize>() == 1 => {
            let value = t
                .numeric_value_at(0)
                .expect("validated scalar numeric tensor storage");
            if matches!(value, NumericScalar::F64(_) | NumericScalar::F32(_)) {
                let scalar = value.materialize_f64();
                if let Some(idx) = resolve_cell_end_relative_index(scalar, len)? {
                    return Ok(idx);
                }
                if scalar.is_nan() {
                    return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
                }
            }
        }
        _ => {}
    }
    parse_cell_index_value(value)
}

fn parse_cell_index_values_for_assignment(value: &Value) -> Result<Vec<usize>, RuntimeError> {
    (0..cell_assignment_index_count(value))
        .map(|index| cell_assignment_index_at(value, index))
        .collect()
}

fn cell_assignment_index_count(value: &Value) -> usize {
    match value {
        // An empty numeric selector has zero brace-assignment destinations.
        // The prepared-output cardinality owner already reports zero for it;
        // do not misroute it through scalar index conversion.
        Value::Tensor(t) if t.is_empty() => 0,
        Value::Tensor(t) if tensor_element_len(t) > 1 => t.len(),
        _ => 1,
    }
}

fn cell_assignment_index_at(value: &Value, index: usize) -> Result<usize, RuntimeError> {
    match value {
        Value::Tensor(t) if tensor_element_len(t) > 1 => {
            parse_positive_cell_index(index_scalar_from_numeric(
                t.numeric_value_at(index)
                    .expect("validated numeric tensor storage"),
            )?)
        }
        _ => parse_cell_index_value(value),
    }
}

/// Same linear Cell brace-selector policy as the ordinary Value adapter,
/// retaining the final position table's exact admission through mutation.
pub(crate) fn resolve_cell_assignment_components_in_context(
    context: Option<&crate::context::RuntimeContext>,
    cell: &CellArray,
    components: &[crate::object::indexing::ObjectIndexComponent],
) -> Result<runmat_value::ExactArrayStorage<usize>, RuntimeError> {
    use crate::object::indexing::ObjectIndexComponent;
    let [component] = components else {
        return Err(mex("CellIndexType", "Unsupported cell index type"));
    };
    let count = match component {
        ObjectIndexComponent::Colon => cell.data().len(),
        ObjectIndexComponent::End => 1,
        ObjectIndexComponent::Value(value) => cell_assignment_index_count(value),
    };
    crate::context::host_allocation::initialize_host_array(
        context,
        count,
        |index| match component {
            ObjectIndexComponent::Colon => Ok(index + 1),
            ObjectIndexComponent::End => Ok(cell.data().len()),
            ObjectIndexComponent::Value(value) => cell_assignment_index_at(value, index),
        },
    )
}

fn decode_cell_end_plus(value: f64) -> Option<usize> {
    if !value.is_nan() {
        return None;
    }
    let bits = value.to_bits();
    if (bits & CELL_END_PLUS_TAG_MASK) != CELL_END_PLUS_TAG_VALUE {
        return None;
    }
    Some((bits & CELL_END_PLUS_OFFSET_MASK) as usize)
}

fn resolve_cell_end_relative_index(value: f64, len: usize) -> Result<Option<usize>, RuntimeError> {
    if let Some(offset) = decode_cell_end_plus(value) {
        let idx = len + offset;
        if idx < 1 || idx > len {
            return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
        }
        return Ok(Some(idx));
    }
    if value == 0.0 && value.is_sign_negative() {
        return Ok(Some(len));
    }
    if value < 0.0 {
        let offset = exact_index_from_f64(value)
            .ok_or_else(|| mex("CellIndexType", "Unsupported cell index type"))?;
        let idx = len as i64 + offset;
        if idx < 1 || idx > len as i64 {
            return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
        }
        return Ok(Some(idx as usize));
    }
    Ok(None)
}

fn is_empty_tensor(value: &Value) -> bool {
    let Value::Tensor(tensor) = value else {
        return false;
    };
    let (rows, cols) = match tensor.shape() {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    };
    tensor.is_empty() || rows == 0 || cols == 0
}

pub fn create_cell_2d(values: Vec<Value>, rows: usize, cols: usize) -> Result<Value, RuntimeError> {
    crate::make_cell_with_shape(values, vec![rows, cols])
        .map_err(|e| map_cell_shape_error("cell creation error", e))
}

/// Return the MATLAB-visible extent used by one selector in a cell indexing
/// expression. The final selector collapses any remaining trailing dimensions.
pub fn cell_selector_extent(
    cell: &CellArray,
    selector_count: usize,
    position: usize,
) -> Result<usize, RuntimeError> {
    selection::cell_selector_extent(cell, selector_count, position)
}

pub fn index_cell_value(ca: &CellArray, indices: &[usize]) -> Result<Value, RuntimeError> {
    match indices.len() {
        1 => {
            let i = indices[0];
            if i == 0 {
                return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
            }
            Ok(ca.data()[row_major_pos_from_linear(ca, i)?].clone())
        }
        0 => Err(mex(
            "UnsupportedCellIndexCount",
            "Unsupported number of cell indices",
        )),
        _ => {
            let linear = linear_index_from_subscripts(ca, indices)?;
            Ok(ca.data()[row_major_pos_from_linear(ca, linear)?].clone())
        }
    }
}

pub fn expand_cell_values(
    ca: &CellArray,
    indices: &[usize],
    out_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    let mut values: Vec<Value> = Vec::new();
    if indices.is_empty() {
        for idx in 1..=ca.data().len() {
            values.push(index_cell_value(ca, &[idx])?);
        }
    } else {
        values.push(index_cell_value(ca, indices)?);
    }
    if values.len() >= out_count {
        Ok(values.into_iter().take(out_count).collect())
    } else {
        let mut out = values;
        out.resize(out_count, Value::Num(0.0));
        Ok(out)
    }
}

pub fn expand_all_cell_values(ca: &CellArray) -> Result<Vec<Value>, RuntimeError> {
    (1..=ca.data().len())
        .map(|idx| index_cell_value(ca, &[idx]))
        .collect()
}

pub fn gather_cell_paren_linear_indices(
    ca: &CellArray,
    indices: &[usize],
    output_shape: &[usize],
) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    gather_cell_paren_linear_indices_in_context(context.as_ref(), ca, indices, output_shape)
}

pub(crate) fn gather_cell_paren_linear_indices_in_context(
    context: Option<&crate::context::RuntimeContext>,
    ca: &CellArray,
    indices: &[usize],
    output_shape: &[usize],
) -> Result<Value, RuntimeError> {
    // Validate before entering the constructor's repeatable borrowed view.
    for &idx in indices {
        row_major_pos_from_linear(ca, idx)?;
    }
    let fallback = [1, indices.len().max(1)];
    let shape = if output_shape.is_empty() {
        fallback.as_slice()
    } else {
        output_shape
    };
    let expected = shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| {
            map_cell_shape_error(
                "cell paren indexing error",
                "Cell data shape exceeds platform limits",
            )
        })?;
    if expected != indices.len() {
        return Err(map_cell_shape_error(
            "cell paren indexing error",
            format!(
                "Cell data length {} doesn't match shape {:?} ({} elements)",
                indices.len(),
                shape,
                expected
            ),
        ));
    }
    crate::execution::structure::cell::from_borrowed_values(
        context,
        shape,
        indices.iter().map(|&idx| {
            &ca.data()[row_major_pos_from_linear(ca, idx).expect("validated cell index")]
        }),
    )
    .map(|value| value.with_owned(|value| value))
}

pub fn expand_cell_indices(ca: &CellArray, indices: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    if indices.len() > 1 {
        expand_cell_subscripts(ca, indices)
    } else {
        selection::CellSelection::new(ca, indices)?.collect()
    }
}

/// Project the same brace selection directly into its final admitted output.
/// The original Cell children remain rooted until the completed output owns
/// its roots, including when a later selected index fails validation.
pub fn expand_cell_indices_in_context(
    context: &crate::context::RuntimeContext,
    cell: &CellArray,
    indices: &[Value],
) -> Result<crate::execution::RootedValues, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(Some(context))?;
    let selection = selection::CellSelection::new(cell, indices)?;
    let _source_roots = cell_deletion_roots(Some(context), cell)?;
    let output = crate::execution::RootedValues::try_initialize_in_context(
        selection.len(),
        context,
        |ordinal| {
            if ordinal % 4096 == 0 {
                owner.check()?;
            }
            selection.value_at(ordinal)
        },
    )?;
    owner.check()?;
    Ok(output)
}

/// Structural prefix reads have the ordinary brace expansion's one-output
/// padding/truncation contract. Validate the entire selection first, but do
/// not construct discarded output slots or a second unrooted collection.
pub(crate) fn read_cell_brace_single_in_context(
    context: Option<&crate::context::RuntimeContext>,
    cell: &CellArray,
    indices: &[Value],
) -> Result<crate::execution::RootedValue, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let selected = if indices.is_empty() {
        if cell.data().is_empty() {
            Value::Num(0.0)
        } else {
            index_cell_value(cell, &[1])?
        }
    } else {
        let selection = selection::CellSelection::new(cell, indices)?;
        let mut first = None;
        for index in 0..selection.len() {
            if index % 4096 == 0 {
                owner.check()?;
            }
            let value = selection.value_ref_at(index)?;
            if first.is_none() {
                first = Some(value);
            }
        }
        first.cloned().unwrap_or(Value::Num(0.0))
    };
    owner.check()?;
    let selected = crate::execution::RootedValue::in_optional_context(selected, context)?;
    owner.check()?;
    Ok(selected)
}

/// The selected child of an initializing fixed path is private until writeback.
/// Validate using the same scalar growth rules, but never grow/copy the cell
/// just to obtain a new member-assignment receiver.
pub(crate) fn read_cell_assignment_child_in_context(
    context: Option<&crate::context::RuntimeContext>,
    cell: &CellArray,
    positions: &[usize],
    intent: runmat_types::AssignmentChildCreation,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    if positions.contains(&0) {
        return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
    }
    let growing = match positions {
        [position] if *position > cell.data().len() => {
            linear_cell_growth_shape(cell, *position)?;
            true
        }
        [row, column] if *row > cell.rows || *column > cell.cols => {
            rectangular_cell_growth_shape(cell, *row, *column)?;
            true
        }
        _ => false,
    };
    let result = if growing {
        crate::sequence::assignment_container_in_context(context, intent)?
    } else {
        crate::execution::RootedValue::in_optional_context(
            index_cell_value(cell, positions)?,
            context,
        )?
    };
    owner.check()?;
    Ok(result)
}

pub(crate) fn resolve_cell_assignment_child_components_in_context(
    context: Option<&crate::context::RuntimeContext>,
    cell: &CellArray,
    components: &[crate::object::indexing::ObjectIndexComponent],
) -> Result<runmat_value::ExactArrayStorage<usize>, RuntimeError> {
    use crate::object::indexing::ObjectIndexComponent;
    if components.is_empty() {
        return Err(mex(
            "UnsupportedCellIndexCount",
            "Unsupported number of cell indices",
        ));
    }
    crate::context::host_allocation::initialize_host_array(context, components.len(), |axis| {
        match &components[axis] {
            ObjectIndexComponent::Value(value) if cell_assignment_index_count(value) == 1 => {
                cell_assignment_index_at(value, 0)
            }
            ObjectIndexComponent::Colon
                if cell_selector_extent(cell, components.len(), axis)? == 1 =>
            {
                Ok(1)
            }
            _ => Err(mex(
                "CellAssignmentArityMismatch",
                "nested cell assignment requires one destination",
            )),
        }
    })
}

fn linear_cell_growth_shape(cell: &CellArray, maximum: usize) -> Result<[usize; 2], RuntimeError> {
    if !(cell.data().is_empty() || cell.rows <= 1 || cell.cols <= 1) {
        return Err(mex(
            "UnsupportedCellGrowth",
            "Cell growth via linear brace assignment is only supported for vectors",
        ));
    }
    Ok(if cell.data().is_empty() || cell.rows <= 1 {
        [1, maximum]
    } else {
        [maximum, 1]
    })
}

fn rectangular_cell_growth_shape(
    cell: &CellArray,
    row: usize,
    column: usize,
) -> Result<[usize; 2], RuntimeError> {
    let shape = [cell.rows.max(row), cell.cols.max(column)];
    shape[0].checked_mul(shape[1]).ok_or_else(|| {
        mex(
            "CellSubscriptOutOfBounds",
            "Cell array expansion exceeds supported size",
        )
    })?;
    Ok(shape)
}

pub fn assign_cell_value<OnWrite>(
    ca: CellArray,
    indices: &[usize],
    rhs: Value,
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    assign_cell_value_in_context(None, ca, indices, rhs, on_write)
}

pub fn assign_cell_value_in_context<OnWrite>(
    context: Option<&crate::context::RuntimeContext>,
    ca: CellArray,
    indices: &[usize],
    rhs: Value,
    mut on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    match indices.len() {
        1 => {
            let i = indices[0];
            let old_len = ca.data().len();
            let (mut ca, _roots) = if i > old_len {
                if !(ca.data().is_empty() || ca.rows <= 1 || ca.cols <= 1) {
                    return Err(mex(
                        "UnsupportedCellGrowth",
                        "Cell growth via linear brace assignment is only supported for vectors",
                    ));
                }
                let filler = empty_numeric_cell_value(context)?;
                let shape = if old_len == 0 {
                    // MATLAB linear growth from empty cell shapes (for example `0x5` or `5x0`)
                    // normalizes to a row vector.
                    [1, i]
                } else if ca.rows <= 1 {
                    [1, i]
                } else {
                    [i, 1]
                };
                let old = ca.data();
                let values = (0..i).map(|position| old.get(position).unwrap_or(&filler));
                let (cell, roots) = borrowed_cell(context, &shape, values)?;
                (cell, Some(roots))
            } else {
                writable_cell(ca, context)?
            };
            let pos = row_major_pos_from_linear(&ca, i)?;
            if let Some(oldv) = ca.data().get(pos) {
                on_write(oldv, &rhs);
            }
            writable_cell_data(&mut ca)?[pos] = rhs;
            Ok(Value::Cell(ca))
        }
        2 => {
            let i = indices[0];
            let j = indices[1];
            if i == 0 || j == 0 {
                return Err(mex(
                    "CellSubscriptOutOfBounds",
                    "Cell subscript out of bounds",
                ));
            }
            let (mut ca, _roots) = if i > ca.rows || j > ca.cols {
                let old_cols = ca.cols;
                let [new_rows, new_cols] = rectangular_cell_growth_shape(&ca, i, j)?;
                let total = new_rows * new_cols;
                let filler = empty_numeric_cell_value(context)?;
                let old = ca.data();
                let old_rows = ca.rows;
                let values = (0..total).map(|linear| {
                    let row = linear / new_cols;
                    let column = linear % new_cols;
                    if row < old_rows && column < old_cols {
                        &old[row * old_cols + column]
                    } else {
                        &filler
                    }
                });
                let (cell, roots) = borrowed_cell(context, &[new_rows, new_cols], values)?;
                (cell, Some(roots))
            } else {
                writable_cell(ca, context)?
            };
            let lin = (i - 1) * ca.cols + (j - 1);
            if let Some(oldv) = ca.data().get(lin) {
                on_write(oldv, &rhs);
            }
            writable_cell_data(&mut ca)?[lin] = rhs;
            Ok(Value::Cell(ca))
        }
        0 => Err(mex(
            "UnsupportedCellIndexCount",
            "Unsupported number of cell indices",
        )),
        _ => {
            let linear = linear_index_from_subscripts(&ca, indices)?;
            let position = row_major_pos_from_linear(&ca, linear)?;
            let (mut ca, _roots) = writable_cell(ca, context)?;
            if let Some(old_value) = ca.data().get(position) {
                on_write(old_value, &rhs);
            }
            writable_cell_data(&mut ca)?[position] = rhs;
            Ok(Value::Cell(ca))
        }
    }
}

pub fn assign_cell_value_multi<OnWrite>(
    ca: CellArray,
    positions: &[usize],
    rhs_values: &[Value],
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    assign_cell_value_multi_in_context(None, ca, positions, rhs_values, on_write)
}

pub fn assign_cell_value_multi_in_context<OnWrite>(
    context: Option<&crate::context::RuntimeContext>,
    ca: CellArray,
    positions: &[usize],
    rhs_values: &[Value],
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    assign_cell_values_in_context(
        context,
        ca,
        positions,
        CellAssignmentSource::Elements(rhs_values),
        on_write,
    )
}

/// Ordinary fixed-RHS brace assignment repeats a borrowed scalar at the
/// mutation owner, not an intermediate RHS vector proportional to the target.
pub fn assign_cell_value_scalar_in_context<OnWrite>(
    context: Option<&crate::context::RuntimeContext>,
    ca: CellArray,
    positions: &[usize],
    rhs: &Value,
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    assign_cell_values_in_context(
        context,
        ca,
        positions,
        CellAssignmentSource::Replicate(rhs),
        on_write,
    )
}

enum CellAssignmentSource<'a> {
    Elements(&'a [Value]),
    Replicate(&'a Value),
}

fn assign_cell_values_in_context<OnWrite>(
    context: Option<&crate::context::RuntimeContext>,
    ca: CellArray,
    positions: &[usize],
    source: CellAssignmentSource<'_>,
    mut on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    if positions.is_empty() {
        return Ok(Value::Cell(ca));
    }
    if matches!(&source, CellAssignmentSource::Elements(values) if positions.len() != values.len())
    {
        return Err(mex(
            "CellAssignmentArityMismatch",
            "Cell brace assignment target count does not match source value count",
        ));
    }
    if positions.contains(&0) {
        return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
    }
    let old_len = ca.data().len();
    let (mut ca, _roots) = if let Some(max_pos) = positions.iter().copied().max() {
        if max_pos > old_len {
            let shape = linear_cell_growth_shape(&ca, max_pos)?;
            let filler = empty_numeric_cell_value(context)?;
            let old = ca.data();
            let values = (0..max_pos).map(|position| old.get(position).unwrap_or(&filler));
            let (cell, roots) = borrowed_cell(context, &shape, values)?;
            (cell, Some(roots))
        } else {
            writable_cell(ca, context)?
        }
    } else {
        writable_cell(ca, context)?
    };
    owner.check()?;
    for (index, position) in positions.iter().enumerate() {
        let rhs = match &source {
            CellAssignmentSource::Elements(values) => &values[index],
            CellAssignmentSource::Replicate(value) => *value,
        };
        if *position == 0 || *position > ca.data().len() {
            return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
        }
        let pos = row_major_pos_from_linear(&ca, *position)?;
        if let Some(oldv) = ca.data().get(pos) {
            on_write(oldv, rhs);
        }
        writable_cell_data(&mut ca)?[pos] = rhs.clone();
    }
    Ok(Value::Cell(ca))
}

pub fn resolve_cell_assignment_positions(
    ca: &CellArray,
    indices: &[Value],
) -> Result<Vec<usize>, RuntimeError> {
    if indices.len() != 1 {
        return Err(mex("CellIndexType", "Unsupported cell index type"));
    }
    let _ = ca;
    parse_cell_index_values_for_assignment(&indices[0])
}

/// Preserve the ordinary Value-selector assignment policy, but construct the
/// final native position table under the selected operation's exact owner.
pub fn resolve_cell_assignment_positions_in_context(
    context: Option<&crate::context::RuntimeContext>,
    cell: &CellArray,
    indices: &[Value],
) -> Result<runmat_value::ExactArrayStorage<usize>, RuntimeError> {
    let [index] = indices else {
        return Err(mex("CellIndexType", "Unsupported cell index type"));
    };
    let _ = cell;
    crate::context::host_allocation::initialize_host_array(
        context,
        cell_assignment_index_count(index),
        |ordinal| cell_assignment_index_at(index, ordinal),
    )
}

pub fn delete_cell_linear(ca: CellArray, idx: usize) -> Result<Value, RuntimeError> {
    delete_cell_linear_in_context(None, ca, idx)
}

pub fn delete_cell_linear_in_context(
    context: Option<&crate::context::RuntimeContext>,
    ca: CellArray,
    idx: usize,
) -> Result<Value, RuntimeError> {
    let total = ca.data().len();
    if idx == 0 || idx > total {
        return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
    }
    if !(ca.rows == 1 || ca.cols == 1) {
        return Err(mex(
            "UnsupportedCellDeletion",
            "Linear cell deletion is only supported for vectors",
        ));
    }
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let _source_roots = cell_deletion_roots(context, &ca)?;
    let len = ca.data().len() - 1;
    let shape = if len == 0 {
        [0, 0]
    } else if ca.rows == 1 {
        [1, len]
    } else {
        [len, 1]
    };
    let position = idx - 1;
    let values = ca
        .data()
        .iter()
        .enumerate()
        .filter(move |(index, _)| *index != position)
        .map(|(_, value)| value);
    let (cell, _roots) = borrowed_cell(context, &shape, values)?;
    owner.check()?;
    Ok(Value::Cell(cell))
}

pub fn assign_cell_paren_with_policy(
    ca: CellArray,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    assign_cell_paren_with_policy_in_context(None, ca, indices, rhs, delete)
}

/// Planned deletion consumes native zero-based positions without a separate
/// one-based index vector at the dispatch boundary.
pub fn delete_cell_with_plan_in_context(
    context: Option<&crate::context::RuntimeContext>,
    ca: CellArray,
    plan: &crate::indexing::plan::IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let one_based = |index: u32| {
        usize::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .ok_or_else(|| mex("IndexOutOfBounds", "cell index exceeds the supported range"))
    };
    // Keep the dispatch's original conversion-before-RHS diagnostic order.
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        one_based(index)?;
    }
    validate_cell_delete(&ca, rhs)?;
    delete_cell_positions(context, &ca, plan.indices.len(), |ordinal| {
        row_major_pos_from_linear(&ca, one_based(plan.indices[ordinal])?)
    })
}

fn validate_cell_delete(ca: &CellArray, rhs: &Value) -> Result<(), RuntimeError> {
    if !is_empty_tensor(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Cell deletion requires empty RHS",
        ));
    }
    if !(ca.rows == 1 || ca.cols == 1) {
        return Err(mex(
            "UnsupportedCellDeletion",
            "Linear cell deletion is only supported for vectors",
        ));
    }
    Ok(())
}

fn delete_cell_positions(
    context: Option<&crate::context::RuntimeContext>,
    ca: &CellArray,
    count: usize,
    mut position: impl FnMut(usize) -> Result<usize, RuntimeError>,
) -> Result<Value, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let _source_roots = cell_deletion_roots(context, ca)?;
    let positions =
        crate::context::host_allocation::initialize_host_array(context, count, |index| {
            if index % 4096 == 0 {
                owner.check()?;
            }
            position(index)
        })?;
    let mut positions = positions
        .try_into_owned()
        .unwrap_or_else(|_| unreachable!("fresh deletion positions have one owner"));
    positions.as_mut_slice().sort_unstable();
    let mut unique = 0;
    for index in 0..positions.len() {
        if index % 4096 == 0 {
            owner.check()?;
        }
        if unique == 0 || positions.as_slice()[index] != positions.as_slice()[unique - 1] {
            let value = positions.as_slice()[index];
            positions.as_mut_slice()[unique] = value;
            unique += 1;
        }
    }
    owner.check()?;
    let removed = &positions.as_slice()[..unique];
    let len = ca.data().len() - unique;
    let shape = if len == 0 {
        [0, 0]
    } else if ca.rows == 1 {
        [1, len]
    } else {
        [len, 1]
    };
    let values = ca
        .data()
        .iter()
        .enumerate()
        .filter(|(index, _)| removed.binary_search(index).is_err())
        .map(|(_, value)| value);
    let (cell, _roots) = borrowed_cell(context, &shape, values)?;
    owner.check()?;
    Ok(Value::Cell(cell))
}

pub fn assign_cell_paren_with_policy_in_context(
    context: Option<&crate::context::RuntimeContext>,
    ca: CellArray,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if delete {
        if indices.len() != 1 {
            return Err(mex(
                "UnsupportedCellDeletion",
                "Linear cell deletion is only supported for vector indices",
            ));
        }
        if !is_empty_tensor(rhs) {
            return Err(mex(
                "DeletionRequiresEmptyRhs",
                "Cell deletion requires empty RHS",
            ));
        }
        return delete_cell_linear_in_context(context, ca, indices[0]);
    }
    if let Value::Cell(rhs_cell) = rhs {
        return assign_cell_paren_from_cell(context, ca, indices, rhs_cell);
    }
    Err(mex(
        "UnsupportedCellParenAssignment",
        "Cell paren assignment requires a cell RHS",
    ))
}

fn cell_deletion_roots(
    context: Option<&crate::context::RuntimeContext>,
    ca: &CellArray,
) -> Result<crate::execution::RootedValueLease, RuntimeError> {
    match context {
        Some(context) => crate::execution::RootedValueLease::for_values_iterator_in_context(
            ca.data().iter(),
            context,
        ),
        None => crate::execution::RootedValueLease::for_values(ca.data().iter())
            .map_err(|error| error.into_runtime_error()),
    }
}

pub fn assign_cell_paren_linear_indices_with_policy(
    ca: CellArray,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    assign_cell_paren_linear_indices_with_policy_in_context(None, ca, indices, rhs, delete)
}

pub fn assign_cell_paren_linear_indices_with_policy_in_context(
    context: Option<&crate::context::RuntimeContext>,
    ca: CellArray,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if delete {
        validate_cell_delete(&ca, rhs)?;
        return delete_cell_positions(context, &ca, indices.len(), |ordinal| {
            row_major_pos_from_linear(&ca, indices[ordinal])
        });
    }
    let Value::Cell(rhs_cell) = rhs else {
        return Err(mex(
            "UnsupportedCellParenAssignment",
            "Cell paren assignment requires a cell RHS",
        ));
    };
    if rhs_cell.data().len() != indices.len() && rhs_cell.data().len() != 1 {
        return Err(mex(
            "UnsupportedCellParenAssignment",
            "Cell RHS must be scalar or match assignment size",
        ));
    }
    for &idx in indices {
        row_major_pos_from_linear(&ca, idx)?;
    }
    let (mut ca, _roots) = writable_cell(ca, context)?;
    for (k, &idx) in indices.iter().enumerate() {
        let pos = row_major_pos_from_linear(&ca, idx)?;
        let rhs_pos = if rhs_cell.data().len() == 1 { 0 } else { k };
        let newv = rhs_cell.data()[rhs_pos].clone();
        if let Some(oldv) = ca.data().get(pos) {
            runmat_gc::gc_record_write(oldv, &newv);
        }
        writable_cell_data(&mut ca)?[pos] = newv;
    }
    Ok(Value::Cell(ca))
}

fn assign_cell_paren_from_cell(
    context: Option<&crate::context::RuntimeContext>,
    ca: CellArray,
    indices: &[usize],
    rhs: &CellArray,
) -> Result<Value, RuntimeError> {
    if rhs.data().len() != 1 {
        return Err(mex(
            "UnsupportedCellParenAssignment",
            "Only scalar cell paren assignment is supported",
        ));
    }
    let newv = rhs.data()[0].clone();
    let lin = match indices.len() {
        1 => {
            let i = indices[0];
            row_major_pos_from_linear(&ca, i)?
        }
        2 => {
            let i = indices[0];
            let j = indices[1];
            if i == 0 || i > ca.rows || j == 0 || j > ca.cols {
                return Err(mex(
                    "CellSubscriptOutOfBounds",
                    "Cell subscript out of bounds",
                ));
            }
            (i - 1) * ca.cols + (j - 1)
        }
        0 => {
            return Err(mex(
                "UnsupportedCellIndexCount",
                "Unsupported number of cell indices",
            ))
        }
        _ => {
            let linear = linear_index_from_subscripts(&ca, indices)?;
            row_major_pos_from_linear(&ca, linear)?
        }
    };
    let (mut ca, _roots) = writable_cell(ca, context)?;
    if let Some(oldv) = ca.data().get(lin) {
        runmat_gc::gc_record_write(oldv, &newv);
    }
    writable_cell_data(&mut ca)?[lin] = newv;
    Ok(Value::Cell(ca))
}

#[cfg(test)]
mod tests {
    use super::{expand_cell_indices, map_cell_shape_error, resolve_cell_assignment_positions};
    use runmat_value::{CellArray, IntegerStorage, Tensor, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn cell_index_and_deletion_preserve_rank_boundaries() {
        let cell = || CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).unwrap();
        for shape in [vec![], vec![1]] {
            let index = Value::Tensor(Tensor::new(vec![2.0], shape).unwrap());
            assert_eq!(
                expand_cell_indices(&cell(), &[index]).unwrap(),
                vec![Value::Num(20.0)]
            );
        }
        let scalar_rhs = Value::Tensor(Tensor::new(vec![7.0], Vec::new()).unwrap());
        let Value::Cell(result) =
            super::assign_cell_paren_with_policy(cell(), &[1], &scalar_rhs, true).unwrap()
        else {
            panic!("expected cell after deletion");
        };
        assert_eq!(result.data(), &[Value::Num(20.0)]);
        let vector_rhs = Value::Tensor(Tensor::new(vec![7.0], vec![1]).unwrap());
        let error =
            super::assign_cell_paren_with_policy(cell(), &[1], &vector_rhs, true).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:DeletionRequiresEmptyRhs"));
    }

    #[test]
    fn cell_shape_error_mapping_reports_identifier() {
        let err = map_cell_shape_error("cell creation", "invalid shape");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn assign_cell_value_linear_growth_records_write_barrier() {
        let cell = CellArray::new(Vec::new(), 0, 0).expect("empty cell");
        let mut writes = Vec::new();
        let result = super::assign_cell_value(cell, &[3], Value::Num(42.0), |old, new| {
            writes.push((old.clone(), new.clone()));
        })
        .expect("linear growth assignment");

        assert_eq!(writes.len(), 1);
        let empty_filler = Tensor::new(Vec::new(), vec![0, 0]).expect("empty filler");
        assert_eq!(writes[0], (Value::Tensor(empty_filler), Value::Num(42.0)));
        let Value::Cell(cell) = result else {
            panic!("expected cell result");
        };
        assert_eq!(cell.data()[2], Value::Num(42.0));
    }

    #[test]
    fn scalar_brace_replication_uses_the_shared_mutation_and_barrier_order() {
        let cell = CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).unwrap();
        let positions = [2, 1, 2];
        let rhs = Value::Num(7.0);
        let mut scalar_writes = Vec::new();
        let scalar = super::assign_cell_value_scalar_in_context(
            None,
            cell.clone(),
            &positions,
            &rhs,
            |old, new| scalar_writes.push((old.clone(), new.clone())),
        )
        .unwrap();
        let mut explicit_writes = Vec::new();
        let explicit = super::assign_cell_value_multi(
            cell,
            &positions,
            &[rhs.clone(), rhs.clone(), rhs],
            |old, new| explicit_writes.push((old.clone(), new.clone())),
        )
        .unwrap();
        assert_eq!(scalar, explicit);
        assert_eq!(scalar_writes, explicit_writes);
        assert_eq!(scalar_writes.len(), positions.len());
    }

    #[test]
    fn bounded_typed_brace_positions_and_mutation_retain_original_payloads() {
        use crate::context::{
            host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeContext,
            RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
        };
        use crate::object::indexing::ObjectIndexComponent;
        use std::rc::Rc;
        struct Parallel(ManagedHostEffectLedger);
        impl RuntimeParallelService for Parallel {
            fn supports(&self, _: ParallelCapability) -> bool {
                false
            }
            fn placement_resources(&self) -> RuntimeParallelResources {
                let state = self.0.snapshot();
                RuntimeParallelResources {
                    cpu_millicores_available: 1000,
                    memory_available_bytes: Some(state.free_bytes()),
                    epoch: state.epoch,
                }
            }
            fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
                Some(self.0.clone())
            }
        }
        let context = |ledger: &ManagedHostEffectLedger| {
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
                )
        };
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let runtime = context(&ledger);
        {
            let original = CellArray::new(
                vec![
                    Value::Num(11.),
                    Value::Num(12.),
                    Value::Num(21.),
                    Value::Num(22.),
                ],
                2,
                2,
            )
            .unwrap();
            let indices = [ObjectIndexComponent::Value(Value::Tensor(
                Tensor::new(vec![4., 1., 4.], vec![1, 3]).unwrap(),
            ))];
            let positions = super::resolve_cell_assignment_components_in_context(
                Some(&runtime),
                &original,
                &indices,
            )
            .unwrap();
            assert_eq!(positions.as_slice(), &[4, 1, 4]);
            let native_indices = [Value::Tensor(
                Tensor::new(vec![4., 1., 4.], vec![1, 3]).unwrap(),
            )];
            let native_positions = super::resolve_cell_assignment_positions_in_context(
                Some(&runtime),
                &original,
                &native_indices,
            )
            .unwrap();
            assert_eq!(native_positions.as_slice(), positions.as_slice());
            let selected =
                super::expand_cell_indices_in_context(&runtime, &original, &native_indices)
                    .unwrap();
            assert_eq!(
                selected.values(),
                &[Value::Num(22.), Value::Num(11.), Value::Num(22.)]
            );
            let invalid = [Value::Tensor(
                Tensor::new(vec![1., 5.], vec![1, 2]).unwrap(),
            )];
            let error = super::expand_cell_indices_in_context(&runtime, &original, &invalid)
                .expect_err("later selected index must fail");
            assert_eq!(error.identifier(), Some("RunMat:CellIndexOutOfBounds"));
            let rhs = Value::Tensor(Tensor::new(vec![7., 8.], vec![1, 2]).unwrap());
            let Value::Tensor(tensor) = &rhs else {
                unreachable!()
            };
            let pointer = tensor.as_f64_slice().unwrap().as_ptr();
            let mut writes = Vec::new();
            let result = super::assign_cell_value_scalar_in_context(
                Some(&runtime),
                original.clone(),
                &positions,
                &rhs,
                |old, _| writes.push(old.clone()),
            )
            .unwrap();
            assert_eq!(writes, [Value::Num(22.), Value::Num(11.), rhs.clone()]);
            assert_eq!(
                original.data(),
                &[
                    Value::Num(11.),
                    Value::Num(12.),
                    Value::Num(21.),
                    Value::Num(22.)
                ]
            );
            let Value::Cell(result) = result else {
                panic!("Cell output")
            };
            for position in [0, 3] {
                let Value::Tensor(tensor) = &result.data()[position] else {
                    panic!("retained Tensor")
                };
                assert_eq!(tensor.as_f64_slice().unwrap().as_ptr(), pointer);
            }
            assert!(ledger.snapshot().live_claimed_bytes > 0);
            let all = super::resolve_cell_assignment_components_in_context(
                Some(&runtime),
                &original,
                &[ObjectIndexComponent::Colon],
            )
            .unwrap();
            assert_eq!(all.as_slice(), &[1, 2, 3, 4]);
            assert!(super::resolve_cell_assignment_components_in_context(
                Some(&runtime),
                &original,
                &[ObjectIndexComponent::Colon, ObjectIndexComponent::Colon]
            )
            .is_err());
            let denied = ManagedHostEffectLedger::new(1);
            assert!(super::expand_cell_indices_in_context(
                &context(&denied),
                &original,
                &native_indices,
            )
            .is_err());
            assert!(super::resolve_cell_assignment_positions_in_context(
                Some(&context(&denied)),
                &original,
                &native_indices,
            )
            .is_err());
            assert!(super::resolve_cell_assignment_components_in_context(
                Some(&context(&denied)),
                &original,
                &indices
            )
            .is_err());
            assert_eq!(denied.snapshot().live_claimed_bytes, 0);
            assert_eq!(denied.snapshot().live_reserved_bytes, 0);
            let mut denied_writes = 0;
            assert!(super::assign_cell_value_multi_in_context(
                Some(&runtime),
                original.clone(),
                &positions,
                &[],
                |_, _| denied_writes += 1
            )
            .is_err());
            assert_eq!(denied_writes, 0);
        }
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        {
            let original = CellArray::new(Vec::new(), 0, 0).unwrap();
            let result = super::assign_cell_value_scalar_in_context(
                Some(&runtime),
                original,
                &[3],
                &Value::Num(9.),
                |_, _| {},
            )
            .unwrap();
            let Value::Cell(result) = result else {
                panic!("grown Cell")
            };
            assert_eq!(result.shape(), &[1, 3]);
            let (Value::Tensor(left), Value::Tensor(right)) =
                (&result.data()[0], &result.data()[1])
            else {
                panic!("empty fillers")
            };
            assert_eq!(left.shape(), &[0, 0]);
            assert_eq!(left.shape().as_ptr(), right.shape().as_ptr());
        }
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn expand_cell_indices_rejects_fractional_linear_index() {
        let cell = CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).expect("cell");
        let err = expand_cell_indices(&cell, &[Value::Num(1.5)])
            .expect_err("fractional index should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexType"));
    }

    #[test]
    fn expand_cell_indices_rejects_fractional_tensor_indices() {
        let cell = CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).expect("cell");
        let tensor = Tensor::new(vec![1.0, 1.25], vec![1, 2]).expect("tensor");
        let err = expand_cell_indices(&cell, &[Value::Tensor(tensor)])
            .expect_err("fractional tensor index should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexType"));
    }

    #[test]
    fn empty_typed_numeric_brace_assignment_selectors_have_zero_positions() {
        let cell = CellArray::new(vec![Value::Num(10.0)], 1, 1).expect("cell");
        let mut selectors = vec![
            Value::Tensor(Tensor::new(Vec::new(), vec![1, 0]).expect("empty double")),
            Value::Tensor(Tensor::from_f32(Vec::new(), vec![1, 0]).expect("empty single")),
        ];
        for storage in [
            IntegerStorage::I8(Vec::new()),
            IntegerStorage::I16(Vec::new()),
            IntegerStorage::I32(Vec::new()),
            IntegerStorage::I64(Vec::new()),
            IntegerStorage::U8(Vec::new()),
            IntegerStorage::U16(Vec::new()),
            IntegerStorage::U32(Vec::new()),
            IntegerStorage::U64(Vec::new()),
        ] {
            selectors.push(Value::Tensor(
                Tensor::new_integer(storage, vec![1, 0]).expect("empty integer"),
            ));
        }
        for selector in selectors {
            let positions = resolve_cell_assignment_positions(&cell, &[selector])
                .expect("empty numeric selector has zero destinations");
            assert!(positions.is_empty());
            let unchanged =
                super::assign_cell_value_multi(cell.clone(), &positions, &[], |_, _| {
                    panic!("zero-output assignment must not write a cell element")
                })
                .expect("zero-output assignment is a no-op");
            assert_eq!(unchanged, Value::Cell(cell.clone()));
        }
    }

    #[test]
    fn expand_cell_indices_accepts_scalar_tensor_subscripts_for_2d_cells() {
        let cell = CellArray::new(
            vec![
                Value::Num(11.0),
                Value::Num(12.0),
                Value::Num(21.0),
                Value::Num(22.0),
            ],
            2,
            2,
        )
        .expect("2d cell");
        let row = Tensor::new(vec![2.0], vec![1, 1]).expect("row scalar tensor");
        let col = Tensor::new(vec![1.0], vec![1, 1]).expect("col scalar tensor");
        let values = expand_cell_indices(&cell, &[Value::Tensor(row), Value::Tensor(col)])
            .expect("scalar tensor selectors should index 2d cell expansion");
        assert_eq!(values, vec![Value::Num(21.0)]);
    }

    #[test]
    fn expand_cell_indices_reads_scalar_typed_integer_storage_exactly() {
        let cell = CellArray::new(
            vec![
                Value::Num(11.0),
                Value::Num(12.0),
                Value::Num(21.0),
                Value::Num(22.0),
            ],
            2,
            2,
        )
        .expect("2d cell");
        let row =
            Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("row selector");
        let col =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("col selector");

        let values = expand_cell_indices(&cell, &[Value::Tensor(row), Value::Tensor(col)])
            .expect("typed integer selectors should use exact storage");
        assert_eq!(values, vec![Value::Num(21.0)]);
    }

    #[test]
    fn cell_scalar_indices_read_all_integer_classes_exactly() {
        let cell = CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).expect("cell");
        macro_rules! assert_index {
            ($storage:expr) => {{
                let index = Tensor::new_integer($storage, vec![1, 1]).expect("index");
                let values = expand_cell_indices(&cell, &[Value::Tensor(index)])
                    .expect("exact typed scalar index");
                assert_eq!(values, vec![Value::Num(20.0)]);
            }};
        }

        assert_index!(IntegerStorage::I8(vec![2]));
        assert_index!(IntegerStorage::I16(vec![2]));
        assert_index!(IntegerStorage::I32(vec![2]));
        assert_index!(IntegerStorage::I64(vec![2]));
        assert_index!(IntegerStorage::U8(vec![2]));
        assert_index!(IntegerStorage::U16(vec![2]));
        assert_index!(IntegerStorage::U32(vec![2]));
        assert_index!(IntegerStorage::U64(vec![2]));
    }

    #[test]
    fn expand_cell_indices_reads_vector_typed_integer_storage_exactly() {
        let cell = CellArray::new(
            vec![Value::Num(10.0), Value::Num(20.0), Value::Num(30.0)],
            1,
            3,
        )
        .expect("cell");
        let indices =
            Tensor::new_integer(IntegerStorage::U16(vec![3, 1]), vec![1, 2]).expect("indices");

        let values = expand_cell_indices(&cell, &[Value::Tensor(indices)])
            .expect("typed integer vector indices should use exact storage");
        assert_eq!(values, vec![Value::Num(30.0), Value::Num(10.0)]);
    }

    #[test]
    fn expand_cell_indices_accepts_native_single_vector_indices() {
        let cell = CellArray::new(
            vec![Value::Num(10.0), Value::Num(20.0), Value::Num(30.0)],
            1,
            3,
        )
        .expect("cell");
        let indices = Tensor::from_f32(vec![3.0, 1.0], vec![1, 2]).expect("single indices");
        let values = expand_cell_indices(&cell, &[Value::Tensor(indices)])
            .expect("native-single vector indices");
        assert_eq!(values, vec![Value::Num(30.0), Value::Num(10.0)]);
    }

    #[test]
    fn typed_integer_cell_subscripts_use_exact_storage() {
        let cell = CellArray::new(
            vec![Value::Num(10.0), Value::Num(20.0), Value::Num(30.0)],
            1,
            3,
        )
        .expect("cell");

        let scalar = Tensor::new_integer(IntegerStorage::U64(vec![3]), vec![1, 1])
            .expect("wide scalar selector");
        let values =
            expand_cell_indices(&cell, &[Value::Tensor(scalar)]).expect("typed scalar selector");
        assert_eq!(values, vec![Value::Num(30.0)]);

        let vector = Tensor::new_integer(IntegerStorage::I64(vec![3, 1]), vec![1, 2])
            .expect("signed vector selector");
        assert_eq!(
            resolve_cell_assignment_positions(&cell, &[Value::Tensor(vector)])
                .expect("typed vector selector"),
            vec![3, 1]
        );

        let wide = Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
            .expect("wide unsigned selector");
        let err = expand_cell_indices(&cell, &[Value::Tensor(wide)])
            .expect_err("wide uint64 index must remain exact");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexOutOfBounds"));
    }

    #[test]
    fn typed_integer_cell_subscript_bounds_use_exact_storage() {
        let cell = CellArray::new(vec![Value::Num(10.0)], 1, 1).expect("cell");
        let index = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN]), vec![1, 1])
            .expect("signed edge selector");

        let err = expand_cell_indices(&cell, &[Value::Tensor(index)])
            .expect_err("signed edge index must be rejected from exact storage");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexOutOfBounds"));
    }

    #[test]
    fn expand_cell_indices_rejects_nonscalar_tensor_subscripts_for_2d_cells() {
        let cell = CellArray::new(
            vec![
                Value::Num(11.0),
                Value::Num(12.0),
                Value::Num(21.0),
                Value::Num(22.0),
            ],
            2,
            2,
        )
        .expect("2d cell");
        let row = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("non-scalar row tensor");
        let err = expand_cell_indices(&cell, &[Value::Tensor(row), Value::Num(1.0)])
            .expect_err("non-scalar tensor row selector should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexType"));
    }

    #[test]
    fn expand_cell_indices_supports_end_selectors_for_2d_cells() {
        let cell = CellArray::new(
            vec![
                Value::Num(11.0),
                Value::Num(12.0),
                Value::Num(21.0),
                Value::Num(22.0),
            ],
            2,
            2,
        )
        .expect("2d cell");
        let row_end = expand_cell_indices(&cell, &[Value::Num(-0.0), Value::Num(1.0)])
            .expect("row end selector should resolve");
        assert_eq!(row_end, vec![Value::Num(21.0)]);

        let col_end = expand_cell_indices(&cell, &[Value::Num(1.0), Value::Num(-0.0)])
            .expect("col end selector should resolve");
        assert_eq!(col_end, vec![Value::Num(12.0)]);
    }
}
