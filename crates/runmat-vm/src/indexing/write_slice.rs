use crate::indexing::integer_assignment::{
    self, ComplexIntegerAssignmentValue, IntegerAssignmentValue,
};
use crate::indexing::plan::IndexPlan;
use crate::interpreter::errors::mex;
use runmat_accelerate_api::{HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView};
use runmat_builtins::{
    ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType, NumericScalar,
    NumericStorage, SparseTensor, StringArray, Tensor, Value,
};
use runmat_runtime::builtins::common::tensor::{
    self, complex_tensor_element_len, complex_tensor_values_complex64, is_scalar_tensor,
    tensor_element_len, tensor_value_f64,
};
use runmat_runtime::RuntimeError;

fn map_slice_shape_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("ShapeMismatch", &format!("{context}: {err}"))
}

fn map_acceleration_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("AccelerationOperationFailed", &format!("{context}: {err}"))
}

fn is_empty_delete_rhs(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(t)
            if tensor_element_len(t) == 0 || t.rows == 0 || t.cols == 0
    ) || matches!(
        value,
        Value::ComplexTensor(t)
            if complex_tensor_element_len(t) == 0 || t.rows == 0 || t.cols == 0
    ) || matches!(value, Value::OutputList(values) if values.is_empty())
}

pub(crate) fn real_tensor_to_complex(
    tensor: Tensor,
    context: &str,
) -> Result<ComplexTensor, RuntimeError> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| map_slice_shape_error(context, error))?;
    match storage.into_integer_storage() {
        Ok(real) => {
            let imag = real.zeros_like(real.len());
            let storage = IntegerComplexStorage::new(real, imag)
                .expect("same-class real and imaginary integer storage must be valid");
            ComplexTensor::new_integer(storage, shape)
                .map_err(|error| map_slice_shape_error(context, error))
        }
        Err(storage) => ComplexTensor::new(
            storage
                .materialize_f64()
                .into_iter()
                .map(|real| (real, 0.0))
                .collect(),
            shape,
        )
        .map_err(|error| map_slice_shape_error(context, error)),
    }
}

pub(crate) fn deleted_vector_shape(rows: usize, _cols: usize, len: usize) -> Vec<usize> {
    if len == 0 {
        vec![0, 0]
    } else if rows == 1 {
        vec![1, len]
    } else {
        vec![len, 1]
    }
}

fn sorted_unique_positions_desc(
    plan: &IndexPlan,
    total: usize,
) -> Result<Vec<usize>, RuntimeError> {
    let mut positions = Vec::with_capacity(plan.indices.len());
    for &idx in &plan.indices {
        let pos = idx as usize;
        if pos >= total {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        positions.push(pos);
    }
    positions.sort_unstable();
    positions.dedup();
    positions.reverse();
    Ok(positions)
}

fn delete_integer_storage_positions(
    storage: IntegerStorage,
    positions: &[usize],
) -> IntegerStorage {
    macro_rules! delete_positions {
        ($values:expr, $variant:ident) => {{
            let mut values = $values;
            for &position in positions {
                values.remove(position);
            }
            IntegerStorage::$variant(values)
        }};
    }

    match storage {
        IntegerStorage::I8(values) => delete_positions!(values, I8),
        IntegerStorage::I16(values) => delete_positions!(values, I16),
        IntegerStorage::I32(values) => delete_positions!(values, I32),
        IntegerStorage::I64(values) => delete_positions!(values, I64),
        IntegerStorage::U8(values) => delete_positions!(values, U8),
        IntegerStorage::U16(values) => delete_positions!(values, U16),
        IntegerStorage::U32(values) => delete_positions!(values, U32),
        IntegerStorage::U64(values) => delete_positions!(values, U64),
    }
}

pub(crate) fn delete_integer_complex_storage_positions(
    storage: IntegerComplexStorage,
    positions: &[usize],
) -> IntegerComplexStorage {
    IntegerComplexStorage::new(
        delete_integer_storage_positions(storage.real, positions),
        delete_integer_storage_positions(storage.imag, positions),
    )
    .expect("paired integer complex storage must retain matching classes and lengths")
}

fn scalar_integer_value(value: &Value) -> Result<IntegerAssignmentValue, RuntimeError> {
    match value {
        Value::Int(value) => Ok(IntegerAssignmentValue::Exact(value.clone())),
        Value::Num(value) => Ok(IntegerAssignmentValue::Float(*value)),
        Value::Bool(value) => Ok(IntegerAssignmentValue::Float(if *value {
            1.0
        } else {
            0.0
        })),
        Value::Tensor(tensor) if is_scalar_tensor(tensor) => {
            let value = tensor
                .numeric_value_at(0)
                .expect("scalar tensor must contain one numeric value");
            Ok(match value.into_int_value() {
                Some(value) => IntegerAssignmentValue::Exact(value),
                None => IntegerAssignmentValue::Float(value.materialize_f64()),
            })
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(IntegerAssignmentValue::Float(if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }))
        }
        _ => Err(mex(
            "InvalidSliceAssignmentRhs",
            "rhs must be numeric or logical",
        )),
    }
}

pub enum ComplexRhsView {
    Scalar((f64, f64)),
    Tensor {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

pub fn build_complex_rhs_view(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<ComplexRhsView, RuntimeError> {
    match rhs {
        Value::Complex(re, im) => Ok(ComplexRhsView::Scalar((*re, *im))),
        Value::Num(n) => Ok(ComplexRhsView::Scalar((*n, 0.0))),
        Value::ComplexTensor(rt) => {
            let dims = selection_lengths.len();
            let mut shape = rt.shape.clone();
            if shape.len() < dims {
                shape.resize(dims, 1);
            }
            if shape.len() > dims {
                if shape.iter().skip(dims).any(|&s| s != 1) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
                shape.truncate(dims);
            }
            for d in 0..dims {
                let out_len = selection_lengths[d];
                let rhs_len = shape[d];
                if !(rhs_len == 1 || rhs_len == out_len) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut rstrides = vec![0usize; dims];
            let mut racc = 1usize;
            for d in 0..dims {
                rstrides[d] = racc;
                racc *= shape[d];
            }
            Ok(ComplexRhsView::Tensor {
                data: complex_tensor_values_complex64(rt)
                    .into_iter()
                    .map(|value| (value.re, value.im))
                    .collect(),
                shape,
                strides: rstrides,
            })
        }
        _ => Err(mex(
            "InvalidSliceAssignmentRhs",
            "rhs must be numeric or tensor",
        )),
    }
}

pub fn scatter_complex_with_plan(
    t: &mut ComplexTensor,
    plan: &IndexPlan,
    rhs_view: &ComplexRhsView,
) -> Result<(), RuntimeError> {
    let dims = plan.dims;
    let mut idx = vec![0usize; dims];
    if plan.indices.is_empty() {
        return Ok(());
    }
    let selection_lengths = if plan.selection_lengths.is_empty() {
        plan.output_shape.clone()
    } else {
        plan.selection_lengths.clone()
    };
    loop {
        let mut rlin = 0usize;
        match rhs_view {
            ComplexRhsView::Scalar(val) => {
                let lin_pos = {
                    let mut p = 0usize;
                    let mut mul = 1usize;
                    for d in 0..dims {
                        p += idx[d] * mul;
                        mul *= selection_lengths[d].max(1);
                    }
                    p
                };
                let dst = plan.indices[lin_pos] as usize;
                t.data[dst] = *val;
            }
            ComplexRhsView::Tensor {
                data,
                shape,
                strides,
            } => {
                for d in 0..dims {
                    let rhs_len = shape[d];
                    let pos = if rhs_len == 1 { 0 } else { idx[d] };
                    rlin += pos * strides[d];
                }
                let lin_pos = {
                    let mut p = 0usize;
                    let mut mul = 1usize;
                    for d in 0..dims {
                        p += idx[d] * mul;
                        mul *= selection_lengths[d].max(1);
                    }
                    p
                };
                let dst = plan.indices[lin_pos] as usize;
                t.data[dst] = data[rlin];
            }
        }
        let mut d = 0usize;
        while d < dims {
            idx[d] += 1;
            if idx[d] < selection_lengths[d].max(1) {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == dims {
            break;
        }
    }
    Ok(())
}

pub enum StringRhsView {
    Scalar(String),
    Tensor {
        data: Vec<String>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

pub fn build_string_rhs_view(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<StringRhsView, RuntimeError> {
    let scalar = match rhs {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) => Some(ca.to_string()),
        _ => None,
    };
    if let Some(s) = scalar {
        return Ok(StringRhsView::Scalar(s));
    }
    if let Value::StringArray(rt) = rhs {
        let dims = selection_lengths.len();
        let mut shape = rt.shape.clone();
        if dims == 1 && shape.iter().filter(|&&dim| dim != 1).count() <= 1 {
            shape = vec![rt.data.len()];
        } else if shape.len() < dims {
            shape.resize(dims, 1);
        }
        if shape.len() > dims {
            if shape.iter().skip(dims).any(|&s| s != 1) {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            shape.truncate(dims);
        }
        for d in 0..dims {
            let out_len = selection_lengths[d];
            let rhs_len = shape[d];
            if !(rhs_len == 1 || rhs_len == out_len) {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
        }
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &len| acc.checked_mul(len));
        if expected != Some(rt.data.len()) {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
        }
        let mut rstrides = vec![0usize; dims];
        let mut racc = 1usize;
        for d in 0..dims {
            rstrides[d] = racc;
            racc *= shape[d];
        }
        return Ok(StringRhsView::Tensor {
            data: rt.data.clone(),
            shape,
            strides: rstrides,
        });
    }
    if let Value::Cell(cell) = rhs {
        let dims = selection_lengths.len();
        let mut data = Vec::with_capacity(cell.data.len());
        for handle in &cell.data {
            let value = handle;
            match value {
                Value::String(text) => data.push(text.clone()),
                Value::CharArray(chars) => data.push(chars.to_string()),
                Value::StringArray(strings) if strings.data.len() == 1 => {
                    data.push(strings.data[0].clone())
                }
                other => {
                    return Err(mex(
                        "InvalidSliceAssignmentRhs",
                        &format!(
                            "rhs cell elements must be string scalars or character vectors, got {other:?}"
                        ),
                    ))
                }
            }
        }
        let mut shape = cell.shape.clone();
        if dims == 1 && shape.iter().filter(|&&dim| dim != 1).count() <= 1 {
            shape = vec![cell.data.len()];
        } else if shape.len() < dims {
            shape.resize(dims, 1);
        }
        if shape.len() > dims {
            if shape.iter().skip(dims).any(|&s| s != 1) {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            shape.truncate(dims);
        }
        for d in 0..dims {
            let out_len = selection_lengths[d];
            let rhs_len = shape[d];
            if !(rhs_len == 1 || rhs_len == out_len) {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
        }
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &len| acc.checked_mul(len));
        if expected != Some(data.len()) {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
        }
        let mut rstrides = vec![0usize; dims];
        let mut racc = 1usize;
        for d in 0..dims {
            rstrides[d] = racc;
            racc *= shape[d];
        }
        return Ok(StringRhsView::Tensor {
            data,
            shape,
            strides: rstrides,
        });
    }
    Err(mex(
        "InvalidSliceAssignmentRhs",
        "rhs must be string, string array, or cellstr",
    ))
}

pub fn scatter_string_with_plan(
    sa: &mut StringArray,
    plan: &IndexPlan,
    rhs_view: &StringRhsView,
) -> Result<(), RuntimeError> {
    let dims = plan.dims;
    let mut idx = vec![0usize; dims];
    if plan.indices.is_empty() {
        return Ok(());
    }
    let selection_lengths = if plan.selection_lengths.is_empty() {
        plan.output_shape.clone()
    } else {
        plan.selection_lengths.clone()
    };
    loop {
        match rhs_view {
            StringRhsView::Scalar(val) => {
                let lin_pos = {
                    let mut p = 0usize;
                    let mut mul = 1usize;
                    for d in 0..dims {
                        p += idx[d] * mul;
                        mul *= selection_lengths[d].max(1);
                    }
                    p
                };
                let dst = plan.indices[lin_pos] as usize;
                sa.data[dst] = val.clone();
            }
            StringRhsView::Tensor {
                data,
                shape,
                strides,
            } => {
                let mut rlin = 0usize;
                for d in 0..dims {
                    let rhs_len = shape[d];
                    let pos = if rhs_len == 1 { 0 } else { idx[d] };
                    rlin += pos * strides[d];
                }
                let lin_pos = {
                    let mut p = 0usize;
                    let mut mul = 1usize;
                    for d in 0..dims {
                        p += idx[d] * mul;
                        mul *= selection_lengths[d].max(1);
                    }
                    p
                };
                let dst = plan.indices[lin_pos] as usize;
                sa.data[dst] = data[rlin].clone();
            }
        }
        let mut d = 0usize;
        while d < dims {
            idx[d] += 1;
            if idx[d] < selection_lengths[d].max(1) {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == dims {
            break;
        }
    }
    Ok(())
}

pub async fn materialize_rhs_real_for_plan(
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<Vec<f64>, RuntimeError> {
    if plan.dims == 1 {
        let count = plan.selection_lengths.first().copied().unwrap_or(0);
        materialize_rhs_linear_real(rhs, count).await
    } else {
        materialize_rhs_nd_real(rhs, &plan.selection_lengths).await
    }
}

pub(crate) async fn materialize_integer_rhs_for_plan(
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<Vec<IntegerAssignmentValue>, RuntimeError> {
    match rhs {
        Value::Int(value) => Ok(vec![
            IntegerAssignmentValue::Exact(value.clone());
            plan.indices.len()
        ]),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            materialize_integer_tensor_rhs_for_plan(tensor, plan)
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_some() =>
        {
            let tensor = download_integer_tensor(handle).await?;
            materialize_integer_tensor_rhs_for_plan(&tensor, plan)
        }
        Value::OutputList(values) => {
            if values.len() == plan.indices.len() {
                return values.iter().map(scalar_integer_value).collect();
            }
            if values.len() == 1 {
                return Ok(vec![scalar_integer_value(&values[0])?; plan.indices.len()]);
            }
            Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
        }
        _ => materialize_rhs_real_for_plan(rhs, plan)
            .await
            .map(|values| {
                values
                    .into_iter()
                    .map(IntegerAssignmentValue::Float)
                    .collect()
            }),
    }
}

fn materialize_integer_tensor_rhs_for_plan(
    tensor: &Tensor,
    plan: &IndexPlan,
) -> Result<Vec<IntegerAssignmentValue>, RuntimeError> {
    let values = integer_assignment::values(
        tensor
            .integer_storage()
            .expect("integer RHS must retain exact storage"),
    );
    integer_gpu_rhs_indices_for_plan(&tensor.shape, plan)?
        .into_iter()
        .map(|index| {
            values
                .get(index as usize)
                .cloned()
                .map(IntegerAssignmentValue::Exact)
                .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))
        })
        .collect()
}

async fn materialize_complex_integer_rhs_for_plan(
    rhs: &Value,
    plan: &IndexPlan,
) -> Result<Vec<ComplexIntegerAssignmentValue>, RuntimeError> {
    match rhs {
        Value::Complex(real, imag) => Ok(vec![
            ComplexIntegerAssignmentValue {
                real: IntegerAssignmentValue::Float(*real),
                imag: IntegerAssignmentValue::Float(*imag),
            };
            plan.indices.len()
        ]),
        Value::ComplexTensor(tensor) => {
            let values = if let Some(storage) = &tensor.integer_data {
                (0..storage.len())
                    .map(|index| ComplexIntegerAssignmentValue {
                        real: IntegerAssignmentValue::Exact(
                            storage
                                .real
                                .value_at(index)
                                .expect("typed complex storage length was validated"),
                        ),
                        imag: IntegerAssignmentValue::Exact(
                            storage
                                .imag
                                .value_at(index)
                                .expect("typed complex storage length was validated"),
                        ),
                    })
                    .collect()
            } else {
                tensor
                    .data
                    .iter()
                    .map(|&(real, imag)| ComplexIntegerAssignmentValue {
                        real: IntegerAssignmentValue::Float(real),
                        imag: IntegerAssignmentValue::Float(imag),
                    })
                    .collect()
            };
            materialize_complex_integer_values_for_plan(values, &tensor.shape, plan)
        }
        _ => materialize_integer_rhs_for_plan(rhs, plan)
            .await
            .map(|values| {
                values
                    .into_iter()
                    .map(|real| ComplexIntegerAssignmentValue {
                        real,
                        imag: IntegerAssignmentValue::Float(0.0),
                    })
                    .collect()
            }),
    }
}

fn materialize_complex_integer_values_for_plan(
    values: Vec<ComplexIntegerAssignmentValue>,
    rhs_shape: &[usize],
    plan: &IndexPlan,
) -> Result<Vec<ComplexIntegerAssignmentValue>, RuntimeError> {
    if plan.dims == 1 {
        if values.len() == plan.indices.len() {
            return Ok(values);
        }
        if values.len() == 1 {
            return Ok(vec![values[0].clone(); plan.indices.len()]);
        }
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }

    let dims = plan.selection_lengths.len();
    let mut shape = rhs_shape.to_vec();
    if shape.len() < dims {
        shape.resize(dims, 1);
    }
    if shape.len() > dims {
        if shape.iter().skip(dims).any(|&dimension| dimension != 1) {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
        }
        shape.truncate(dims);
    }
    for (&rhs_len, &selection_len) in shape.iter().zip(&plan.selection_lengths) {
        if rhs_len != 1 && rhs_len != selection_len {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
        }
    }
    let expected = shape
        .iter()
        .copied()
        .fold(1usize, |acc, length| acc.saturating_mul(length.max(1)));
    if values.len() != expected {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    let mut strides = vec![1usize; dims];
    for dimension in 1..dims {
        strides[dimension] = strides[dimension - 1] * shape[dimension - 1].max(1);
    }
    let mut output = Vec::with_capacity(plan.indices.len());
    let mut coordinates = vec![0usize; dims];
    for _ in 0..plan.indices.len() {
        let mut rhs_index = 0usize;
        for dimension in 0..dims {
            let coordinate = if shape[dimension] == 1 {
                0
            } else {
                coordinates[dimension]
            };
            rhs_index += coordinate * strides[dimension];
        }
        output.push(values[rhs_index].clone());
        for (dimension, coordinate) in coordinates.iter_mut().enumerate() {
            *coordinate += 1;
            if *coordinate < plan.selection_lengths[dimension].max(1) {
                break;
            }
            *coordinate = 0;
        }
    }
    Ok(output)
}

pub fn scatter_real_with_plan(
    storage: &mut NumericStorage,
    plan: &IndexPlan,
    rhs_values: &[f64],
) -> Result<(), RuntimeError> {
    if rhs_values.len() != plan.indices.len() {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    for (&dst, &value) in plan.indices.iter().zip(rhs_values.iter()) {
        let value = match storage.numeric_dtype() {
            NumericDType::F64 => NumericScalar::F64(value),
            NumericDType::F32 => NumericScalar::F32(value as f32),
            _ => unreachable!("real floating scatter requires floating storage"),
        };
        storage
            .set_value(dst as usize, value)
            .map_err(|error| map_slice_shape_error("slice assign", error))?;
    }
    Ok(())
}

/// Assigns an index-plan selection into a host-resident CSC sparse matrix.
/// Sparse storage owns the batched merge; this module owns MATLAB selector and
/// RHS broadcasting semantics shared with dense tensor assignment.
pub async fn assign_sparse_with_plan(
    sparse: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if is_empty_delete_rhs(rhs) {
        return Err(mex(
            "SparseAssignmentUnsupported",
            "Sparse indexed deletion is not yet supported",
        ));
    }
    let target_rows = plan.base_shape.first().copied().unwrap_or(sparse.rows);
    let target_cols = plan.base_shape.get(1).copied().unwrap_or(sparse.cols);
    let sparse = if target_rows != sparse.rows || target_cols != sparse.cols {
        sparse
            .with_expanded_shape(target_rows, target_cols)
            .map_err(|error| map_slice_shape_error("sparse slice expansion", error))?
    } else {
        sparse
    };
    if plan.indices.is_empty() {
        return Ok(Value::SparseTensor(sparse));
    }
    let updated = if let Some(storage) = sparse.integer_storage() {
        let rhs_values = materialize_integer_rhs_for_plan(rhs, plan).await?;
        let updates = plan
            .indices
            .iter()
            .zip(rhs_values.iter())
            .map(|(&index, value)| {
                (
                    index as usize,
                    integer_assignment::scalar_value(storage, value),
                )
            })
            .collect::<Vec<_>>();
        sparse
            .with_updated_integer_linear_values(&updates)
            .map_err(|error| map_slice_shape_error("sparse slice assign", error))?
    } else {
        let rhs_values = materialize_rhs_real_for_plan(rhs, plan).await?;
        let updates = plan
            .indices
            .iter()
            .zip(rhs_values)
            .map(|(&index, value)| (index as usize, value))
            .collect::<Vec<_>>();
        sparse
            .with_updated_linear_values(&updates)
            .map_err(|error| map_slice_shape_error("sparse slice assign", error))?
    };
    Ok(Value::SparseTensor(updated))
}

enum SparseDeletionAxis {
    Rows(Vec<usize>),
    Columns(Vec<usize>),
    All,
}

fn sparse_deletion_axis(
    sparse: &SparseTensor,
    plan: &IndexPlan,
) -> Result<SparseDeletionAxis, RuntimeError> {
    if plan.indices.is_empty() {
        return Ok(SparseDeletionAxis::Rows(Vec::new()));
    }
    if plan.dims == 1 {
        let indices = plan.indices.iter().map(|&index| index as usize).collect();
        if sparse.rows == 1 {
            return Ok(SparseDeletionAxis::Columns(indices));
        }
        if sparse.cols == 1 {
            return Ok(SparseDeletionAxis::Rows(indices));
        }
        return Err(mex(
            "UnsupportedDeletion",
            "Linear sparse deletion is only supported for vectors",
        ));
    }
    if plan.dims != 2 {
        return Err(mex(
            "UnsupportedDeletion",
            "Sparse deletion currently supports vectors and complete matrix rows or columns",
        ));
    }
    let selected_rows = plan.selection_lengths.first().copied().unwrap_or(0);
    let selected_cols = plan.selection_lengths.get(1).copied().unwrap_or(0);
    if selected_rows == sparse.rows && selected_cols == sparse.cols {
        return Ok(SparseDeletionAxis::All);
    }
    if selected_rows == sparse.rows {
        let columns = plan
            .indices
            .chunks(sparse.rows)
            .map(|chunk| chunk[0] as usize / sparse.rows)
            .collect();
        return Ok(SparseDeletionAxis::Columns(columns));
    }
    if selected_cols == sparse.cols {
        let rows = plan
            .indices
            .iter()
            .take(selected_rows)
            .map(|index| *index as usize % sparse.rows)
            .collect();
        return Ok(SparseDeletionAxis::Rows(rows));
    }
    Err(mex(
        "UnsupportedDeletion",
        "Sparse deletion requires selecting complete rows or columns",
    ))
}

pub fn delete_sparse_with_plan(
    sparse: SparseTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if !is_empty_delete_rhs(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    let updated = match sparse_deletion_axis(&sparse, plan)? {
        SparseDeletionAxis::Rows(rows) => sparse.with_deleted_rows(&rows),
        SparseDeletionAxis::Columns(columns) => sparse.with_deleted_columns(&columns),
        SparseDeletionAxis::All => {
            let rows = (0..sparse.rows).collect::<Vec<_>>();
            let columns = (0..sparse.cols).collect::<Vec<_>>();
            sparse
                .with_deleted_rows(&rows)
                .and_then(|sparse| sparse.with_deleted_columns(&columns))
        }
    }
    .map_err(|error| map_slice_shape_error("sparse deletion", error))?;
    Ok(Value::SparseTensor(updated))
}

pub async fn assign_tensor_with_plan(
    t: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        return Ok(Value::Tensor(t));
    }
    if matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        let tensor = real_tensor_to_complex(t, "slice complex promotion")?;
        return assign_complex_with_plan(tensor, plan, rhs).await;
    }
    let shape = t.shape.clone();
    let storage = t
        .into_numeric_storage()
        .map_err(|error| map_slice_shape_error("slice assign", error))?;
    let storage = match storage.into_integer_storage() {
        Ok(mut storage) => {
            let rhs_values = materialize_integer_rhs_for_plan(rhs, plan).await?;
            integer_assignment::scatter(&mut storage, plan, &rhs_values)?;
            NumericStorage::from_integer_storage(storage)
        }
        Err(mut storage) => {
            let rhs_values = materialize_rhs_real_for_plan(rhs, plan).await?;
            scatter_real_with_plan(&mut storage, plan, &rhs_values)?;
            storage
        }
    };
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(|error| map_slice_shape_error("slice assign", error))
}

pub async fn assign_complex_with_plan(
    mut tensor: ComplexTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        return Ok(Value::ComplexTensor(tensor));
    }
    if tensor.integer_data.is_some() {
        let rhs_values = materialize_complex_integer_rhs_for_plan(rhs, plan).await?;
        let storage = tensor
            .integer_data
            .take()
            .expect("typed complex tensor must retain exact storage");
        let real_values: Vec<IntegerAssignmentValue> =
            rhs_values.iter().map(|value| value.real.clone()).collect();
        let imag_values: Vec<IntegerAssignmentValue> =
            rhs_values.iter().map(|value| value.imag.clone()).collect();
        let mut real = storage.real;
        let mut imag = storage.imag;
        integer_assignment::scatter(&mut real, plan, &real_values)?;
        integer_assignment::scatter(&mut imag, plan, &imag_values)?;
        return IntegerComplexStorage::new(real, imag)
            .and_then(|storage| ComplexTensor::new_integer(storage, tensor.shape))
            .map(Value::ComplexTensor)
            .map_err(|error| map_slice_shape_error("typed complex slice assign", error));
    }
    let rhs_view = build_complex_rhs_view(rhs, &plan.selection_lengths)?;
    scatter_complex_with_plan(&mut tensor, plan, &rhs_view)?;
    Ok(Value::ComplexTensor(tensor))
}

pub fn delete_tensor_with_plan(
    t: Tensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if !is_empty_delete_rhs(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    if plan.indices.is_empty() {
        return Ok(Value::Tensor(t));
    }
    if !(t.rows == 1 || t.cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    let positions = sorted_unique_positions_desc(plan, tensor_element_len(&t))?;
    let rows = t.rows;
    let cols = t.cols;
    let mut storage = t
        .into_numeric_storage()
        .map_err(|error| map_slice_shape_error("slice deletion", error))?;
    storage
        .remove_positions(&positions)
        .map_err(|error| map_slice_shape_error("slice deletion", error))?;
    let shape = deleted_vector_shape(rows, cols, storage.len());
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(|error| map_slice_shape_error("slice deletion", error))
}

pub fn delete_complex_with_plan(
    mut t: ComplexTensor,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if !is_empty_delete_rhs(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    if plan.indices.is_empty() {
        return Ok(Value::ComplexTensor(t));
    }
    if !(t.rows == 1 || t.cols == 1) {
        return Err(mex(
            "UnsupportedDeletion",
            "Linear deletion is only supported for vectors",
        ));
    }
    let positions = sorted_unique_positions_desc(plan, complex_tensor_element_len(&t))?;
    if let Some(storage) = t.integer_data.take() {
        let storage = delete_integer_complex_storage_positions(storage, &positions);
        let shape = deleted_vector_shape(t.rows, t.cols, storage.len());
        return ComplexTensor::new_integer(storage, shape)
            .map(Value::ComplexTensor)
            .map_err(|error| map_slice_shape_error("complex slice deletion", error));
    }
    for pos in positions {
        t.data.remove(pos);
    }
    let shape = deleted_vector_shape(t.rows, t.cols, t.data.len());
    t.rows = shape.first().copied().unwrap_or(0);
    t.cols = shape.get(1).copied().unwrap_or(0);
    t.shape = shape;
    Ok(Value::ComplexTensor(t))
}

pub async fn assign_gpu_slice_with_plan(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        return Ok(Value::GpuTensor(handle.clone()));
    }
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    if runmat_accelerate_api::handle_integer_type(handle).is_some() {
        if let Value::GpuTensor(rhs_handle) = rhs {
            if runmat_accelerate_api::handle_integer_type(rhs_handle)
                == runmat_accelerate_api::handle_integer_type(handle)
            {
                if let Ok(rhs_indices) = integer_gpu_rhs_indices_for_plan(&rhs_handle.shape, plan) {
                    let reuses_rhs = rhs_indices
                        .iter()
                        .enumerate()
                        .all(|(index, &rhs_index)| rhs_index as usize == index);
                    let values = if reuses_rhs {
                        rhs_handle.clone()
                    } else {
                        provider
                            .gather_linear(rhs_handle, &rhs_indices, &plan.output_shape)
                            .map_err(|e| {
                                map_acceleration_error(
                                    "expand exact integer gpuArray assignment rhs",
                                    e,
                                )
                            })?
                    };
                    let result = provider
                        .scatter_linear(handle, &plan.indices, &values)
                        .map_err(|e| {
                            map_acceleration_error("exact integer gpuArray slice assignment", e)
                        });
                    if !reuses_rhs {
                        let _ = provider.free(&values);
                    }
                    result?;
                    return Ok(Value::GpuTensor(handle.clone()));
                }
            }
        }
        let tensor = download_integer_tensor(handle).await?;
        let Value::Tensor(updated) = assign_tensor_with_plan(tensor, plan, rhs).await? else {
            unreachable!("real integer slice assignment must produce a real tensor")
        };
        return upload_tensor_to_gpu(&updated);
    }
    if let Value::GpuTensor(vh) = rhs {
        let rows = plan.base_shape.first().copied().unwrap_or(1);
        let cols = plan.base_shape.get(1).copied().unwrap_or(1);
        if let Some(col) = plan.properties.full_column {
            if col < cols {
                let v_rows = match vh.shape.len() {
                    1 | 2 => vh.shape[0],
                    _ => 0,
                };
                if v_rows == rows {
                    if let Ok(new_h) = provider.scatter_column(handle, col, vh) {
                        return Ok(Value::GpuTensor(new_h));
                    }
                }
            }
        }
        if let Some(row) = plan.properties.full_row {
            if row < rows {
                let v_cols = match vh.shape.len() {
                    1 => vh.shape[0],
                    2 => vh.shape[1],
                    _ => 0,
                };
                if v_cols == cols {
                    if let Ok(new_h) = provider.scatter_row(handle, row, vh) {
                        return Ok(Value::GpuTensor(new_h));
                    }
                }
            }
        }
    }
    let rhs_values = materialize_rhs_real_for_plan(rhs, plan).await?;
    let value_shape = vec![rhs_values.len().max(1), 1];
    let upload_result = if rhs_values.is_empty() {
        provider.zeros(&[0, 1])
    } else {
        provider.upload(&runmat_accelerate_api::HostTensorView {
            data: &rhs_values,
            shape: &value_shape,
        })
    };
    if let Ok(values_handle) = upload_result {
        if provider
            .scatter_linear(handle, &plan.indices, &values_handle)
            .is_ok()
        {
            return Ok(Value::GpuTensor(handle.clone()));
        }
    }

    let host = provider
        .download(handle)
        .await
        .map_err(|e| map_acceleration_error("gather for slice assign", e))?;
    let t =
        Tensor::new(host.data, host.shape).map_err(|e| map_slice_shape_error("slice assign", e))?;
    let shape = t.shape.clone();
    let mut storage = t
        .into_numeric_storage()
        .map_err(|error| map_slice_shape_error("slice assign", error))?;
    scatter_real_with_plan(&mut storage, plan, &rhs_values)?;
    let t = Tensor::from_numeric_storage(storage, shape)
        .map_err(|error| map_slice_shape_error("slice assign", error))?;
    upload_tensor_to_gpu(&t)
}

fn integer_gpu_rhs_indices_for_plan(
    rhs_shape: &[usize],
    plan: &IndexPlan,
) -> Result<Vec<u32>, RuntimeError> {
    let rhs_len = rhs_shape.iter().try_fold(1usize, |len, dimension| {
        len.checked_mul(*dimension)
            .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))
    })?;
    if plan.dims == 1 {
        if rhs_len == plan.indices.len() {
            return (0..rhs_len)
                .map(|index| {
                    u32::try_from(index).map_err(|_| {
                        mex(
                            "AccelerationOperationFailed",
                            "GPU rhs exceeds indexing limits",
                        )
                    })
                })
                .collect();
        }
        if rhs_len == 1 {
            return Ok(vec![0; plan.indices.len()]);
        }
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }

    let dims = plan.selection_lengths.len();
    let mut shape = rhs_shape.to_vec();
    if shape.len() < dims {
        shape.resize(dims, 1);
    }
    if shape.len() > dims {
        if shape.iter().skip(dims).any(|&dimension| dimension != 1) {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
        }
        shape.truncate(dims);
    }
    for (&rhs_len, &selection_len) in shape.iter().zip(&plan.selection_lengths) {
        if rhs_len != 1 && rhs_len != selection_len {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
        }
    }
    let expected = shape
        .iter()
        .try_fold(1usize, |len, dimension| {
            len.checked_mul((*dimension).max(1))
        })
        .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))?;
    if rhs_len != expected {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    let mut strides = vec![1usize; dims];
    for dimension in 1..dims {
        strides[dimension] = strides[dimension - 1]
            .checked_mul(shape[dimension - 1].max(1))
            .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))?;
    }
    let mut output = Vec::with_capacity(plan.indices.len());
    let mut coordinates = vec![0usize; dims];
    for _ in 0..plan.indices.len() {
        let mut rhs_index = 0usize;
        for dimension in 0..dims {
            let coordinate = if shape[dimension] == 1 {
                0
            } else {
                coordinates[dimension]
            };
            rhs_index = rhs_index
                .checked_add(
                    coordinate
                        .checked_mul(strides[dimension])
                        .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))?,
                )
                .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))?;
        }
        output.push(u32::try_from(rhs_index).map_err(|_| {
            mex(
                "AccelerationOperationFailed",
                "GPU rhs exceeds indexing limits",
            )
        })?);
        for (dimension, coordinate) in coordinates.iter_mut().enumerate().take(dims) {
            *coordinate += 1;
            if *coordinate < plan.selection_lengths[dimension].max(1) {
                break;
            }
            *coordinate = 0;
        }
    }
    Ok(output)
}

pub async fn delete_gpu_slice_with_plan(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if !is_empty_delete_rhs(rhs) {
        return Err(mex(
            "DeletionRequiresEmptyRhs",
            "Indexed deletion requires empty RHS",
        ));
    }
    if plan.indices.is_empty() {
        return Ok(Value::GpuTensor(handle.clone()));
    }
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    if runmat_accelerate_api::handle_integer_type(handle).is_some() {
        let tensor = download_integer_tensor(handle).await?;
        let Value::Tensor(updated) = delete_tensor_with_plan(tensor, plan, rhs)? else {
            unreachable!("integer slice deletion must produce a real tensor")
        };
        return upload_tensor_to_gpu(&updated);
    }
    let host = provider
        .download(handle)
        .await
        .map_err(|e| map_acceleration_error("gather for slice deletion", e))?;
    let t = Tensor::new(host.data, host.shape)
        .map_err(|e| map_slice_shape_error("slice deletion", e))?;
    let Value::Tensor(updated) = delete_tensor_with_plan(t, plan, rhs)? else {
        unreachable!()
    };
    upload_tensor_to_gpu(&updated)
}

pub async fn materialize_rhs_linear_real(
    rhs: &Value,
    count: usize,
) -> Result<Vec<f64>, RuntimeError> {
    let host_rhs = runmat_runtime::dispatcher::gather_if_needed_async(rhs).await?;
    match host_rhs {
        Value::Num(n) => Ok(vec![n; count]),
        Value::Int(int_val) => Ok(vec![int_val.to_f64(); count]),
        Value::Bool(b) => Ok(vec![if b { 1.0 } else { 0.0 }; count]),
        Value::Tensor(t) => {
            let len = tensor_element_len(&t);
            if len == count {
                Ok(tensor::tensor_into_values_f64(t))
            } else if len == 1 {
                Ok(vec![tensor_value_f64(&t, 0); count])
            } else {
                Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == count {
                Ok(la
                    .data
                    .into_iter()
                    .map(|b| if b != 0 { 1.0 } else { 0.0 })
                    .collect())
            } else if la.data.len() == 1 {
                let val = if la.data[0] != 0 { 1.0 } else { 0.0 };
                Ok(vec![val; count])
            } else {
                Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
            }
        }
        Value::OutputList(values) => materialize_output_list_real(&values, count),
        other => Err(mex(
            "InvalidSliceAssignmentRhs",
            &format!("slice assign: unsupported RHS type {:?}", other),
        )),
    }
}

pub async fn materialize_rhs_nd_real(
    rhs: &Value,
    selection_lengths: &[usize],
) -> Result<Vec<f64>, RuntimeError> {
    let rhs_host = runmat_runtime::dispatcher::gather_if_needed_async(rhs).await?;
    enum RhsView {
        Scalar(f64),
        Tensor {
            data: Vec<f64>,
            shape: Vec<usize>,
            strides: Vec<usize>,
        },
    }
    let view = match rhs_host {
        Value::Num(n) => RhsView::Scalar(n),
        Value::Int(iv) => RhsView::Scalar(iv.to_f64()),
        Value::Bool(b) => RhsView::Scalar(if b { 1.0 } else { 0.0 }),
        Value::Tensor(t) => {
            let mut shape = t.shape.clone();
            if shape.len() < selection_lengths.len() {
                shape.resize(selection_lengths.len(), 1);
            }
            if shape.len() > selection_lengths.len() {
                if shape.iter().skip(selection_lengths.len()).any(|&s| s != 1) {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
                shape.truncate(selection_lengths.len());
            }
            for (dim_len, &sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if *dim_len != 1 && *dim_len != sel_len {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![1usize; selection_lengths.len()];
            for d in 1..selection_lengths.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            let data = tensor::tensor_into_values_f64(t);
            if data.len()
                != shape
                    .iter()
                    .copied()
                    .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)))
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            RhsView::Tensor {
                data,
                shape,
                strides,
            }
        }
        Value::LogicalArray(la) => {
            if la.shape.len() > selection_lengths.len()
                && la
                    .shape
                    .iter()
                    .skip(selection_lengths.len())
                    .any(|&s| s != 1)
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            let mut shape = la.shape.clone();
            if shape.len() < selection_lengths.len() {
                shape.resize(selection_lengths.len(), 1);
            } else {
                shape.truncate(selection_lengths.len());
            }
            for (dim_len, &sel_len) in shape.iter().zip(selection_lengths.iter()) {
                if *dim_len != 1 && *dim_len != sel_len {
                    return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
                }
            }
            let mut strides = vec![1usize; selection_lengths.len()];
            for d in 1..selection_lengths.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            if la.data.len()
                != shape
                    .iter()
                    .copied()
                    .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)))
            {
                return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
            }
            let data: Vec<f64> = la
                .data
                .into_iter()
                .map(|b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            RhsView::Tensor {
                data,
                shape,
                strides,
            }
        }
        Value::OutputList(values) => {
            let count = selection_lengths
                .iter()
                .copied()
                .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)));
            let data = materialize_output_list_real(&values, count)?;
            let shape = if selection_lengths.is_empty() {
                vec![1]
            } else {
                selection_lengths.to_vec()
            };
            let mut strides = vec![1usize; shape.len()];
            for d in 1..shape.len() {
                strides[d] = strides[d - 1] * shape[d - 1].max(1);
            }
            RhsView::Tensor {
                data,
                shape,
                strides,
            }
        }
        other => {
            return Err(mex(
                "InvalidSliceAssignmentRhs",
                &format!("slice assign: unsupported RHS type {:?}", other),
            ))
        }
    };

    let total = selection_lengths
        .iter()
        .copied()
        .fold(1usize, |acc, len| acc.saturating_mul(len.max(1)));
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; selection_lengths.len()];
    if selection_lengths.is_empty() {
        return Ok(out);
    }
    loop {
        match &view {
            RhsView::Scalar(val) => out.push(*val),
            RhsView::Tensor {
                data,
                shape,
                strides,
            } => {
                let mut rlin = 0usize;
                for d in 0..idx.len() {
                    let rhs_len = shape[d];
                    let pos = if rhs_len == 1 { 0 } else { idx[d] };
                    rlin += pos * strides[d];
                }
                out.push(data.get(rlin).copied().unwrap_or(0.0));
            }
        }
        let mut d = 0usize;
        while d < idx.len() {
            idx[d] += 1;
            if idx[d] < selection_lengths[d].max(1) {
                break;
            }
            idx[d] = 0;
            d += 1;
        }
        if d == idx.len() {
            break;
        }
    }
    Ok(out)
}

fn materialize_output_list_real(values: &[Value], count: usize) -> Result<Vec<f64>, RuntimeError> {
    if values.len() == count {
        values.iter().map(value_to_real_scalar).collect()
    } else if values.len() == 1 {
        let value = value_to_real_scalar(&values[0])?;
        Ok(vec![value; count])
    } else {
        Err(mex("ShapeMismatch", "shape mismatch for slice assign"))
    }
}

fn value_to_real_scalar(value: &Value) -> Result<f64, RuntimeError> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(int_val) => Ok(int_val.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if is_scalar_tensor(t) => Ok(tensor_value_f64(t, 0)),
        _ => f64::try_from(value).map_err(Into::into),
    }
}

pub fn upload_tensor_to_gpu(t: &Tensor) -> Result<Value, RuntimeError> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    let new_h = if let Some(storage) = t.integer_storage() {
        let view = integer_tensor_view(storage, &t.shape);
        provider
            .upload_integer(&view)
            .map_err(|e| map_acceleration_error("exact integer reupload after slice assign", e))?
    } else {
        let data = t.materialize_f64();
        let view = runmat_accelerate_api::HostTensorView {
            data: &data,
            shape: &t.shape,
        };
        provider
            .upload(&view)
            .map_err(|e| map_acceleration_error("reupload after slice assign", e))?
    };
    Ok(Value::GpuTensor(new_h))
}

pub(crate) async fn download_integer_tensor(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> Result<Tensor, RuntimeError> {
    let provider = runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
        mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered for integer gpuArray",
        )
    })?;
    let integer = provider
        .download_integer(handle)
        .await
        .map_err(|e| map_acceleration_error("exact integer gather for assignment", e))?;
    let storage = match integer.data {
        HostIntegerDataOwned::I8(values) => IntegerStorage::I8(values),
        HostIntegerDataOwned::I16(values) => IntegerStorage::I16(values),
        HostIntegerDataOwned::I32(values) => IntegerStorage::I32(values),
        HostIntegerDataOwned::I64(values) => IntegerStorage::I64(values),
        HostIntegerDataOwned::U8(values) => IntegerStorage::U8(values),
        HostIntegerDataOwned::U16(values) => IntegerStorage::U16(values),
        HostIntegerDataOwned::U32(values) => IntegerStorage::U32(values),
        HostIntegerDataOwned::U64(values) => IntegerStorage::U64(values),
    };
    Tensor::new_integer(storage, integer.shape)
        .map_err(|e| map_slice_shape_error("integer gpuArray assignment gather", e))
}

fn integer_tensor_view<'a>(
    storage: &'a IntegerStorage,
    shape: &'a [usize],
) -> HostIntegerTensorView<'a> {
    let data = match storage {
        IntegerStorage::I8(values) => HostIntegerDataView::I8(values),
        IntegerStorage::I16(values) => HostIntegerDataView::I16(values),
        IntegerStorage::I32(values) => HostIntegerDataView::I32(values),
        IntegerStorage::I64(values) => HostIntegerDataView::I64(values),
        IntegerStorage::U8(values) => HostIntegerDataView::U8(values),
        IntegerStorage::U16(values) => HostIntegerDataView::U16(values),
        IntegerStorage::U32(values) => HostIntegerDataView::U32(values),
        IntegerStorage::U64(values) => HostIntegerDataView::U64(values),
    };
    HostIntegerTensorView { data, shape }
}

#[cfg(test)]
mod tests {
    use super::{
        assign_complex_with_plan, assign_sparse_with_plan, assign_tensor_with_plan,
        build_complex_rhs_view, build_string_rhs_view, delete_complex_with_plan,
        delete_gpu_slice_with_plan, delete_tensor_with_plan, integer_gpu_rhs_indices_for_plan,
        map_acceleration_error, materialize_rhs_linear_real, materialize_rhs_nd_real,
        ComplexRhsView,
    };
    use crate::indexing::plan::IndexPlan;
    use futures::executor::block_on;
    use runmat_builtins::{
        CellArray, ComplexTensor, IntegerComplexStorage, IntegerStorage, NumericDType,
        NumericStorage, SparseTensor, StringArray, Tensor, Value,
    };

    #[test]
    fn integer_plan_assignment_preserves_exact_uint64_rhs() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3]), vec![1, 3]).expect("tensor");
        let mut rhs_tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 9]), vec![1, 2]).expect("rhs");
        rhs_tensor.data.clear();
        let rhs = Value::Tensor(rhs_tensor);
        let plan = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 3]);
        let result = block_on(assign_tensor_with_plan(tensor, &plan, &rhs)).expect("assign");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 9, 3]))
        );
    }

    #[test]
    fn native_single_plan_assignment_and_deletion_preserve_class() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("single tensor");
        let plan = IndexPlan::new(vec![0, 2], vec![1, 2], vec![2], 1, vec![1, 3]);
        let Value::Tensor(updated) = block_on(assign_tensor_with_plan(
            tensor,
            &plan,
            &Value::Num(1.234_567_890_123),
        ))
        .expect("single slice assignment") else {
            panic!("expected tensor");
        };
        assert_eq!(updated.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            updated.clone().into_numeric_storage(),
            Ok(NumericStorage::F32(vec![
                1.234_567_890_123_f64 as f32,
                2.0,
                1.234_567_890_123_f64 as f32,
            ]))
        );

        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let delete_plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let Value::Tensor(deleted) =
            delete_tensor_with_plan(updated, &delete_plan, &empty).expect("single slice deletion")
        else {
            panic!("expected tensor");
        };
        assert_eq!(deleted.numeric_dtype(), NumericDType::F32);
        assert_eq!(deleted.shape, vec![1, 2]);
        assert_eq!(
            deleted.into_numeric_storage(),
            Ok(NumericStorage::F32(vec![
                1.234_567_890_123_f64 as f32,
                1.234_567_890_123_f64 as f32,
            ]))
        );
    }

    #[test]
    fn typed_slice_deletion_uses_exact_storage_when_f64_mirrors_are_unavailable() {
        let mut tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX, 3]), vec![1, 3])
            .expect("tensor");
        tensor.data.clear();
        let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty rhs"));

        let Value::Tensor(output) =
            delete_tensor_with_plan(tensor, &plan, &empty).expect("typed uint64 deletion")
        else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, 3]))
        );

        let mut complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MIN, -2, i64::MAX]),
                IntegerStorage::I64(vec![1, 2, 3]),
            )
            .expect("integer complex storage"),
            vec![1, 3],
        )
        .expect("complex tensor");
        complex.data = vec![(f64::NAN, f64::NAN)];

        let Value::ComplexTensor(output) = delete_complex_with_plan(complex, &plan, &empty)
            .expect("typed signed complex deletion")
        else {
            panic!("expected complex tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(
            output.integer_data,
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                    IntegerStorage::I64(vec![1, 3]),
                )
                .expect("integer complex storage")
            )
        );
    }

    #[test]
    fn integer_plan_assignment_preserves_all_typed_rhs_classes_with_poisoned_mirrors() {
        macro_rules! assert_assignment {
            ($storage:ident, $value:expr) => {{
                let tensor = Tensor::new_integer(
                    IntegerStorage::$storage(vec![Default::default(), Default::default()]),
                    vec![1, 2],
                )
                .expect("destination tensor");
                let mut rhs =
                    Tensor::new_integer(IntegerStorage::$storage(vec![$value]), vec![1, 1])
                        .expect("rhs tensor");
                rhs.data = vec![f64::NAN];
                let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 2]);

                let Value::Tensor(output) =
                    block_on(assign_tensor_with_plan(tensor, &plan, &Value::Tensor(rhs)))
                        .expect("assignment")
                else {
                    panic!("expected tensor");
                };
                assert_eq!(
                    output.integer_storage(),
                    Some(&IntegerStorage::$storage(vec![Default::default(), $value]))
                );
            }};
        }

        assert_assignment!(I8, i8::MIN);
        assert_assignment!(I16, i16::MIN);
        assert_assignment!(I32, i32::MIN);
        assert_assignment!(I64, i64::MIN);
        assert_assignment!(U8, u8::MAX);
        assert_assignment!(U16, u16::MAX);
        assert_assignment!(U32, u32::MAX);
        assert_assignment!(U64, u64::MAX);
    }

    #[test]
    fn real_plan_assignment_reads_typed_integer_rhs_without_mirror() {
        let tensor = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).expect("tensor");
        let mut rhs_tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![4, 9]), vec![1, 2]).expect("rhs");
        rhs_tensor.data.clear();
        let rhs = Value::Tensor(rhs_tensor);
        let plan = IndexPlan::new(vec![0, 2], vec![1, 2], vec![2], 1, vec![1, 3]);
        let result = block_on(assign_tensor_with_plan(tensor, &plan, &rhs)).expect("assign");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(output.data, vec![4.0, 0.0, 9.0]);
    }

    #[test]
    fn real_scalar_expansion_reads_all_typed_integer_classes_without_f64_mirrors() {
        macro_rules! assert_scalar_expansion {
            ($storage:expr, $expected:expr) => {{
                let mut tensor = Tensor::new_integer($storage, vec![1, 1]).expect("scalar rhs");
                tensor.data.clear();
                let rhs = Value::Tensor(tensor);

                assert_eq!(
                    block_on(materialize_rhs_linear_real(&rhs, 3)).expect("linear expansion"),
                    vec![$expected; 3]
                );
                assert_eq!(
                    block_on(materialize_rhs_nd_real(&rhs, &[2, 2])).expect("nd expansion"),
                    vec![$expected; 4]
                );
            }};
        }

        assert_scalar_expansion!(IntegerStorage::I8(vec![-8]), -8.0);
        assert_scalar_expansion!(IntegerStorage::I16(vec![-16]), -16.0);
        assert_scalar_expansion!(IntegerStorage::I32(vec![-32]), -32.0);
        assert_scalar_expansion!(IntegerStorage::I64(vec![-64]), -64.0);
        assert_scalar_expansion!(IntegerStorage::U8(vec![8]), 8.0);
        assert_scalar_expansion!(IntegerStorage::U16(vec![16]), 16.0);
        assert_scalar_expansion!(IntegerStorage::U32(vec![32]), 32.0);
        assert_scalar_expansion!(IntegerStorage::U64(vec![64]), 64.0);
    }

    #[test]
    fn integer_plan_assignment_broadcasts_exact_tensor_rhs() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I8(vec![0; 4]), vec![2, 2]).expect("tensor");
        let rhs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I8(vec![5, 6]), vec![1, 2]).expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let result = block_on(assign_tensor_with_plan(tensor, &plan, &rhs)).expect("assign");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![5, 5, 6, 6]))
        );
    }

    #[test]
    fn integer_gpu_rhs_indices_cover_exact_scalar_and_nd_broadcast_forms() {
        let linear = IndexPlan::new(vec![5, 1, 3], vec![1, 3], vec![3], 1, vec![2, 3]);
        assert_eq!(
            integer_gpu_rhs_indices_for_plan(&[1, 3], &linear).expect("linear indices"),
            vec![0, 1, 2]
        );
        assert_eq!(
            integer_gpu_rhs_indices_for_plan(&[1, 1], &linear).expect("scalar indices"),
            vec![0, 0, 0]
        );

        let nd = IndexPlan::new(
            vec![0, 1, 2, 3, 4, 5],
            vec![2, 3],
            vec![2, 3],
            2,
            vec![2, 3],
        );
        assert_eq!(
            integer_gpu_rhs_indices_for_plan(&[1, 3], &nd).expect("nd broadcast indices"),
            vec![0, 0, 1, 1, 2, 2]
        );
    }

    #[test]
    fn integer_gpu_rhs_indices_reject_incompatible_shape() {
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let error = integer_gpu_rhs_indices_for_plan(&[3, 1], &plan)
            .expect_err("incompatible GPU rhs shape must fail");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn integer_plan_assignment_converts_float_rhs_with_saturation() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I8(vec![0, 0]), vec![1, 2]).expect("tensor");
        let plan = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 2]);
        let result =
            block_on(assign_tensor_with_plan(tensor, &plan, &Value::Num(300.5))).expect("assign");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![i8::MAX, i8::MAX]))
        );
    }

    #[test]
    fn real_integer_plan_assignment_promotes_to_complex_exact_storage_for_every_class() {
        macro_rules! assert_promotion {
            ($storage:ident, $values:expr, $real:expr, $imag:expr) => {{
                let tensor = Tensor::new_integer(IntegerStorage::$storage($values), vec![2, 2])
                    .expect("tensor");
                let mut rhs_tensor = ComplexTensor::new_integer(
                    IntegerComplexStorage::new(
                        IntegerStorage::$storage(vec![$real]),
                        IntegerStorage::$storage(vec![$imag]),
                    )
                    .expect("rhs storage"),
                    vec![1, 1],
                )
                .expect("rhs");
                rhs_tensor.data.clear();
                let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
                let result = block_on(assign_tensor_with_plan(
                    tensor,
                    &plan,
                    &Value::ComplexTensor(rhs_tensor),
                ))
                .expect("assign");
                let Value::ComplexTensor(output) = result else {
                    panic!("integer tensor should promote to complex tensor");
                };
                assert_eq!(
                    output
                        .integer_data
                        .as_ref()
                        .map(|storage| (&storage.real, &storage.imag)),
                    Some((
                        &IntegerStorage::$storage(vec![$real; 4]),
                        &IntegerStorage::$storage(vec![$imag; 4]),
                    ))
                );
            }};
        }

        assert_promotion!(I8, vec![i8::MIN, -1, 0, i8::MAX], i8::MIN, i8::MAX);
        assert_promotion!(I16, vec![i16::MIN, -1, 0, i16::MAX], i16::MIN, i16::MAX);
        assert_promotion!(I32, vec![i32::MIN, -1, 0, i32::MAX], i32::MIN, i32::MAX);
        assert_promotion!(I64, vec![i64::MIN, -1, 0, i64::MAX], i64::MIN, i64::MAX);
        assert_promotion!(U8, vec![0, 1, 2, u8::MAX], 1, u8::MAX);
        assert_promotion!(U16, vec![0, 1, 2, u16::MAX], 1, u16::MAX);
        assert_promotion!(U32, vec![0, 1, 2, u32::MAX], 1, u32::MAX);
        assert_promotion!(
            U64,
            vec![0, 1, 2, u64::MAX],
            9_223_372_036_854_775_809,
            u64::MAX
        );
    }

    #[test]
    fn real_integer_complex_promotion_rejects_shape_mismatch() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![0; 4]), vec![2, 2]).expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I16(vec![1, 2, 3]),
                    IntegerStorage::I16(vec![-1, -2, -3]),
                )
                .expect("rhs storage"),
                vec![1, 3],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);

        let error = block_on(assign_tensor_with_plan(tensor, &plan, &rhs))
            .expect_err("incompatible complex RHS shape must fail");
        assert_eq!(error.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn real_integer_complex_promotion_preserves_exact_non_2d_scalar_expansion() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3, 4]), vec![2, 1, 2])
            .expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![9_223_372_036_854_775_809]),
                )
                .expect("rhs storage"),
                vec![1, 1, 1],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(
            vec![0, 1, 2, 3],
            vec![2, 1, 2],
            vec![2, 1, 2],
            3,
            vec![2, 1, 2],
        );

        let result = block_on(assign_tensor_with_plan(tensor, &plan, &rhs)).expect("assign");
        let Value::ComplexTensor(output) = result else {
            panic!("integer tensor should promote to complex tensor");
        };
        assert_eq!(output.shape, vec![2, 1, 2]);
        assert_eq!(
            output
                .integer_data
                .as_ref()
                .map(|storage| (&storage.real, &storage.imag)),
            Some((
                &IntegerStorage::U64(vec![u64::MAX; 4]),
                &IntegerStorage::U64(vec![9_223_372_036_854_775_809; 4]),
            ))
        );
    }

    #[test]
    fn typed_complex_integer_plan_assignment_preserves_exact_components_and_broadcasts() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1, 2, 3, 4]),
                IntegerStorage::U64(vec![10, 20, 30, 40]),
            )
            .expect("storage"),
            vec![2, 2],
        )
        .expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]),
                    IntegerStorage::U64(vec![7, 8]),
                )
                .expect("storage"),
                vec![1, 2],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let result = block_on(assign_complex_with_plan(tensor, &plan, &rhs)).expect("assign");

        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            output
                .integer_data
                .as_ref()
                .map(|storage| (&storage.real, &storage.imag)),
            Some((
                &IntegerStorage::U64(vec![
                    u64::MAX,
                    u64::MAX,
                    9_223_372_036_854_775_808,
                    9_223_372_036_854_775_808
                ]),
                &IntegerStorage::U64(vec![7, 7, 8, 8]),
            ))
        );
    }

    #[test]
    fn typed_complex_integer_plan_assignment_rejects_shape_mismatch() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I8(vec![0; 4]),
                IntegerStorage::I8(vec![0; 4]),
            )
            .expect("storage"),
            vec![2, 2],
        )
        .expect("tensor");
        let rhs = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I8(vec![1, 2, 3]),
                    IntegerStorage::I8(vec![4, 5, 6]),
                )
                .expect("storage"),
                vec![1, 3],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let err = block_on(assign_complex_with_plan(tensor, &plan, &rhs))
            .expect_err("shape mismatch should fail");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn typed_complex_integer_plan_assignment_converts_float_components_independently() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I8(vec![0; 2]),
                IntegerStorage::I8(vec![0; 2]),
            )
            .expect("storage"),
            vec![1, 2],
        )
        .expect("tensor");
        let plan = IndexPlan::new(vec![0, 1], vec![1, 2], vec![2], 1, vec![1, 2]);
        let result = block_on(assign_complex_with_plan(
            tensor,
            &plan,
            &Value::Complex(300.5, -300.5),
        ))
        .expect("assign");

        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(
            output
                .integer_data
                .as_ref()
                .map(|storage| (&storage.real, &storage.imag)),
            Some((
                &IntegerStorage::I8(vec![i8::MAX, i8::MAX]),
                &IntegerStorage::I8(vec![i8::MIN, i8::MIN]),
            ))
        );
    }

    #[test]
    fn sparse_integer_plan_assignment_preserves_exact_values_and_last_write_wins() {
        let sparse =
            SparseTensor::new_integer(2, 2, vec![0, 0, 0], vec![], IntegerStorage::U64(vec![]))
                .expect("sparse");
        let rhs = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808, 3, u64::MAX]),
                vec![1, 4],
            )
            .expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 3, 1, 3], vec![1, 4], vec![4], 1, vec![2, 2]);
        let result = block_on(assign_sparse_with_plan(sparse, &plan, &rhs)).expect("assign");

        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert_eq!(output.col_ptrs, vec![0, 2, 3]);
        assert_eq!(output.row_indices, vec![0, 1, 1]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, 3, u64::MAX]))
        );
    }

    #[test]
    fn sparse_integer_plan_assignment_broadcasts_exact_rhs_across_rows() {
        let sparse =
            SparseTensor::new_integer(2, 2, vec![0, 0, 0], vec![], IntegerStorage::I8(vec![]))
                .expect("sparse");
        let rhs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I8(vec![5, 6]), vec![1, 2]).expect("rhs"),
        );
        let plan = IndexPlan::new(vec![0, 1, 2, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let result = block_on(assign_sparse_with_plan(sparse, &plan, &rhs)).expect("assign");

        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert_eq!(output.col_ptrs, vec![0, 2, 4]);
        assert_eq!(output.row_indices, vec![0, 1, 0, 1]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![5, 5, 6, 6]))
        );
    }

    #[test]
    fn sparse_real_plan_assignment_elides_zero_values() {
        let sparse = SparseTensor::new(2, 2, vec![0, 2, 3], vec![0, 1, 0], vec![1.0, 2.0, 3.0])
            .expect("sparse");
        let plan = IndexPlan::new(vec![0, 1], vec![2, 1], vec![2, 1], 2, vec![2, 2]);
        let result =
            block_on(assign_sparse_with_plan(sparse, &plan, &Value::Num(0.0))).expect("assign");

        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert_eq!(output.col_ptrs, vec![0, 0, 1]);
        assert_eq!(output.row_indices, vec![0]);
        assert_eq!(output.values, vec![3.0]);
    }

    #[test]
    fn integer_plan_deletion_preserves_exact_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![1, i64::MAX, 3]), vec![1, 3])
            .expect("tensor");
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let result = delete_tensor_with_plan(tensor, &plan, &empty).expect("delete");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I64(vec![1, 3]))
        );
        assert_eq!(output.shape, vec![1, 2]);
    }

    #[test]
    fn gpu_integer_plan_deletion_preserves_wide_uint64_storage() {
        runmat_accelerate_api::set_thread_provider(None);
        runmat_accelerate_api::clear_provider();
        runmat_accelerate::simple_provider::register_inprocess_provider();
        let provider = runmat_accelerate_api::provider().expect("test provider");
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let source = provider
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data: runmat_accelerate_api::HostIntegerDataView::U64(&[
                    1,
                    9_223_372_036_854_775_808,
                    u64::MAX,
                ]),
                shape: &[1, 3],
            })
            .expect("upload integer source");
        let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty"));
        let plan = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);

        let Value::GpuTensor(updated) =
            block_on(delete_gpu_slice_with_plan(&source, &plan, &empty)).expect("delete")
        else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&updated),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let host = block_on(provider.download_integer(&updated)).expect("download updated");
        assert_eq!(host.shape, vec![1, 2]);
        assert_eq!(
            host.data,
            runmat_accelerate_api::HostIntegerDataOwned::U64(vec![1, u64::MAX])
        );
    }

    #[test]
    fn complex_rhs_view_shape_mismatch_reports_identifier() {
        let rhs = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], vec![1, 3])
                .expect("complex tensor"),
        );
        let err = match build_complex_rhs_view(&rhs, &[2, 2]) {
            Ok(_) => panic!("shape mismatch should fail"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn complex_rhs_view_invalid_rhs_type_reports_identifier() {
        let rhs = Value::String("x".to_string());
        let err = match build_complex_rhs_view(&rhs, &[1]) {
            Ok(_) => panic!("non-numeric rhs should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:InvalidSliceAssignmentRhs"));
    }

    #[test]
    fn complex_rhs_view_reads_all_typed_integer_classes_without_f64_mirrors() {
        macro_rules! assert_typed_rhs {
            ($storage:ident, $real:expr, $imag:expr) => {{
                let mut rhs = ComplexTensor::new_integer(
                    IntegerComplexStorage::new(
                        IntegerStorage::$storage(vec![$real]),
                        IntegerStorage::$storage(vec![$imag]),
                    )
                    .expect("typed integer components"),
                    vec![1, 1],
                )
                .expect("typed complex rhs");
                rhs.data = vec![(f64::NAN, f64::NAN)];

                let ComplexRhsView::Tensor { data, .. } =
                    build_complex_rhs_view(&Value::ComplexTensor(rhs), &[1])
                        .expect("typed complex rhs view")
                else {
                    panic!("complex tensor rhs must produce a tensor view");
                };
                assert_eq!(data, vec![($real as f64, $imag as f64)]);
            }};
        }

        assert_typed_rhs!(I8, i8::MIN, i8::MAX);
        assert_typed_rhs!(I16, i16::MIN, i16::MAX);
        assert_typed_rhs!(I32, i32::MIN, i32::MAX);
        assert_typed_rhs!(I64, i64::MIN, i64::MAX);
        assert_typed_rhs!(U8, u8::MIN, u8::MAX);
        assert_typed_rhs!(U16, u16::MIN, u16::MAX);
        assert_typed_rhs!(U32, u32::MIN, u32::MAX);
        assert_typed_rhs!(U64, u64::MIN, u64::MAX);
    }

    #[test]
    fn string_rhs_view_shape_mismatch_reports_identifier() {
        let rhs = Value::StringArray(
            StringArray::new(
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                vec![1, 3],
            )
            .expect("string array"),
        );
        let err = match build_string_rhs_view(&rhs, &[2, 2]) {
            Ok(_) => panic!("shape mismatch should fail"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn string_cell_rhs_view_rejects_shape_data_length_mismatch() {
        let data = ["a", "b", "c"]
            .into_iter()
            .map(|text| Value::String(text.to_string()))
            .collect();
        let rhs = Value::Cell(CellArray {
            data,
            shape: vec![2, 2],
            rows: 2,
            cols: 2,
        });
        let err = match build_string_rhs_view(&rhs, &[2, 2]) {
            Ok(_) => panic!("cell shape/data mismatch should fail"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn string_rhs_view_invalid_rhs_type_reports_identifier() {
        let rhs = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("tensor"));
        let err = match build_string_rhs_view(&rhs, &[1]) {
            Ok(_) => panic!("non-string rhs should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), Some("RunMat:InvalidSliceAssignmentRhs"));
    }

    #[test]
    fn slice_acceleration_error_mapping_reports_identifier() {
        let err = map_acceleration_error("slice assign", "provider failed");
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
    }
}
