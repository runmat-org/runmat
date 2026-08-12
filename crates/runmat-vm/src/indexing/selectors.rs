use crate::indexing::plan::total_len_from_shape;
use crate::interpreter::errors::mex;
use runmat_builtins::{IntValue, LogicalArray, NumericScalar, Tensor, Value};
use runmat_runtime::{
    builtins::common::{shape::is_scalar_shape, tensor},
    dispatcher::gather_if_needed_async,
    RuntimeError,
};

pub type VmResult<T> = Result<T, RuntimeError>;

fn checked_total_len_from_shape(shape: &[usize]) -> VmResult<usize> {
    if runmat_runtime::builtins::common::shape::is_scalar_shape(shape) {
        return Ok(1);
    }
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))
    })
}

fn map_index_gather_error(err: impl std::fmt::Display) -> RuntimeError {
    mex(
        "AccelerationOperationFailed",
        &format!("index gather: {err}"),
    )
}

fn selector_mask_has_dim(mask: u32, dim: usize) -> bool {
    dim < u32::BITS as usize && (mask & (1u32 << dim)) != 0
}

#[derive(Clone, Debug)]
pub enum SliceSelector {
    Colon,
    Scalar(usize),
    Indices(Vec<usize>),
    LinearIndices {
        values: Vec<usize>,
        output_shape: Vec<usize>,
    },
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IndexScalar {
    Signed(i64),
    Unsigned(u64),
}

impl IndexScalar {
    pub(crate) fn from_int(value: &IntValue) -> Self {
        match value {
            IntValue::I8(value) => Self::Signed(i64::from(*value)),
            IntValue::I16(value) => Self::Signed(i64::from(*value)),
            IntValue::I32(value) => Self::Signed(i64::from(*value)),
            IntValue::I64(value) => Self::Signed(*value),
            IntValue::U8(value) => Self::Unsigned(u64::from(*value)),
            IntValue::U16(value) => Self::Unsigned(u64::from(*value)),
            IntValue::U32(value) => Self::Unsigned(u64::from(*value)),
            IntValue::U64(value) => Self::Unsigned(*value),
        }
    }

    pub(crate) fn positive_usize(self) -> Option<usize> {
        match self {
            Self::Signed(value) if value >= 1 => usize::try_from(value).ok(),
            Self::Unsigned(value) if value >= 1 => usize::try_from(value).ok(),
            _ => None,
        }
    }

    pub(crate) fn is_below_one(self) -> bool {
        match self {
            Self::Signed(value) => value < 1,
            Self::Unsigned(value) => value < 1,
        }
    }
}

fn index_scalar_from_host_value(value: &Value) -> Option<IndexScalar> {
    match value {
        Value::Num(n) => exact_index_from_f64(*n).map(IndexScalar::Signed),
        Value::Int(int_val) => Some(IndexScalar::from_int(int_val)),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) && is_scalar_shape(&t.shape) => {
            if let Some(storage) = t.integer_storage() {
                storage
                    .value_at(0)
                    .map(|value| IndexScalar::from_int(&value))
            } else {
                exact_index_from_f64(tensor::tensor_value_f64(t, 0)).map(IndexScalar::Signed)
            }
        }
        _ => None,
    }
}

pub(crate) async fn index_scalar_from_value(value: &Value) -> VmResult<Option<IndexScalar>> {
    if let Value::GpuTensor(handle) = value {
        let total = total_len_from_shape(&handle.shape);
        if total != 1 {
            return Ok(None);
        }
        let gathered = gather_if_needed_async(value).await?;
        return Ok(index_scalar_from_host_value(&gathered));
    }
    Ok(index_scalar_from_host_value(value))
}

fn checked_positive_index(scalar: IndexScalar, upper_bound: Option<usize>) -> VmResult<usize> {
    let Some(index) = scalar.positive_usize() else {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    };
    if upper_bound.is_some_and(|bound| index > bound) {
        return Err(mex("IndexOutOfBounds", "Index out of bounds"));
    }
    Ok(index)
}

pub(crate) fn numeric_tensor_indices(
    tensor: &Tensor,
    upper_bound: Option<usize>,
) -> VmResult<Vec<usize>> {
    let mut indices = Vec::with_capacity(tensor.len());
    for index in 0..tensor.len() {
        let scalar = match tensor
            .numeric_value_at(index)
            .expect("numeric tensor storage length must match its shape")
        {
            NumericScalar::F64(value) => exact_index_from_f64(value).map(IndexScalar::Signed),
            NumericScalar::F32(value) => {
                exact_index_from_f64(f64::from(value)).map(IndexScalar::Signed)
            }
            value => value
                .into_int_value()
                .map(|value| IndexScalar::from_int(&value)),
        }
        .ok_or_else(|| {
            mex(
                "UnsupportedIndexType",
                "Index values must be positive integers or logical values",
            )
        })?;
        indices.push(checked_positive_index(scalar, upper_bound)?);
    }
    Ok(indices)
}

pub async fn materialize_index_value(value: &Value) -> VmResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        return gather_if_needed_async(value)
            .await
            .map_err(map_index_gather_error);
    }
    Ok(value.clone())
}

pub(crate) fn logical_indices_linear(
    array: &LogicalArray,
    total_len: usize,
) -> VmResult<Vec<usize>> {
    let mut indices = Vec::new();
    for (index, &selected) in array.data.iter().enumerate() {
        if selected == 0 {
            continue;
        }
        if index >= total_len {
            return Err(mex(
                "IndexOutOfBounds",
                "Logical index exceeds array bounds",
            ));
        }
        indices.push(index + 1);
    }
    Ok(indices)
}

pub async fn indices_from_value_linear(value: &Value, total_len: usize) -> VmResult<Vec<usize>> {
    if let Value::Bool(b) = value {
        return Ok(if *b { vec![1] } else { Vec::new() });
    }
    if let Value::LogicalArray(la) = value {
        if la.data.len() == 1 && is_scalar_shape(&la.shape) {
            return Ok(if la.data[0] != 0 { vec![1] } else { Vec::new() });
        }
    }
    if let Some(idx_val) = index_scalar_from_value(value).await? {
        return checked_positive_index(idx_val, Some(total_len)).map(|index| vec![index]);
    }
    let materialized;
    let value = if matches!(value, Value::GpuTensor(_)) {
        materialized = materialize_index_value(value).await?;
        &materialized
    } else {
        value
    };
    match value {
        Value::Tensor(idx_t) => numeric_tensor_indices(idx_t, Some(total_len)),
        Value::LogicalArray(la) => logical_indices_linear(la, total_len),
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for linear indexing",
        )),
    }
}

async fn selector_from_value_dim_with_bounds(
    value: &Value,
    dim_len: usize,
    require_in_bounds: bool,
) -> VmResult<SliceSelector> {
    if let Value::Bool(b) = value {
        if *b {
            return Ok(SliceSelector::Indices(vec![1]));
        }
        return Ok(SliceSelector::Indices(Vec::new()));
    }
    if let Value::LogicalArray(la) = value {
        if la.data.len() == 1 && is_scalar_shape(&la.shape) {
            if la.data[0] != 0 {
                return Ok(SliceSelector::Indices(vec![1]));
            }
            return Ok(SliceSelector::Indices(Vec::new()));
        }
    }
    if let Some(idx_val) = index_scalar_from_value(value).await? {
        let bound = require_in_bounds.then_some(dim_len);
        return checked_positive_index(idx_val, bound).map(SliceSelector::Scalar);
    }
    let materialized;
    let value = if matches!(value, Value::GpuTensor(_)) {
        materialized = materialize_index_value(value).await?;
        &materialized
    } else {
        value
    };
    match value {
        Value::Tensor(idx_t) => numeric_tensor_indices(idx_t, require_in_bounds.then_some(dim_len))
            .map(SliceSelector::Indices),
        Value::LogicalArray(la) => {
            if la.data.len() != dim_len {
                return Err(mex(
                    "IndexShape",
                    "Logical mask length mismatch for dimension",
                ));
            }
            let mut indices = Vec::new();
            for (i, &b) in la.data.iter().enumerate() {
                if b != 0 {
                    indices.push(i + 1);
                }
            }
            Ok(SliceSelector::Indices(indices))
        }
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for slicing",
        )),
    }
}

pub async fn selector_from_value_dim(value: &Value, dim_len: usize) -> VmResult<SliceSelector> {
    selector_from_value_dim_with_bounds(value, dim_len, true).await
}

pub async fn build_slice_selectors(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Vec<SliceSelector>> {
    let mut selectors = Vec::with_capacity(dims);
    if dims == 1 {
        let total_len = checked_total_len_from_shape(base_shape)?;
        if (colon_mask & 1u32) != 0 {
            selectors.push(SliceSelector::Colon);
            return Ok(selectors);
        }
        if (end_mask & 1u32) != 0 {
            selectors.push(SliceSelector::Scalar(total_len.max(1)));
            return Ok(selectors);
        }
        let value = numeric.first().ok_or_else(|| {
            mex(
                "MissingNumericIndex",
                "missing numeric index for linear slice",
            )
        })?;
        let materialized = materialize_index_value(value).await?;
        if let Value::Tensor(idx_t) = &materialized {
            let indices = numeric_tensor_indices(idx_t, Some(total_len))?;
            selectors.push(SliceSelector::LinearIndices {
                values: indices,
                output_shape: idx_t.shape.clone(),
            });
        } else if let Value::LogicalArray(_) = &materialized {
            // MATLAB linear logical indexing always yields a column vector.
            let idxs = indices_from_value_linear(&materialized, total_len).await?;
            selectors.push(SliceSelector::LinearIndices {
                output_shape: vec![idxs.len(), 1],
                values: idxs,
            });
        } else {
            let idxs = indices_from_value_linear(&materialized, total_len).await?;
            selectors.push(SliceSelector::Indices(idxs));
        }
        return Ok(selectors);
    }

    let mut numeric_iter = 0usize;
    for d in 0..dims {
        let is_colon = selector_mask_has_dim(colon_mask, d);
        if is_colon {
            selectors.push(SliceSelector::Colon);
            continue;
        }
        let dim_len = base_shape.get(d).copied().unwrap_or(1);
        let is_end = selector_mask_has_dim(end_mask, d);
        if is_end {
            selectors.push(SliceSelector::Scalar(dim_len));
            continue;
        }
        let value = numeric
            .get(numeric_iter)
            .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index for slice"))?;
        numeric_iter += 1;
        selectors.push(selector_from_value_dim(value, dim_len).await?);
    }
    Ok(selectors)
}

/// Builds selectors for sparse two-subscript assignment. Numeric selectors may
/// grow their addressed dimension; colon and logical selectors remain tied to
/// the existing shape, matching MATLAB indexed-assignment rules.
pub async fn build_sparse_assignment_selectors(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Vec<SliceSelector>> {
    if dims != 2 {
        return build_slice_selectors(dims, colon_mask, end_mask, numeric, base_shape).await;
    }

    let mut selectors = Vec::with_capacity(dims);
    let mut numeric_iter = 0usize;
    for d in 0..dims {
        if selector_mask_has_dim(colon_mask, d) {
            selectors.push(SliceSelector::Colon);
            continue;
        }
        let dim_len = base_shape.get(d).copied().unwrap_or(1);
        if selector_mask_has_dim(end_mask, d) {
            selectors.push(SliceSelector::Scalar(dim_len));
            continue;
        }
        let value = numeric
            .get(numeric_iter)
            .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index for slice"))?;
        numeric_iter += 1;
        match value {
            Value::Bool(_) | Value::LogicalArray(_) => {
                selectors.push(selector_from_value_dim_with_bounds(value, dim_len, true).await?);
            }
            _ => {
                selectors.push(selector_from_value_dim_with_bounds(value, dim_len, false).await?);
            }
        }
    }
    Ok(selectors)
}

pub async fn build_cell_scalar_selectors(raw_indices: &[Value]) -> VmResult<Vec<SliceSelector>> {
    let mut selectors = Vec::with_capacity(raw_indices.len());
    for value in raw_indices {
        let idx_val = index_scalar_from_value(value).await?.ok_or_else(|| {
            mex(
                "ScalarIndexRequired",
                "Cell indexing requires scalar numeric indices",
            )
        })?;
        if idx_val.is_below_one() {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        let idx = idx_val
            .positive_usize()
            .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
        selectors.push(SliceSelector::Scalar(idx));
    }
    Ok(selectors)
}

#[cfg(test)]
mod tests {
    use super::{
        build_cell_scalar_selectors, build_slice_selectors, indices_from_value_linear,
        selector_from_value_dim, SliceSelector,
    };
    use runmat_builtins::{IntValue, IntegerStorage, LogicalArray, Tensor, Value};

    #[test]
    fn selector_from_value_dim_rejects_fractional_numeric_indices() {
        let err =
            futures::executor::block_on(selector_from_value_dim(&Value::Num(2.5), 8)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
    }

    #[test]
    fn linear_indices_reject_fractional_tensor_indices() {
        let value = Value::Tensor(
            Tensor::new(vec![1.0, 2.5], vec![1, 2]).expect("fractional index tensor"),
        );
        let err = futures::executor::block_on(indices_from_value_linear(&value, 8)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
    }

    #[test]
    fn linear_logical_indices_accept_prefix_masks_and_false_overhang() {
        let shorter = Value::LogicalArray(
            LogicalArray::new(vec![0, 1, 1], vec![1, 3]).expect("short logical mask"),
        );
        let indices = futures::executor::block_on(indices_from_value_linear(&shorter, 5))
            .expect("short linear logical mask");
        assert_eq!(indices, vec![2, 3]);

        let false_overhang = Value::LogicalArray(
            LogicalArray::new(vec![1, 0, 0, 0, 0, 0], vec![1, 6]).expect("long logical mask"),
        );
        let indices = futures::executor::block_on(indices_from_value_linear(&false_overhang, 5))
            .expect("false logical overhang");
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn linear_logical_indices_reject_true_positions_beyond_target() {
        let value = Value::LogicalArray(
            LogicalArray::new(vec![0, 0, 0, 1], vec![1, 4]).expect("logical mask"),
        );
        let error = futures::executor::block_on(indices_from_value_linear(&value, 3))
            .expect_err("true logical overhang must reject");
        assert_eq!(error.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn per_dimension_logical_indices_still_require_exact_length() {
        let value =
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical mask"));
        let error = futures::executor::block_on(selector_from_value_dim(&value, 3))
            .expect_err("short per-dimension logical mask must reject");
        assert_eq!(error.identifier(), Some("RunMat:IndexShape"));
    }

    #[test]
    fn build_cell_scalar_selectors_rejects_zero_index() {
        let err = futures::executor::block_on(build_cell_scalar_selectors(&[Value::Num(0.0)]))
            .expect_err("zero cell scalar index should fail");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn build_cell_scalar_selectors_rejects_negative_index() {
        let err = futures::executor::block_on(build_cell_scalar_selectors(&[Value::Num(-2.0)]))
            .expect_err("negative cell scalar index should fail");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn uint64_scalar_indices_do_not_clamp_through_i64() {
        let err = futures::executor::block_on(indices_from_value_linear(
            &Value::Int(IntValue::U64(u64::MAX)),
            8,
        ))
        .expect_err("huge uint64 index should be out of bounds");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));

        let selector =
            futures::executor::block_on(selector_from_value_dim(&Value::Int(IntValue::U64(3)), 8))
                .expect("representable uint64 selector");
        assert!(matches!(selector, SliceSelector::Scalar(3)));
    }

    #[test]
    fn scalar_integer_tensor_indices_use_exact_integer_storage() {
        let exact = (1_u64 << 53) + 1;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![exact]), vec![1, 1])
            .expect("scalar uint64 tensor");
        let err = futures::executor::block_on(indices_from_value_linear(&Value::Tensor(tensor), 8))
            .expect_err("huge scalar integer tensor index should be out of bounds");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn integer_index_vectors_use_exact_storage() {
        let wide = (1_u64 << 53) + 1;
        let indices = Tensor::new_integer(IntegerStorage::U64(vec![2, wide]), vec![1, 2])
            .expect("uint64 index tensor");

        let error =
            futures::executor::block_on(indices_from_value_linear(&Value::Tensor(indices), 3))
                .expect_err("wide exact uint64 vector index must not be ignored");
        assert_eq!(error.identifier(), Some("RunMat:IndexOutOfBounds"));

        let signed = Tensor::new_integer(IntegerStorage::I64(vec![1, i64::MIN]), vec![1, 2])
            .expect("int64 index tensor");
        let error = futures::executor::block_on(selector_from_value_dim(&Value::Tensor(signed), 2))
            .expect_err("signed boundary index must remain out of bounds");
        assert_eq!(error.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn integer_index_vector_selectors_preserve_typed_values() {
        let indices = Tensor::new_integer(IntegerStorage::U16(vec![2, 1]), vec![1, 2])
            .expect("uint16 index tensor");

        let selector =
            futures::executor::block_on(selector_from_value_dim(&Value::Tensor(indices), 2))
                .expect("typed integer vector selector");
        assert!(matches!(selector, SliceSelector::Indices(values) if values == vec![2, 1]));
    }

    #[test]
    fn index_gather_error_maps_to_acceleration_identifier() {
        let err = super::map_index_gather_error("boom");
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
        assert!(err.message().contains("index gather"));
        assert!(err.message().contains("boom"));
    }

    #[test]
    fn build_slice_selectors_supports_dims_beyond_mask_width() {
        let numeric: Vec<Value> = (0..31).map(|v| Value::Num((v + 1) as f64)).collect();
        let base_shape = vec![40usize; 33];
        let selectors = futures::executor::block_on(build_slice_selectors(
            33,
            0b1,
            0b10,
            &numeric,
            &base_shape,
        ))
        .expect("slice selectors for dims beyond mask width");
        assert_eq!(selectors.len(), 33);
        assert!(matches!(selectors[0], SliceSelector::Colon));
        assert!(matches!(selectors[1], SliceSelector::Scalar(40)));
        assert!(matches!(selectors[32], SliceSelector::Scalar(31)));
    }
}
