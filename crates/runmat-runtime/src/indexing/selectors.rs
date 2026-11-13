use super::resident::gather_dispatch_value_in_context as gather_selector_in_context;
use crate::indexing::plan::total_len_from_shape;
use crate::runtime_error::semantic_error as mex;
use crate::{
    builtins::common::{shape::is_scalar_shape, tensor},
    RuntimeError,
};
use runmat_value::{
    ExactArrayStorage, IntValue, LogicalArray, NumericScalar, SharedShape, Tensor, Value,
};
use std::borrow::{Borrow, Cow};

pub type VmResult<T> = Result<T, RuntimeError>;

fn map_index_gather_error(err: impl std::fmt::Display) -> RuntimeError {
    mex(
        "AccelerationOperationFailed",
        format!("index gather: {err}"),
    )
}

fn selector_mask_has_dim(mask: u32, dim: usize) -> bool {
    dim < u32::BITS as usize && (mask & (1u32 << dim)) != 0
}

#[derive(Clone, Debug)]
pub enum SliceSelector {
    Colon,
    Scalar(usize),
    Indices(ExactArrayStorage<usize>),
    LinearIndices {
        values: ExactArrayStorage<usize>,
        output_shape: SharedShape,
    },
}

impl SliceSelector {
    pub(crate) fn selection_len(&self, extent: usize) -> usize {
        match self {
            Self::Colon => extent,
            Self::Scalar(_) => 1,
            Self::Indices(values) | Self::LinearIndices { values, .. } => values.len(),
        }
    }
    pub(crate) fn selection_value(&self, position: usize) -> Option<usize> {
        match self {
            Self::Colon => position.checked_add(1),
            Self::Scalar(value) => (position == 0).then_some(*value),
            Self::Indices(values) | Self::LinearIndices { values, .. } => {
                values.get(position).copied()
            }
        }
    }
    /// Borrow the canonical selector; colon remains an implicit range.
    pub(crate) fn iter_indices(
        &self,
        extent: usize,
    ) -> impl ExactSizeIterator<Item = usize> + Clone + '_ {
        (0..self.selection_len(extent)).map(move |position| {
            self.selection_value(position)
                .expect("bounded selector position")
        })
    }
}

pub async fn build_component_selectors(
    components: &[crate::object::indexing::ObjectIndexComponent],
    base_shape: &[usize],
    assignment: bool,
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    let context = crate::context::legacy::active();
    build_component_selectors_with_context(context.as_ref(), components, base_shape, assignment)
        .await
}

pub async fn build_component_selectors_in_context(
    context: &crate::context::RuntimeContext,
    components: &[crate::object::indexing::ObjectIndexComponent],
    base_shape: &[usize],
    assignment: bool,
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    build_component_selectors_with_context(Some(context), components, base_shape, assignment).await
}

pub(crate) async fn build_component_selectors_with_context(
    context: Option<&crate::context::RuntimeContext>,
    components: &[crate::object::indexing::ObjectIndexComponent],
    base_shape: &[usize],
    assignment: bool,
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    use crate::indexing::plan::{
        admission, effective_index_extent, logical_linear_output_shape_in_context,
        numeric_linear_output_shape_in_context,
    };
    use crate::object::indexing::ObjectIndexComponent;
    let owner = admission::Owner::new(context)?;
    let dims = components.len();
    if dims == 0 {
        return Err(mex("MissingIndex", "At least one index is required"));
    }
    // Verify shape overflow before admitting control or initiating a download.
    for dimension in 0..dims {
        effective_index_extent(base_shape, dims, dimension)?;
    }
    let mut selectors =
        crate::context::host_allocation::prepare_host_array_optional(context, dims)?;
    for (dimension, component) in components.iter().enumerate() {
        owner.check()?;
        let extent = effective_index_extent(base_shape, dims, dimension)?;
        let selector = match component {
            ObjectIndexComponent::Colon => SliceSelector::Colon,
            ObjectIndexComponent::End => SliceSelector::Scalar(extent),
            ObjectIndexComponent::Value(value) if dims == 1 && assignment => {
                let materialized = owner
                    .await_stage(materialize_index_value_in_context(context, value))
                    .await??;
                linear_assignment_selector_from_host(context, materialized.as_ref(), extent)?
            }
            ObjectIndexComponent::Value(value) if dims == 1 && !assignment => {
                let materialized = owner
                    .await_stage(materialize_index_value_in_context(context, value))
                    .await??;
                match materialized.as_ref() {
                    Value::Tensor(tensor) => SliceSelector::LinearIndices {
                        values: numeric_tensor_indices_in_context(context, tensor, Some(extent))?,
                        output_shape: numeric_linear_output_shape_in_context(
                            context,
                            base_shape,
                            tensor.shape(),
                            tensor.len(),
                        )?,
                    },
                    value @ (Value::Bool(_) | Value::LogicalArray(_)) => {
                        let values =
                            indices_from_host_value_linear_in_context(context, value, extent)?;
                        let mask_shape: &[usize] = match value {
                            Value::LogicalArray(mask) => &mask.shape,
                            Value::Bool(_) => &[1, 1],
                            _ => unreachable!(),
                        };
                        let output_shape = logical_linear_output_shape_in_context(
                            context,
                            base_shape,
                            mask_shape,
                            values.len(),
                        )?;
                        SliceSelector::LinearIndices {
                            values,
                            output_shape,
                        }
                    }
                    value => SliceSelector::Indices(indices_from_host_value_linear_in_context(
                        context, value, extent,
                    )?),
                }
            }
            ObjectIndexComponent::Value(value) => {
                let bounded =
                    !assignment || matches!(value, Value::Bool(_) | Value::LogicalArray(_));
                owner
                    .await_stage(selector_from_value_dim_with_bounds(
                        context, value, extent, bounded,
                    ))
                    .await??
            }
        };
        selectors
            .try_push(selector)
            .map_err(|_| admission::geometry())?;
    }
    owner.check()?;
    selectors.try_finish().map_err(|_| admission::geometry())
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
pub enum IndexScalar {
    Signed(i64),
    Unsigned(u64),
}

impl IndexScalar {
    pub fn from_int(value: &IntValue) -> Self {
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

    pub fn positive_usize(self) -> Option<usize> {
        match self {
            Self::Signed(value) if value >= 1 => usize::try_from(value).ok(),
            Self::Unsigned(value) if value >= 1 => usize::try_from(value).ok(),
            _ => None,
        }
    }

    pub fn is_below_one(self) -> bool {
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
        Value::Tensor(t) if tensor::is_scalar_tensor(t) && is_scalar_shape(t.shape()) => {
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

pub async fn index_scalar_from_value(value: &Value) -> VmResult<Option<IndexScalar>> {
    let context = crate::context::legacy::active();
    index_scalar_from_value_in_context(context.as_ref(), value).await
}
pub async fn index_scalar_from_value_in_context(
    context: Option<&crate::context::RuntimeContext>,
    value: &Value,
) -> VmResult<Option<IndexScalar>> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    if let Value::GpuTensor(handle) = value {
        let total = total_len_from_shape(handle.shape());
        if total != 1 {
            return Ok(None);
        }
        let gathered = owner
            .await_stage(gather_selector_in_context(value, context))
            .await??;
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

pub(super) fn numeric_tensor_indices_in_context(
    context: Option<&crate::context::RuntimeContext>,
    tensor: &Tensor,
    upper_bound: Option<usize>,
) -> VmResult<ExactArrayStorage<usize>> {
    crate::indexing::plan::admission::array(context, tensor.len(), |index| {
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
        checked_positive_index(scalar, upper_bound)
    })
}

pub async fn materialize_index_value(value: &Value) -> VmResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        let context = crate::context::legacy::active();
        return gather_selector_in_context(value, context.as_ref())
            .await
            .map_err(map_index_gather_error);
    }
    Ok(value.clone())
}

pub(super) async fn materialize_index_value_in_context<'a>(
    context: Option<&crate::context::RuntimeContext>,
    value: &'a Value,
) -> VmResult<Cow<'a, Value>> {
    crate::indexing::plan::admission::checkpoint(context)?;
    if matches!(value, Value::GpuTensor(_)) {
        gather_selector_in_context(value, context)
            .await
            .map(Cow::Owned)
    } else {
        Ok(Cow::Borrowed(value))
    }
}

pub(super) fn logical_indices_linear_in_context(
    context: Option<&crate::context::RuntimeContext>,
    array: &LogicalArray,
    total_len: usize,
) -> VmResult<ExactArrayStorage<usize>> {
    let mut count = 0usize;
    for (index, &selected) in array.data.iter().enumerate() {
        crate::indexing::plan::admission::checkpoint(context)?;
        if selected != 0 {
            if index >= total_len {
                return Err(mex(
                    "IndexOutOfBounds",
                    "Logical index exceeds array bounds",
                ));
            }
            count = count
                .checked_add(1)
                .ok_or_else(crate::indexing::plan::admission::geometry)?;
        }
    }
    let mut source = array
        .data
        .iter()
        .enumerate()
        .filter_map(|(index, &selected)| (selected != 0).then_some(index + 1));
    crate::indexing::plan::admission::array(context, count, |_| {
        source
            .next()
            .ok_or_else(crate::indexing::plan::admission::geometry)
    })
}

pub async fn indices_from_value_linear(
    value: &Value,
    total_len: usize,
) -> VmResult<ExactArrayStorage<usize>> {
    let context = crate::context::legacy::active();
    indices_from_value_linear_in_context(context.as_ref(), value, total_len).await
}
async fn indices_from_value_linear_in_context(
    context: Option<&crate::context::RuntimeContext>,
    value: &Value,
    total_len: usize,
) -> VmResult<ExactArrayStorage<usize>> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    if matches!(value, Value::GpuTensor(_)) {
        let gathered = owner
            .await_stage(gather_selector_in_context(value, context))
            .await??;
        return indices_from_host_value_linear_in_context(context, &gathered, total_len);
    }
    indices_from_host_value_linear_in_context(context, value, total_len)
}

async fn selector_from_value_dim_with_bounds(
    context: Option<&crate::context::RuntimeContext>,
    value: &Value,
    dim_len: usize,
    require_in_bounds: bool,
) -> VmResult<SliceSelector> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    if matches!(value, Value::GpuTensor(_)) {
        let gathered = owner
            .await_stage(gather_selector_in_context(value, context))
            .await??;
        return selector_from_host_value_dim(context, &gathered, dim_len, require_in_bounds);
    }
    selector_from_host_value_dim(context, value, dim_len, require_in_bounds)
}

pub async fn selector_from_value_dim(value: &Value, dim_len: usize) -> VmResult<SliceSelector> {
    let context = crate::context::legacy::active();
    selector_from_value_dim_with_bounds(context.as_ref(), value, dim_len, true).await
}

pub(super) fn truth_indices(
    context: Option<&crate::context::RuntimeContext>,
    selected: bool,
) -> VmResult<ExactArrayStorage<usize>> {
    crate::indexing::plan::admission::array(context, usize::from(selected), |_| Ok(1))
}

fn indices_from_host_value_linear_in_context(
    context: Option<&crate::context::RuntimeContext>,
    value: &Value,
    total_len: usize,
) -> VmResult<ExactArrayStorage<usize>> {
    if let Value::Bool(value) = value {
        return truth_indices(context, *value);
    }
    if let Value::LogicalArray(value) = value {
        if value.data.len() == 1 && is_scalar_shape(&value.shape) {
            return truth_indices(context, value.data[0] != 0);
        }
    }
    if let Some(value) = index_scalar_from_host_value(value) {
        let index = checked_positive_index(value, Some(total_len))?;
        return crate::indexing::plan::admission::array(context, 1, |_| Ok(index));
    }
    match value {
        Value::Tensor(value) => numeric_tensor_indices_in_context(context, value, Some(total_len)),
        Value::LogicalArray(value) => logical_indices_linear_in_context(context, value, total_len),
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for linear indexing",
        )),
    }
}

/// Linear logical stores use true-position bounds, not dimensional mask
/// length equality. Numeric assignment selectors keep their existing growth
/// policy; a logical true beyond the original extent never grants growth.
fn linear_assignment_selector_from_host(
    context: Option<&crate::context::RuntimeContext>,
    value: &Value,
    extent: usize,
) -> VmResult<SliceSelector> {
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
        let indices = indices_from_host_value_linear_in_context(context, value, extent)?;
        if indices.iter().any(|index| *index > extent) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        Ok(SliceSelector::Indices(indices))
    } else {
        selector_from_host_value_dim(context, value, extent, false)
    }
}

fn selector_from_host_value_dim(
    context: Option<&crate::context::RuntimeContext>,
    value: &Value,
    dim_len: usize,
    require_in_bounds: bool,
) -> VmResult<SliceSelector> {
    if let Value::Bool(value) = value {
        return truth_indices(context, *value).map(SliceSelector::Indices);
    }
    if let Value::LogicalArray(value) = value {
        if value.data.len() == 1 && is_scalar_shape(&value.shape) {
            return truth_indices(context, value.data[0] != 0).map(SliceSelector::Indices);
        }
    }
    if let Some(index) = index_scalar_from_host_value(value) {
        return checked_positive_index(index, require_in_bounds.then_some(dim_len))
            .map(SliceSelector::Scalar);
    }
    match value {
        Value::Tensor(value) => {
            numeric_tensor_indices_in_context(context, value, require_in_bounds.then_some(dim_len))
                .map(SliceSelector::Indices)
        }
        Value::LogicalArray(value) => {
            if value.data.len() != dim_len {
                return Err(mex(
                    "IndexShape",
                    "Logical mask length mismatch for dimension",
                ));
            }
            logical_indices_linear_in_context(context, value, dim_len).map(SliceSelector::Indices)
        }
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for slicing",
        )),
    }
}

fn used_numeric_has_gpu(dims: usize, colon_mask: u32, end_mask: u32, numeric: &[Value]) -> bool {
    if dims == 1 {
        return (colon_mask & 1) == 0
            && (end_mask & 1) == 0
            && numeric
                .first()
                .is_some_and(|value| matches!(value, Value::GpuTensor(_)));
    }
    let mut numeric = numeric.iter();
    (0..dims).any(|dimension| {
        if selector_mask_has_dim(colon_mask | end_mask, dimension) {
            false
        } else {
            numeric
                .next()
                .is_some_and(|value| matches!(value, Value::GpuTensor(_)))
        }
    })
}

async fn materialize_used_numeric<'a>(
    context: Option<&crate::context::RuntimeContext>,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &'a [Value],
    linear: bool,
) -> VmResult<ExactArrayStorage<Cow<'a, Value>>> {
    use crate::indexing::plan::admission;
    let owner = admission::Owner::new(context)?;
    let count = (0..dims)
        .filter(|dimension| !selector_mask_has_dim(colon_mask | end_mask, *dimension))
        .count();
    let mut values = crate::context::host_allocation::prepare_host_array_optional(context, count)?;
    let mut used = numeric.iter();
    for dimension in 0..dims {
        owner.check()?;
        if selector_mask_has_dim(colon_mask | end_mask, dimension) {
            continue;
        }
        let value = used.next().ok_or_else(|| {
            mex(
                "MissingNumericIndex",
                if linear {
                    "missing numeric index for linear slice"
                } else {
                    "missing numeric index for slice"
                },
            )
        })?;
        let value = if let Value::GpuTensor(handle) = value {
            let result = owner
                .await_stage(gather_selector_in_context(value, context))
                .await?;
            let gathered = if linear || total_len_from_shape(handle.shape()) != 1 {
                result.map_err(map_index_gather_error)?
            } else {
                result?
            };
            Cow::Owned(gathered)
        } else {
            Cow::Borrowed(value)
        };
        values.try_push(value).map_err(|_| admission::geometry())?;
    }
    owner.check()?;
    values.try_finish().map_err(|_| admission::geometry())
}

pub async fn build_slice_selectors(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    let context = crate::context::legacy::active();
    build_selectors(
        context.as_ref(),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        false,
    )
    .await
}
pub async fn build_slice_selectors_in_context(
    context: &crate::context::RuntimeContext,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    build_selectors(
        Some(context),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        false,
    )
    .await
}
pub async fn build_assignment_selectors(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    let context = crate::context::legacy::active();
    build_selectors(
        context.as_ref(),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        true,
    )
    .await
}
pub async fn build_assignment_selectors_in_context(
    context: &crate::context::RuntimeContext,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    build_selectors(
        Some(context),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        true,
    )
    .await
}
pub(super) async fn build_selectors(
    context: Option<&crate::context::RuntimeContext>,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
    assignment: bool,
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    validate_shape(base_shape, dims)?;
    let materialized = materialize_used_numeric(
        context,
        dims,
        colon_mask,
        end_mask,
        numeric,
        !assignment && dims == 1,
    )
    .await?;
    let result = selectors_from_host(
        context,
        dims,
        colon_mask,
        end_mask,
        &materialized,
        base_shape,
        assignment,
    )?;
    owner.check()?;
    Ok(result)
}
fn validate_shape(base_shape: &[usize], dims: usize) -> VmResult<()> {
    if dims == 0 {
        return Err(mex("MissingIndex", "At least one index is required"));
    }
    for dimension in 0..dims {
        crate::indexing::plan::effective_index_extent(base_shape, dims, dimension)?;
    }
    Ok(())
}
pub fn build_slice_selectors_host(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Option<ExactArrayStorage<SliceSelector>>> {
    let context = crate::context::legacy::active();
    build_selectors_host(
        context.as_ref(),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        false,
    )
}
pub fn build_slice_selectors_host_in_context(
    context: &crate::context::RuntimeContext,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Option<ExactArrayStorage<SliceSelector>>> {
    build_selectors_host(
        Some(context),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        false,
    )
}
pub fn build_assignment_selectors_host(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Option<ExactArrayStorage<SliceSelector>>> {
    let context = crate::context::legacy::active();
    build_selectors_host(
        context.as_ref(),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        true,
    )
}
pub fn build_assignment_selectors_host_in_context(
    context: &crate::context::RuntimeContext,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Option<ExactArrayStorage<SliceSelector>>> {
    build_selectors_host(
        Some(context),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        true,
    )
}
pub(super) fn build_selectors_host(
    context: Option<&crate::context::RuntimeContext>,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
    assignment: bool,
) -> VmResult<Option<ExactArrayStorage<SliceSelector>>> {
    crate::indexing::plan::admission::checkpoint(context)?;
    if used_numeric_has_gpu(dims, colon_mask, end_mask, numeric) {
        return Ok(None);
    }
    selectors_from_host(
        context, dims, colon_mask, end_mask, numeric, base_shape, assignment,
    )
    .map(Some)
}

fn selectors_from_host<B: Borrow<Value>>(
    context: Option<&crate::context::RuntimeContext>,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[B],
    base_shape: &[usize],
    assignment: bool,
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    use crate::indexing::plan::{
        admission, effective_index_extent, logical_linear_output_shape_in_context,
        numeric_linear_output_shape_in_context,
    };
    validate_shape(base_shape, dims)?;
    let mut numeric_position = 0usize;
    admission::array(context, dims, |dimension| {
        let extent = effective_index_extent(base_shape, dims, dimension)?;
        if selector_mask_has_dim(colon_mask, dimension) {
            return Ok(SliceSelector::Colon);
        }
        if selector_mask_has_dim(end_mask, dimension) {
            return Ok(SliceSelector::Scalar(if !assignment && dims == 1 {
                extent.max(1)
            } else {
                extent
            }));
        }
        let value = numeric
            .get(numeric_position)
            .map(Borrow::borrow)
            .ok_or_else(|| {
                mex(
                    "MissingNumericIndex",
                    if dims == 1 && !assignment {
                        "missing numeric index for linear slice"
                    } else {
                        "missing numeric index for slice"
                    },
                )
            })?;
        numeric_position += 1;
        if dims == 1 && assignment {
            return linear_assignment_selector_from_host(context, value, extent);
        }
        if dims == 1 && !assignment {
            return match value {
                Value::Tensor(tensor) => Ok(SliceSelector::LinearIndices {
                    values: numeric_tensor_indices_in_context(context, tensor, Some(extent))?,
                    output_shape: numeric_linear_output_shape_in_context(
                        context,
                        base_shape,
                        tensor.shape(),
                        tensor.len(),
                    )?,
                }),
                value @ (Value::Bool(_) | Value::LogicalArray(_)) => {
                    let values = indices_from_host_value_linear_in_context(context, value, extent)?;
                    let mask_shape: &[usize] = match value {
                        Value::LogicalArray(mask) => &mask.shape,
                        Value::Bool(_) => &[1, 1],
                        _ => unreachable!(),
                    };
                    let output_shape = logical_linear_output_shape_in_context(
                        context,
                        base_shape,
                        mask_shape,
                        values.len(),
                    )?;
                    Ok(SliceSelector::LinearIndices {
                        values,
                        output_shape,
                    })
                }
                value => indices_from_host_value_linear_in_context(context, value, extent)
                    .map(SliceSelector::Indices),
            };
        }
        let bounded = !assignment || matches!(value, Value::Bool(_) | Value::LogicalArray(_));
        selector_from_host_value_dim(context, value, extent, bounded)
    })
}
pub async fn build_sparse_assignment_selectors(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    build_assignment_selectors(dims, colon_mask, end_mask, numeric, base_shape).await
}
pub async fn build_scalar_selectors(
    raw_indices: &[Value],
) -> VmResult<ExactArrayStorage<SliceSelector>> {
    let context = crate::context::legacy::active();
    let mut selectors = crate::context::host_allocation::prepare_host_array_optional(
        context.as_ref(),
        raw_indices.len(),
    )?;
    for value in raw_indices {
        let scalar = index_scalar_from_value_in_context(context.as_ref(), value)
            .await?
            .ok_or_else(|| {
                mex(
                    "ScalarIndexRequired",
                    "Indexing requires scalar numeric indices",
                )
            })?;
        let index = checked_positive_index(scalar, None)?;
        selectors
            .try_push(SliceSelector::Scalar(index))
            .map_err(|_| crate::indexing::plan::admission::geometry())?;
    }
    selectors
        .try_finish()
        .map_err(|_| crate::indexing::plan::admission::geometry())
}

#[cfg(test)]
mod tests {
    use super::{
        build_component_selectors, build_component_selectors_with_context, build_scalar_selectors,
        build_slice_selectors, indices_from_value_linear, selector_from_value_dim, SliceSelector,
    };
    use runmat_value::{IntValue, IntegerStorage, LogicalArray, Tensor, Value};

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
    fn linear_logical_assignment_routes_share_original_extent_bounds() {
        for (bits, accepted) in [
            (vec![0, 1], true),
            (vec![0, 1, 0, 0, 0], true),
            (vec![0, 0, 0, 1], false),
        ] {
            let mask =
                Value::LogicalArray(LogicalArray::new(bits.clone(), vec![1, bits.len()]).unwrap());
            let numeric = [mask.clone()];
            let host = super::build_assignment_selectors_host(1, 0, 0, &numeric, &[1, 3])
                .map(|selectors| selectors.expect("host selector"));
            let asynchronous = futures::executor::block_on(super::build_assignment_selectors(
                1,
                0,
                0,
                &numeric,
                &[1, 3],
            ));
            let components = [crate::object::indexing::ObjectIndexComponent::Value(mask)];
            let object =
                futures::executor::block_on(build_component_selectors(&components, &[1, 3], true));
            for result in [host, asynchronous, object] {
                if accepted {
                    let selectors = result.expect("prefix/false-overhang assignment");
                    assert_eq!(selectors[0].iter_indices(3).collect::<Vec<_>>(), vec![2]);
                } else {
                    assert_eq!(
                        result.unwrap_err().identifier(),
                        Some("RunMat:IndexOutOfBounds")
                    );
                }
            }
        }
        // Scalar logical true cannot grow an empty target either.
        for value in [
            Value::Bool(true),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
        ] {
            let error =
                super::build_assignment_selectors_host(1, 0, 0, &[value], &[1, 0]).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:IndexOutOfBounds"));
        }
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
    fn build_scalar_selectors_rejects_zero_index() {
        let err = futures::executor::block_on(build_scalar_selectors(&[Value::Num(0.0)]))
            .expect_err("zero cell scalar index should fail");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn build_scalar_selectors_rejects_negative_index() {
        let err = futures::executor::block_on(build_scalar_selectors(&[Value::Num(-2.0)]))
            .expect_err("negative cell scalar index should fail");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
        let err = futures::executor::block_on(build_scalar_selectors(&[Value::String(
            "not an index".into(),
        )]))
        .expect_err("scalar selector conversion preserves its diagnostic");
        assert_eq!(err.identifier(), Some("RunMat:ScalarIndexRequired"));
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn numeric_linear_selectors_preserve_exact_low_rank_output_shape() {
        for (shape, values) in [(Vec::new(), vec![2]), (vec![2], vec![2, 1])] {
            let input = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(values.clone()), shape.clone()).unwrap(),
            );
            let components = [crate::object::indexing::ObjectIndexComponent::Value(
                input.clone(),
            )];
            let component_selectors =
                futures::executor::block_on(build_component_selectors(&components, &[2, 2], false))
                    .unwrap();
            let slice_selectors =
                futures::executor::block_on(build_slice_selectors(1, 0, 0, &[input], &[2, 2]))
                    .unwrap();
            for selectors in [component_selectors, slice_selectors] {
                let [SliceSelector::LinearIndices {
                    values: indices,
                    output_shape,
                }] = selectors.as_slice()
                else {
                    panic!("expected numeric linear selector");
                };
                assert_eq!(
                    indices,
                    &values
                        .iter()
                        .map(|value| usize::from(*value))
                        .collect::<Vec<_>>()
                );
                assert_eq!(output_shape, &shape);
            }
        }
        let components = [
            crate::object::indexing::ObjectIndexComponent::Value(Value::Num(1.0)),
            crate::object::indexing::ObjectIndexComponent::End,
        ];
        let selectors = futures::executor::block_on(build_component_selectors_with_context(
            None,
            &components,
            &[2, 3, 4],
            false,
        ))
        .unwrap();
        assert!(matches!(selectors[1], SliceSelector::Scalar(12)));
        let plan = crate::indexing::plan::build_index_plan(&selectors, 2, &[2, 3, 4]).unwrap();
        assert_eq!(&*plan.indices, &[22]);
        let error = crate::object::indexing::ObjectIndexComponent::End
            .protocol_value()
            .expect_err("unresolved syntax cannot be public protocol text");
        assert_eq!(error.identifier(), Some("RunMat:UnresolvedObjectIndexEnd"));
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
