use crate::indexing::selectors::SliceSelector;
use crate::indexing::EndExpr;
use crate::runtime_error::semantic_error as mex;
use crate::{builtins::common::shape::is_scalar_shape, RuntimeError};
use runmat_value::{ExactArrayStorage, SharedShape, Value};
use std::future::Future;

pub(crate) mod admission;

pub type VmResult<T> = Result<T, RuntimeError>;

/// Whether an indexed store may extend numeric selector bounds. Deletion and
/// other bounded stores retain the ordinary read-selector policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexedStorePlanMode {
    Bounded,
    Grow,
}

/// Executor-neutral selector and plan admission for indexed stores.
pub async fn build_indexed_store_plan(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
    mode: IndexedStorePlanMode,
) -> VmResult<IndexPlan> {
    let context = crate::context::legacy::active();
    build_indexed_store_plan_with_context(
        context.as_ref(),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        mode,
    )
    .await
}
pub async fn build_indexed_store_plan_in_context(
    context: &crate::context::RuntimeContext,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
    mode: IndexedStorePlanMode,
) -> VmResult<IndexPlan> {
    build_indexed_store_plan_with_context(
        Some(context),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        mode,
    )
    .await
}
async fn build_indexed_store_plan_with_context(
    context: Option<&crate::context::RuntimeContext>,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
    mode: IndexedStorePlanMode,
) -> VmResult<IndexPlan> {
    let owner = admission::Owner::new(context)?;
    let selectors = owner
        .await_stage(crate::indexing::selectors::build_selectors(
            context,
            dims,
            colon_mask,
            end_mask,
            numeric,
            base_shape,
            matches!(mode, IndexedStorePlanMode::Grow),
        ))
        .await??;
    let plan = match mode {
        IndexedStorePlanMode::Bounded => build_index_plan_with_context(
            context,
            &selectors,
            dims,
            base_shape,
            LinearShapePolicy::Read,
        )?,
        IndexedStorePlanMode::Grow => {
            build_assignment_plan_with_context(context, &selectors, dims, base_shape)?
        }
    };
    owner.check()?;
    Ok(plan)
}

/// None is a genuine provider gather requirement, not an indexing error.
pub fn build_indexed_store_plan_host(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
    mode: IndexedStorePlanMode,
) -> VmResult<Option<IndexPlan>> {
    let context = crate::context::legacy::active();
    build_indexed_store_plan_host_with_context(
        context.as_ref(),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        mode,
    )
}
pub fn build_indexed_store_plan_host_in_context(
    context: &crate::context::RuntimeContext,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
    mode: IndexedStorePlanMode,
) -> VmResult<Option<IndexPlan>> {
    build_indexed_store_plan_host_with_context(
        Some(context),
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        mode,
    )
}
fn build_indexed_store_plan_host_with_context(
    context: Option<&crate::context::RuntimeContext>,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
    mode: IndexedStorePlanMode,
) -> VmResult<Option<IndexPlan>> {
    let owner = admission::Owner::new(context)?;
    let Some(selectors) = crate::indexing::selectors::build_selectors_host(
        context,
        dims,
        colon_mask,
        end_mask,
        numeric,
        base_shape,
        matches!(mode, IndexedStorePlanMode::Grow),
    )?
    else {
        return Ok(None);
    };
    let plan = match mode {
        IndexedStorePlanMode::Bounded => build_index_plan_with_context(
            context,
            &selectors,
            dims,
            base_shape,
            LinearShapePolicy::Read,
        )?,
        IndexedStorePlanMode::Grow => {
            build_assignment_plan_with_context(context, &selectors, dims, base_shape)?
        }
    };
    owner.check()?;
    Ok(Some(plan))
}

const MAX_INDEX_PLAN_ELEMENTS: usize = 50_000_000;

#[derive(Debug, Clone, Default)]
pub struct IndexPlanProperties {
    pub full_row: Option<usize>,
    pub full_column: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct IndexPlan<I = u32> {
    pub indices: ExactArrayStorage<I>,
    pub output_shape: SharedShape,
    pub selection_lengths: SharedShape,
    pub dims: usize,
    pub base_shape: SharedShape,
    pub properties: IndexPlanProperties,
}

impl IndexPlan {
    /// Existing Char/Cell constructors use row-major pages and normalize only
    /// rank zero/one. Retain a normal shape; fund a real normalization backing.
    pub(crate) fn row_major_output_shape_in_context(
        &self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> VmResult<SharedShape> {
        Self::row_major_shape(context, &self.output_shape)
    }
    pub(crate) fn row_major_base_shape_in_context(
        &self,
        context: Option<&crate::context::RuntimeContext>,
    ) -> VmResult<SharedShape> {
        Self::row_major_shape(context, &self.base_shape)
    }
    fn row_major_shape(
        context: Option<&crate::context::RuntimeContext>,
        source: &SharedShape,
    ) -> VmResult<SharedShape> {
        let owner = admission::Owner::new(context)?;
        let shape = match source.as_slice() {
            [] => admission::shape(context, 2, |_| Ok(0))?,
            [length] => {
                admission::shape(context, 2, |index| Ok(if index == 0 { 1 } else { *length }))?
            }
            _ => source.clone(),
        };
        owner.check()?;
        Ok(shape)
    }
    /// Ordinary compatibility constructor for already-owned, unadmitted
    /// vectors. Managed planners construct their physical owners directly.
    pub fn new(
        indices: Vec<u32>,
        output_shape: Vec<usize>,
        selection_lengths: Vec<usize>,
        dims: usize,
        base_shape: Vec<usize>,
    ) -> Self {
        let properties = derive_plan_properties(&indices, dims, &base_shape);
        Self {
            indices: ExactArrayStorage::shared_unbounded(indices),
            output_shape: SharedShape::new_unmanaged(output_shape),
            selection_lengths: SharedShape::new_unmanaged(selection_lengths),
            dims,
            base_shape: SharedShape::new_unmanaged(base_shape),
            properties,
        }
    }

    fn from_storage(
        indices: ExactArrayStorage<u32>,
        output_shape: SharedShape,
        selection_lengths: SharedShape,
        dims: usize,
        base_shape: SharedShape,
    ) -> Self {
        let properties = derive_plan_properties(&indices, dims, &base_shape);
        Self {
            indices,
            output_shape,
            selection_lengths,
            dims,
            base_shape,
            properties,
        }
    }
}

fn derive_plan_properties(
    indices: &[u32],
    dims: usize,
    base_shape: &[usize],
) -> IndexPlanProperties {
    derive_plan_properties_with(indices, dims, base_shape, |index| index as usize)
}
fn derive_plan_properties_with<I: Copy>(
    indices: &[I],
    dims: usize,
    base_shape: &[usize],
    position: impl Fn(I) -> usize,
) -> IndexPlanProperties {
    let mut properties = IndexPlanProperties {
        full_row: None,
        full_column: None,
    };
    if dims != 2 || indices.is_empty() {
        return properties;
    }
    let rows = base_shape.first().copied().unwrap_or(1);
    let cols = base_shape.get(1).copied().unwrap_or(1);
    if indices.len() == rows {
        let first = position(indices[0]);
        if first.is_multiple_of(rows) {
            let col = first / rows;
            if col < cols
                && indices
                    .iter()
                    .enumerate()
                    .all(|(r, &idx)| position(idx) == col * rows + r)
            {
                properties.full_column = Some(col);
            }
        }
    }
    if indices.len() == cols {
        let first = position(indices[0]);
        let row = first % rows;
        if row < rows
            && indices
                .iter()
                .enumerate()
                .all(|(c, &idx)| position(idx) == row + c * rows)
        {
            properties.full_row = Some(row);
        }
    }
    properties
}

pub fn total_len_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        1
    } else {
        shape.iter().copied().product()
    }
}

/// Returns the dimensions visible to a MATLAB indexing expression.
///
/// A single subscript addresses the linearized array. When an expression has
/// fewer subscripts than the array rank, the final supplied subscript addresses
/// the product of that dimension and every remaining dimension.
pub fn effective_index_shape(base_shape: &[usize], dims: usize) -> VmResult<SharedShape> {
    let context = crate::context::legacy::active();
    effective_index_shape_in_context(context.as_ref(), base_shape, dims)
}

pub(in crate::indexing) fn effective_index_shape_in_context(
    context: Option<&crate::context::RuntimeContext>,
    base_shape: &[usize],
    dims: usize,
) -> VmResult<SharedShape> {
    if dims == 0 {
        return Err(mex("MissingIndex", "At least one index is required"));
    }
    admission::shape(context, dims, |dimension| {
        effective_index_extent(base_shape, dims, dimension)
    })
}

pub(crate) fn effective_index_extent(
    base_shape: &[usize],
    dims: usize,
    dimension: usize,
) -> VmResult<usize> {
    if dims == 1 {
        checked_total_len_from_shape(base_shape)
    } else if dimension + 1 == dims {
        base_shape
            .iter()
            .skip(dimension)
            .try_fold(1usize, |total, extent| total.checked_mul(*extent))
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))
    } else {
        Ok(base_shape.get(dimension).copied().unwrap_or(1))
    }
}

pub(super) fn numeric_linear_output_shape_in_context(
    context: Option<&crate::context::RuntimeContext>,
    source_shape: &[usize],
    selector_shape: &[usize],
    selected_len: usize,
) -> VmResult<SharedShape> {
    let shape = numeric_linear_shape(source_shape, selector_shape, selected_len);
    let shape = shape.as_slice();
    admission::shape(context, shape.len(), |dimension| Ok(shape[dimension]))
}

/// Linear logical selection keeps a vector source's orientation only when
/// the mask is also a vector. Every other selection produces a column.
pub(crate) fn logical_linear_output_shape_in_context(
    context: Option<&crate::context::RuntimeContext>,
    source_shape: &[usize],
    mask_shape: &[usize],
    selected_len: usize,
) -> VmResult<SharedShape> {
    let source_row = is_vector_shape(source_shape)
        && is_vector_shape(mask_shape)
        && source_shape.first().copied().unwrap_or(1) == 1;
    let shape = if source_row {
        [1, selected_len]
    } else {
        [selected_len, 1]
    };
    admission::shape(context, 2, |dimension| Ok(shape[dimension]))
}

fn checked_total_len_from_shape(shape: &[usize]) -> VmResult<usize> {
    if is_scalar_shape(shape) {
        return Ok(1);
    }
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))
    })
}

fn checked_u32_index(index: usize) -> VmResult<u32> {
    u32::try_from(index).map_err(|_| mex("IndexOutOfBounds", "Index exceeds supported range"))
}

fn checked_selection_cardinality(lengths: &[usize]) -> VmResult<usize> {
    let count = lengths.iter().try_fold(1usize, |total, length| {
        total
            .checked_mul(*length)
            .ok_or_else(|| mex("IndexOutOfBounds", "Index selection cardinality overflow"))
    })?;
    if count > MAX_INDEX_PLAN_ELEMENTS {
        return Err(mex(
            "IndexOutOfBounds",
            "Index selection exceeds the supported materialization limit",
        ));
    }
    Ok(count)
}

/// A numeric vector selector inherits a vector source's orientation. A
/// non-vector source, or a matrix-shaped selector, keeps the selector shape.
/// Colon and logical selection have separate shape rules.
#[cfg(test)]
fn numeric_linear_output_shape(
    source_shape: &[usize],
    selector_shape: &[usize],
    selected_len: usize,
) -> Vec<usize> {
    numeric_linear_shape(source_shape, selector_shape, selected_len)
        .as_slice()
        .to_vec()
}

enum NumericLinearShape<'a> {
    Borrowed(&'a [usize]),
    Vector([usize; 2]),
}
impl NumericLinearShape<'_> {
    fn as_slice(&self) -> &[usize] {
        match self {
            Self::Borrowed(shape) => shape,
            Self::Vector(shape) => shape,
        }
    }
}
fn numeric_linear_shape<'a>(
    source_shape: &[usize],
    selector_shape: &'a [usize],
    selected_len: usize,
) -> NumericLinearShape<'a> {
    let source_is_non_scalar = source_shape.iter().any(|&extent| extent != 1);
    if is_vector_shape(source_shape) && source_is_non_scalar && is_vector_shape(selector_shape) {
        if source_shape.first().copied().unwrap_or(1) == 1 {
            NumericLinearShape::Vector([1, selected_len])
        } else {
            NumericLinearShape::Vector([selected_len, 1])
        }
    } else {
        NumericLinearShape::Borrowed(selector_shape)
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    shape.iter().skip(2).all(|&extent| extent == 1) && (rows == 1 || cols == 1)
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

pub fn build_index_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan> {
    let context = crate::context::legacy::active();
    build_index_plan_with_context(
        context.as_ref(),
        selectors,
        dims,
        base_shape,
        LinearShapePolicy::Read,
    )
}

pub fn build_index_plan_in_context(
    context: &crate::context::RuntimeContext,
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan> {
    build_index_plan_with_context(
        Some(context),
        selectors,
        dims,
        base_shape,
        LinearShapePolicy::Read,
    )
}

fn selector_len(selector: Option<&SliceSelector>, extent: usize) -> usize {
    selector.map_or(0, |selector| selector.selection_len(extent))
}
fn selector_value(selector: &SliceSelector, position: usize) -> usize {
    selector
        .selection_value(position)
        .expect("bounded selector position")
}

#[derive(Clone, Copy)]
enum LinearShapePolicy {
    Read,
    Assignment,
}

fn build_index_plan_with_context(
    context: Option<&crate::context::RuntimeContext>,
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
    linear_shape: LinearShapePolicy,
) -> VmResult<IndexPlan> {
    build_index_plan_with_sink(
        context,
        selectors,
        dims,
        base_shape,
        linear_shape,
        checked_u32_index,
        |index| index as usize,
    )
}

/// Virtual arrays retain only the addressed usize offsets. Source geometry
/// and selector semantics are identical to physical-array planning.
pub(crate) fn build_virtual_index_plan_with_context(
    context: Option<&crate::context::RuntimeContext>,
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan<usize>> {
    build_index_plan_with_sink(
        context,
        selectors,
        dims,
        base_shape,
        LinearShapePolicy::Read,
        Ok,
        |index| index,
    )
}

fn build_index_plan_with_sink<I: Copy>(
    context: Option<&crate::context::RuntimeContext>,
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
    linear_shape: LinearShapePolicy,
    store: impl FnMut(usize) -> VmResult<I>,
    position: impl Fn(I) -> usize,
) -> VmResult<IndexPlan<I>> {
    let owner = admission::Owner::new(context)?;
    let total_len = checked_total_len_from_shape(base_shape)?;
    let effective = effective_index_shape_in_context(context, base_shape, dims)?;
    if dims > 1 && selectors.len() < dims {
        return Err(mex("MissingIndex", "An index selector is missing"));
    }
    let lengths = admission::shape(context, dims, |dimension| {
        Ok(selector_len(selectors.get(dimension), effective[dimension]))
    })?;
    let total_out = checked_selection_cardinality(&lengths)?;
    // Validate the original borrowed selector before output construction. No
    // copied per-dimension lists or Cartesian-product payloads are needed.
    for (dimension, selector) in selectors.iter().enumerate().take(dims) {
        if matches!(selector, SliceSelector::Colon) {
            continue;
        }
        let extent = if dims == 1 {
            total_len
        } else {
            effective[dimension]
        };
        for position in 0..lengths[dimension] {
            admission::checkpoint(context)?;
            let value = selector_value(selector, position);
            if value == 0 || value > extent {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
        }
    }
    let indices = enumerate_linear_indices_with_sink(
        context,
        dims,
        &lengths,
        &effective,
        total_out,
        |dimension, position| {
            selectors
                .get(dimension)
                .map(|selector| selector_value(selector, position))
                .ok_or_else(|| mex("MissingIndex", "An index selector is missing"))
        },
        store,
    )?;
    let output_shape = if dims == 1 {
        match selectors.first() {
            Some(SliceSelector::Colon) if matches!(linear_shape, LinearShapePolicy::Read) => {
                admission::shape(context, 2, |dimension| {
                    Ok(if dimension == 0 { total_out } else { 1 })
                })?
            }
            Some(SliceSelector::LinearIndices { output_shape, .. }) => {
                admission::shape(context, output_shape.len(), |dimension| {
                    Ok(output_shape[dimension])
                })?
            }
            _ => {
                let row = total_out > 1
                    && base_shape.first().copied().unwrap_or(1) == 1
                    && base_shape.get(1).copied().unwrap_or(1) > 1;
                let shape = if total_out == 0 {
                    [0, 1]
                } else if total_out <= 1 {
                    [1, 1]
                } else if row {
                    [1, total_out]
                } else {
                    [total_out, 1]
                };
                admission::shape(context, 2, |dimension| Ok(shape[dimension]))?
            }
        }
    } else if total_out == 1 {
        admission::shape(context, 2, |_| Ok(1))?
    } else {
        let mut rank = dims;
        while rank > 2
            && lengths[rank - 1] == 1
            && matches!(selectors[rank - 1], SliceSelector::Scalar(_))
        {
            rank -= 1;
        }
        admission::shape(context, rank, |dimension| Ok(lengths[dimension]))?
    };
    let base_shape = admission::shape(context, base_shape.len(), |dimension| {
        Ok(base_shape[dimension])
    })?;
    owner.check()?;
    let properties = derive_plan_properties_with(&indices, dims, &base_shape, position);
    Ok(IndexPlan {
        indices,
        output_shape,
        selection_lengths: lengths,
        dims,
        base_shape,
        properties,
    })
}

fn enumerate_linear_indices(
    context: Option<&crate::context::RuntimeContext>,
    dims: usize,
    lengths: &[usize],
    shape: &[usize],
    total: usize,
    selector: impl FnMut(usize, usize) -> VmResult<usize>,
) -> VmResult<ExactArrayStorage<u32>> {
    enumerate_linear_indices_with_sink(
        context,
        dims,
        lengths,
        shape,
        total,
        selector,
        checked_u32_index,
    )
}
fn enumerate_linear_indices_with_sink<I>(
    context: Option<&crate::context::RuntimeContext>,
    dims: usize,
    lengths: &[usize],
    shape: &[usize],
    total: usize,
    mut selector: impl FnMut(usize, usize) -> VmResult<usize>,
    mut store: impl FnMut(usize) -> VmResult<I>,
) -> VmResult<ExactArrayStorage<I>> {
    if total != 0 {
        let mut stride = 1usize;
        for dimension in 0..dims.saturating_sub(1) {
            stride = stride
                .checked_mul(shape[dimension].max(1))
                .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
        }
    }
    admission::array(context, total, |ordinal| {
        let mut remainder = ordinal;
        let mut stride = 1usize;
        let mut linear = 0usize;
        for dimension in 0..dims {
            let position = remainder % lengths[dimension];
            remainder /= lengths[dimension];
            let value = selector(dimension, position)?
                .checked_sub(1)
                .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
            let offset = value
                .checked_mul(stride)
                .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
            linear = linear
                .checked_add(offset)
                .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
            if dimension + 1 < dims {
                stride = stride
                    .checked_mul(shape[dimension].max(1))
                    .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
            }
        }
        store(linear)
    })
}

/// Builds a write plan for sparse two-subscript indexed assignment. The plan's
/// base shape is the target shape so CSC updates use the expanded stride, while
/// colon selectors are materialized against the original shape before planning.
pub fn build_sparse_assignment_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan> {
    build_assignment_plan(selectors, dims, base_shape)
}

/// Builds an indexed-assignment plan that may grow an aggregate. Colon and
/// logical selectors must already have been materialized against the original
/// shape; numeric selectors may extend addressed dimensions.
pub fn build_assignment_plan(
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan> {
    let context = crate::context::legacy::active();
    build_assignment_plan_with_context(context.as_ref(), selectors, dims, base_shape)
}
pub fn build_assignment_plan_in_context(
    context: &crate::context::RuntimeContext,
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan> {
    build_assignment_plan_with_context(Some(context), selectors, dims, base_shape)
}
fn selector_max(selector: Option<&SliceSelector>, extent: usize) -> VmResult<usize> {
    match selector {
        Some(SliceSelector::Colon) => Ok(extent),
        Some(SliceSelector::Scalar(0)) => Err(mex("IndexOutOfBounds", "Index out of bounds")),
        Some(SliceSelector::Scalar(value)) => Ok(*value),
        Some(SliceSelector::Indices(values) | SliceSelector::LinearIndices { values, .. }) => {
            if values.contains(&0) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
            Ok(values.iter().copied().max().unwrap_or(0))
        }
        None => Ok(0),
    }
}
fn build_assignment_plan_with_context(
    context: Option<&crate::context::RuntimeContext>,
    selectors: &[SliceSelector],
    dims: usize,
    base_shape: &[usize],
) -> VmResult<IndexPlan> {
    let owner = admission::Owner::new(context)?;
    let effective = effective_index_shape_in_context(context, base_shape, dims)?;
    if dims == 1 {
        let target_len = selector_max(selectors.first(), effective[0])?.max(effective[0]);
        let target_shape = linear_growth_shape_in_context(context, base_shape, target_len)?;
        let plan = build_index_plan_with_context(
            context,
            selectors,
            1,
            &target_shape,
            LinearShapePolicy::Assignment,
        )?;
        owner.check()?;
        return Ok(plan);
    }
    let target_shape = admission::shape(context, dims, |dimension| {
        Ok(selector_max(selectors.get(dimension), effective[dimension])?.max(effective[dimension]))
    })?;
    // Missing internal selectors historically denote an empty dimension.
    // Only that case needs a metadata owner; normal source selectors borrow
    // the original exact storage, including implicit colon ranges.
    let padded = if selectors.len() < dims {
        Some(admission::array(context, dims, |dimension| {
            Ok(selectors
                .get(dimension)
                .cloned()
                .unwrap_or_else(|| SliceSelector::Indices(ExactArrayStorage::default())))
        })?)
    } else {
        None
    };
    let selected = padded
        .as_ref()
        .map_or(selectors, ExactArrayStorage::as_slice);
    let mut plan = build_index_plan_with_context(
        context,
        selected,
        dims,
        &target_shape,
        LinearShapePolicy::Assignment,
    )?;
    plan.base_shape = assignment_result_shape_in_context(context, base_shape, dims, &target_shape)?;
    owner.check()?;
    Ok(plan)
}

fn linear_growth_shape_in_context(
    context: Option<&crate::context::RuntimeContext>,
    base_shape: &[usize],
    target_len: usize,
) -> VmResult<SharedShape> {
    let rank = base_shape.len().max(2);
    let dimension = |index: usize| base_shape.get(index).copied().unwrap_or(1);
    let current_len = (0..rank)
        .try_fold(1usize, |total, index| total.checked_mul(dimension(index)))
        .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
    if target_len <= current_len {
        return admission::shape(context, rank, |index| Ok(dimension(index)));
    }
    if current_len == 0 {
        return admission::shape(context, 2, |index| {
            Ok(if index == 0 { 1 } else { target_len })
        });
    }
    let row = (2..rank).all(|index| dimension(index) == 1) && dimension(0) == 1;
    let column = (1..rank).all(|index| dimension(index) == 1);
    if row || column {
        return admission::shape(context, 2, |index| {
            Ok(if (index == 0) == row { 1 } else { target_len })
        });
    }
    let leading = (0..rank - 1)
        .try_fold(1usize, |total, index| total.checked_mul(dimension(index)))
        .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
    let last = target_len.div_ceil(leading.max(1));
    admission::shape(context, rank, |index| {
        Ok(if index + 1 == rank {
            last
        } else {
            dimension(index)
        })
    })
}

#[derive(Clone)]
enum ExprSel<'a> {
    Colon,
    Scalar(usize),
    Indices(ExactArrayStorage<usize>),
    Range {
        start: i64,
        step: i64,
        end_off: &'a EndExpr,
    },
}

pub struct ExprPlanSpec<'a> {
    pub dims: usize,
    pub colon_mask: u32,
    pub end_mask: u32,
    pub range_dims: &'a [usize],
    pub range_params: &'a [(f64, f64)],
    pub range_start_exprs: &'a [Option<EndExpr>],
    pub range_step_exprs: &'a [Option<EndExpr>],
    pub range_end_exprs: &'a [EndExpr],
    pub numeric: &'a [Value],
    pub shape: &'a [usize],
}

fn selector_mask_has_dim(mask: u32, dim: usize) -> bool {
    dim < u32::BITS as usize && (mask & (1u32 << dim)) != 0
}

fn validate_expr_range_selector_plan(
    context: Option<&crate::context::RuntimeContext>,
    spec: &ExprPlanSpec<'_>,
) -> Result<admission::Scratch<usize>, RuntimeError> {
    let range_len = spec.range_dims.len();
    if spec.range_params.len() != range_len
        || spec.range_start_exprs.len() != range_len
        || spec.range_step_exprs.len() != range_len
        || spec.range_end_exprs.len() != range_len
    {
        return Err(mex(
            "InvalidRangeSelectorPlan",
            "inconsistent range selector metadata",
        ));
    }

    let mut by_dim = admission::Scratch::<usize>::new(context, spec.dims)?;
    for (pos, &dim) in spec.range_dims.iter().enumerate() {
        if dim >= spec.dims {
            return Err(mex(
                "InvalidRangeSelectorDim",
                "range selector dimension is out of bounds",
            ));
        }
        let conflicts_with_colon = selector_mask_has_dim(spec.colon_mask, dim);
        let conflicts_with_end = selector_mask_has_dim(spec.end_mask, dim);
        if conflicts_with_colon || conflicts_with_end {
            return Err(mex(
                "InvalidRangeSelectorPlan",
                "range selector conflicts with colon/end selector masks",
            ));
        }
        let previous = &mut by_dim.as_mut_slice()[dim];
        if *previous != 0 {
            return Err(mex(
                "InvalidRangeSelectorPlan",
                "range selector dimension appears more than once",
            ));
        }
        *previous = pos.checked_add(1).ok_or_else(admission::geometry)?;
    }
    Ok(by_dim)
}

pub async fn build_expr_index_plan<ResolveEnd, Fut>(
    spec: ExprPlanSpec<'_>,
    resolve_end: ResolveEnd,
) -> Result<IndexPlan, RuntimeError>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = Result<f64, RuntimeError>>,
{
    let context = crate::context::legacy::active();
    build_expr_index_plan_with_growth(context.as_ref(), spec, resolve_end, false).await
}

pub async fn build_expr_sparse_assignment_plan<ResolveEnd, Fut>(
    spec: ExprPlanSpec<'_>,
    resolve_end: ResolveEnd,
) -> Result<IndexPlan, RuntimeError>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = Result<f64, RuntimeError>>,
{
    let context = crate::context::legacy::active();
    build_expr_index_plan_with_growth(context.as_ref(), spec, resolve_end, true).await
}

pub async fn build_expr_assignment_plan<ResolveEnd, Fut>(
    spec: ExprPlanSpec<'_>,
    resolve_end: ResolveEnd,
) -> Result<IndexPlan, RuntimeError>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = Result<f64, RuntimeError>>,
{
    let context = crate::context::legacy::active();
    build_expr_index_plan_with_growth(context.as_ref(), spec, resolve_end, true).await
}

pub async fn build_expr_index_plan_in_context<ResolveEnd, Fut>(
    context: &crate::context::RuntimeContext,
    spec: ExprPlanSpec<'_>,
    resolve_end: ResolveEnd,
) -> VmResult<IndexPlan>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = VmResult<f64>>,
{
    build_expr_index_plan_with_growth(Some(context), spec, resolve_end, false).await
}
pub async fn build_expr_assignment_plan_in_context<ResolveEnd, Fut>(
    context: &crate::context::RuntimeContext,
    spec: ExprPlanSpec<'_>,
    resolve_end: ResolveEnd,
) -> VmResult<IndexPlan>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = VmResult<f64>>,
{
    build_expr_index_plan_with_growth(Some(context), spec, resolve_end, true).await
}

enum ResolvedExprSelector {
    Colon(usize),
    Scalar(usize),
    Indices(ExactArrayStorage<usize>),
    Range {
        start: i64,
        step: i64,
        count: usize,
        first: usize,
        last: usize,
    },
}
impl ResolvedExprSelector {
    fn len(&self) -> usize {
        match self {
            Self::Colon(count) | Self::Range { count, .. } => *count,
            Self::Scalar(_) => 1,
            Self::Indices(values) => values.len(),
        }
    }
    fn value(&self, position: usize) -> VmResult<usize> {
        match self {
            Self::Colon(_) => position.checked_add(1).ok_or_else(admission::geometry),
            Self::Scalar(value) => Ok(*value),
            Self::Indices(values) => Ok(values[position]),
            Self::Range { start, step, .. } => {
                usize::try_from(i128::from(*start) + i128::from(*step) * position as i128)
                    .map_err(|_| mex("IndexOutOfBounds", "Index out of bounds"))
            }
        }
    }
    fn max(&self) -> usize {
        match self {
            Self::Colon(count) => *count,
            Self::Scalar(value) => *value,
            Self::Indices(values) => values.iter().copied().max().unwrap_or(0),
            Self::Range {
                first, last, count, ..
            } => {
                if *count == 0 {
                    0
                } else {
                    (*first).max(*last)
                }
            }
        }
    }
}

fn resolve_range(
    start: i64,
    step: i64,
    end_bound: f64,
    extent: usize,
    growth: bool,
) -> VmResult<ResolvedExprSelector> {
    if step == 0 {
        return Err(mex("IndexStepZero", "Index step cannot be zero"));
    }
    if !end_bound.is_finite() {
        return Err(mex(
            "UnsupportedIndexType",
            "Index values must be positive integers or logical values",
        ));
    }
    let end = if step > 0 {
        end_bound.floor()
    } else {
        end_bound.ceil()
    };
    if end < i64::MIN as f64 || end > i64::MAX as f64 {
        return Err(mex(
            "UnsupportedIndexType",
            "Index values must be positive integers or logical values",
        ));
    }
    let end = end as i64;
    let forward = step > 0;
    let count = if (forward && start > end) || (!forward && start < end) {
        0
    } else {
        let distance = (i128::from(end) - i128::from(start)).abs();
        usize::try_from(distance / i128::from(step).abs() + 1)
            .map_err(|_| mex("IndexOutOfBounds", "Index result dimensions overflow"))?
    };
    let (first, last) = if count == 0 {
        (0, 0)
    } else {
        let last = i128::from(start) + i128::from(step) * (count - 1) as i128;
        let first =
            usize::try_from(start).map_err(|_| mex("IndexOutOfBounds", "Index out of bounds"))?;
        let last =
            usize::try_from(last).map_err(|_| mex("IndexOutOfBounds", "Index out of bounds"))?;
        if first == 0 || last == 0 || (!growth && first.max(last) > extent) {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        (first, last)
    };
    Ok(ResolvedExprSelector::Range {
        start,
        step,
        count,
        first,
        last,
    })
}

async fn build_expr_index_plan_with_growth<ResolveEnd, Fut>(
    context: Option<&crate::context::RuntimeContext>,
    spec: ExprPlanSpec<'_>,
    mut resolve_end: ResolveEnd,
    allow_growth: bool,
) -> VmResult<IndexPlan>
where
    ResolveEnd: FnMut(usize, &EndExpr) -> Fut,
    Fut: Future<Output = VmResult<f64>>,
{
    use crate::indexing::selectors::{
        index_scalar_from_value_in_context, logical_indices_linear_in_context,
        materialize_index_value_in_context, numeric_tensor_indices_in_context, truth_indices,
    };
    let owner = admission::Owner::new(context)?;
    let full_shape = effective_index_shape_in_context(context, spec.shape, spec.dims)?;
    let range_positions = validate_expr_range_selector_plan(context, &spec)?;
    // Both exact control arrays are admitted before any end callback/gather.
    let mut selectors =
        crate::context::host_allocation::prepare_host_array_optional(context, spec.dims)?;
    let mut resolved =
        crate::context::host_allocation::prepare_host_array_optional(context, spec.dims)?;
    let mut linear_output_shape = None;
    let mut linear_numeric_selector = false;
    let mut numeric_position = 0usize;
    for dimension in 0..spec.dims {
        owner.check()?;
        let extent = full_shape[dimension];
        let range_position = range_positions.as_slice()[dimension];
        let selector = if selector_mask_has_dim(spec.colon_mask, dimension) {
            ExprSel::Colon
        } else if selector_mask_has_dim(spec.end_mask, dimension) {
            ExprSel::Scalar(extent)
        } else if range_position != 0 {
            let position = range_position - 1;
            let (raw_start, raw_step) = spec.range_params[position];
            let start = if let Some(expression) = &spec.range_start_exprs[position] {
                owner
                    .await_stage(async { resolve_end(extent, expression).await })
                    .await??
            } else {
                raw_start
            };
            let step = if let Some(expression) = &spec.range_step_exprs[position] {
                owner
                    .await_stage(async { resolve_end(extent, expression).await })
                    .await??
            } else {
                raw_step
            };
            let invalid = || {
                mex(
                    "UnsupportedIndexType",
                    "Index values must be positive integers or logical values",
                )
            };
            ExprSel::Range {
                start: exact_index_from_f64(start).ok_or_else(invalid)?,
                step: exact_index_from_f64(step).ok_or_else(invalid)?,
                end_off: &spec.range_end_exprs[position],
            }
        } else {
            let value = spec
                .numeric
                .get(numeric_position)
                .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index"))?;
            numeric_position += 1;
            if let Some(index) = owner
                .await_stage(index_scalar_from_value_in_context(context, value))
                .await??
            {
                let index = index
                    .positive_usize()
                    .ok_or_else(|| mex("IndexOutOfBounds", "Index out of bounds"))?;
                ExprSel::Scalar(index)
            } else {
                let materialized = owner
                    .await_stage(materialize_index_value_in_context(context, value))
                    .await??;
                match materialized.as_ref() {
                    Value::Bool(value) => {
                        let values = truth_indices(context, *value)?;
                        if spec.dims == 1 {
                            linear_output_shape = Some(logical_linear_output_shape_in_context(
                                context,
                                spec.shape,
                                &[1, 1],
                                values.len(),
                            )?);
                        }
                        ExprSel::Indices(values)
                    }
                    Value::LogicalArray(value)
                        if value.data.len() == 1 && is_scalar_shape(&value.shape) =>
                    {
                        let values = truth_indices(context, value.data[0] != 0)?;
                        if spec.dims == 1 {
                            linear_output_shape = Some(logical_linear_output_shape_in_context(
                                context,
                                spec.shape,
                                &value.shape,
                                values.len(),
                            )?);
                        }
                        ExprSel::Indices(values)
                    }
                    Value::LogicalArray(value) => {
                        if spec.dims != 1 && value.data.len() != extent {
                            return Err(mex(
                                "IndexShape",
                                "Logical mask length mismatch for dimension",
                            ));
                        }
                        let values = logical_indices_linear_in_context(context, value, extent)?;
                        if spec.dims == 1 {
                            linear_output_shape = Some(logical_linear_output_shape_in_context(
                                context,
                                spec.shape,
                                &value.shape,
                                values.len(),
                            )?);
                        }
                        ExprSel::Indices(values)
                    }
                    Value::Tensor(value) => {
                        if spec.dims == 1 {
                            linear_output_shape = Some(value.shared_shape());
                            linear_numeric_selector = true;
                        }
                        ExprSel::Indices(numeric_tensor_indices_in_context(context, value, None)?)
                    }
                    _ => return Err(mex("UnsupportedIndexType", "Unsupported index type")),
                }
            }
        };
        selectors
            .try_push(selector)
            .map_err(|_| admission::geometry())?;
    }
    let selectors = selectors.try_finish().map_err(|_| admission::geometry())?;
    // Preserve the original second-pass range-end callback order. Arithmetic
    // ranges stay implicit; no enumerated range mirror is allocated.
    for (dimension, selector) in selectors.iter().enumerate() {
        owner.check()?;
        let extent = full_shape[dimension];
        let value = match selector {
            ExprSel::Colon => ResolvedExprSelector::Colon(extent),
            ExprSel::Scalar(value) => ResolvedExprSelector::Scalar(*value),
            ExprSel::Indices(values) => ResolvedExprSelector::Indices(values.clone()),
            ExprSel::Range {
                start,
                step,
                end_off,
            } => resolve_range(
                *start,
                *step,
                owner
                    .await_stage(async { resolve_end(extent, end_off).await })
                    .await??,
                extent,
                allow_growth,
            )?,
        };
        for position in 0..value.len() {
            owner.check()?;
            // Range endpoints already prove every monotone member. Other
            // selectors retain the original per-element bounds validation.
            if matches!(
                value,
                ResolvedExprSelector::Colon(_) | ResolvedExprSelector::Range { .. }
            ) {
                break;
            }
            let index = value.value(position)?;
            if index == 0 || (!allow_growth && index > extent) {
                return Err(mex("IndexOutOfBounds", "Index out of bounds"));
            }
        }
        resolved
            .try_push(value)
            .map_err(|_| admission::geometry())?;
    }
    let resolved = resolved.try_finish().map_err(|_| admission::geometry())?;
    let lengths = admission::shape(
        context,
        spec.dims,
        |dimension| Ok(resolved[dimension].len()),
    )?;
    let total = lengths
        .iter()
        .try_fold(1usize, |total, length| total.checked_mul(*length))
        .ok_or_else(|| mex("IndexOutOfBounds", "Index result dimensions overflow"))?;
    let planned_shape = admission::shape(context, spec.dims, |dimension| {
        Ok(if allow_growth {
            full_shape[dimension].max(resolved[dimension].max())
        } else {
            full_shape[dimension]
        })
    })?;
    // Expression plans historically validate the complete stride product,
    // including empty selections. Preserve that diagnostic boundary.
    planned_shape
        .iter()
        .try_fold(1usize, |stride, extent| stride.checked_mul(*extent))
        .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
    let indices = enumerate_linear_indices(
        context,
        spec.dims,
        &lengths,
        &planned_shape,
        total,
        |dimension, position| resolved[dimension].value(position),
    )?;
    let first = selectors.first();
    let output_shape = if spec.dims == 1 {
        if let Some(shape) = linear_output_shape {
            if linear_numeric_selector {
                numeric_linear_output_shape_in_context(context, spec.shape, &shape, total)?
            } else {
                shape
            }
        } else if total == 0 && matches!(first, Some(ExprSel::Range { .. })) {
            numeric_linear_output_shape_in_context(context, spec.shape, &[1, 0], 0)?
        } else if total == 0 && matches!(first, Some(ExprSel::Colon)) {
            admission::shape(context, 2, |dimension| {
                Ok(if dimension == 0 { 0 } else { 1 })
            })?
        } else if total <= 1 && total != 0 {
            admission::shape(context, 2, |_| Ok(1))?
        } else if matches!(first, Some(ExprSel::Range { .. })) {
            numeric_linear_output_shape_in_context(context, spec.shape, &[1, total], total)?
        } else {
            let row = !matches!(first, Some(ExprSel::Colon))
                && spec.shape.first().copied().unwrap_or(1) == 1
                && spec.shape.get(1).copied().unwrap_or(1) > 1;
            admission::shape(context, 2, |dimension| {
                Ok(if (dimension == 0) == row { 1 } else { total })
            })?
        }
    } else {
        let mut rank = spec.dims;
        while rank > 2
            && lengths[rank - 1] == 1
            && matches!(selectors[rank - 1], ExprSel::Scalar(_))
        {
            rank -= 1;
        }
        admission::shape(context, rank, |dimension| Ok(lengths[dimension]))?
    };
    let base_shape =
        assignment_result_shape_in_context(context, spec.shape, spec.dims, &planned_shape)?;
    owner.check()?;
    Ok(IndexPlan::from_storage(
        indices,
        output_shape,
        lengths,
        spec.dims,
        base_shape,
    ))
}

fn assignment_result_shape_in_context(
    context: Option<&crate::context::RuntimeContext>,
    base_shape: &[usize],
    dims: usize,
    planned_shape: &[usize],
) -> VmResult<SharedShape> {
    if dims == 1 {
        return linear_growth_shape_in_context(context, base_shape, planned_shape[0]);
    }
    let unchanged = planned_shape.len() == dims
        && (0..dims).try_fold(true, |same, dimension| {
            Ok::<_, RuntimeError>(
                same && planned_shape[dimension]
                    == effective_index_extent(base_shape, dims, dimension)?,
            )
        })?;
    let source = if unchanged { base_shape } else { planned_shape };
    let rank = if unchanged {
        source.len()
    } else {
        source.len().max(2)
    };
    admission::shape(context, rank, |index| {
        Ok(source.get(index).copied().unwrap_or(1))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_assignment_plan, build_expr_index_plan, build_index_plan,
        build_sparse_assignment_plan, checked_selection_cardinality, numeric_linear_output_shape,
        ExprPlanSpec,
    };
    use crate::indexing::selectors::{build_slice_selectors, SliceSelector};
    use crate::indexing::EndExpr;
    use runmat_value::{IntegerStorage, LogicalArray, Tensor, Value};

    #[test]
    fn numeric_linear_vector_result_uses_source_orientation_only_for_vector_pairs() {
        for (source, selector, count, expected) in [
            (vec![4, 1], vec![1, 2], 2, vec![2, 1]),
            (vec![1, 4], vec![2, 1], 2, vec![1, 2]),
            (vec![4, 1], vec![2, 2], 4, vec![2, 2]),
            (vec![2, 2], vec![1, 2], 2, vec![1, 2]),
            (vec![2, 2], vec![2], 2, vec![2]),
            (vec![0, 1], vec![1, 0], 0, vec![0, 1]),
        ] {
            assert_eq!(
                numeric_linear_output_shape(&source, &selector, count),
                expected
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn expression_numeric_selector_uses_vector_source_orientation() {
        let numeric = [Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![2, 1]), vec![2]).unwrap(),
        )];
        let plan = futures::executor::block_on(build_expr_index_plan(
            ExprPlanSpec {
                dims: 1,
                colon_mask: 0,
                end_mask: 0,
                range_dims: &[],
                range_params: &[],
                range_start_exprs: &[],
                range_step_exprs: &[],
                range_end_exprs: &[],
                numeric: &numeric,
                shape: &[1, 2],
            },
            |_, _| async { Ok(0.0) },
        ))
        .expect("rank-one selector");
        assert_eq!(plan.indices, vec![1, 0]);
        assert_eq!(plan.output_shape, vec![1, 2]);
    }

    #[test]
    fn sparse_assignment_plan_expands_numeric_dimensions_but_keeps_colon_at_old_extent() {
        let selectors = vec![
            SliceSelector::Indices(vec![3, 4].into()),
            SliceSelector::Colon,
        ];
        let plan = build_sparse_assignment_plan(&selectors, 2, &[2, 2])
            .expect("sparse assignment plan should grow rows");
        assert_eq!(plan.base_shape, vec![4, 2]);
        assert_eq!(plan.selection_lengths, vec![2, 2]);
        assert_eq!(plan.indices, vec![2, 3, 6, 7]);
    }

    #[test]
    fn fewer_subscripts_collapse_remaining_dimensions() {
        let plan = build_index_plan(
            &[SliceSelector::Scalar(2), SliceSelector::Scalar(4)],
            2,
            &[2, 2, 2],
        )
        .unwrap();
        assert_eq!(plan.indices, vec![7]);
    }

    #[test]
    fn bounded_plan_preserves_selector_geometry_separately_from_result_shape() {
        let shape = [1, 1, 3];
        let scalar = build_index_plan(
            &[
                SliceSelector::Scalar(1),
                SliceSelector::Scalar(1),
                SliceSelector::Scalar(2),
            ],
            3,
            &shape,
        )
        .unwrap();
        assert_eq!(scalar.output_shape, vec![1, 1]);
        assert_eq!(scalar.selection_lengths, vec![1, 1, 1]);
        assert_eq!(scalar.indices, vec![1]);

        let empty = build_index_plan(
            &[
                SliceSelector::Scalar(1),
                SliceSelector::Scalar(1),
                SliceSelector::Indices(Vec::new().into()),
            ],
            3,
            &shape,
        )
        .unwrap();
        assert!(empty.indices.is_empty());
        assert_eq!(empty.selection_lengths, vec![1, 1, 0]);
    }

    #[test]
    fn assignment_plan_grows_linear_and_multidimensional_targets() {
        let linear = build_assignment_plan(&[SliceSelector::Scalar(5)], 1, &[2, 2]).unwrap();
        assert_eq!(linear.indices, vec![4]);
        assert_eq!(linear.base_shape, vec![2, 3]);

        let subscripts = build_assignment_plan(
            &[SliceSelector::Scalar(3), SliceSelector::Scalar(2)],
            2,
            &[2, 2],
        )
        .unwrap();
        assert_eq!(subscripts.indices, vec![5]);
        assert_eq!(subscripts.base_shape, vec![3, 2]);

        let nd_shape = [2, 1, 3];
        let bounded_linear = build_assignment_plan(&[SliceSelector::Scalar(4)], 1, &nd_shape)
            .expect("in-bounds N-D linear assignment");
        assert_eq!(bounded_linear.base_shape, nd_shape.to_vec());
        let growing_linear = build_assignment_plan(&[SliceSelector::Scalar(7)], 1, &nd_shape)
            .expect("N-D linear assignment grows the final axis");
        assert_eq!(growing_linear.base_shape, vec![2, 1, 4]);
        let nd_with_leading_singleton = [1, 2, 3];
        let growing_leading_singleton =
            build_assignment_plan(&[SliceSelector::Scalar(7)], 1, &nd_with_leading_singleton)
                .expect("N-D linear growth retains nonsingleton trailing axes");
        assert_eq!(growing_leading_singleton.base_shape, vec![1, 2, 4]);

        let bounded_nd = build_assignment_plan(
            &[
                SliceSelector::Scalar(1),
                SliceSelector::Scalar(1),
                SliceSelector::Scalar(2),
            ],
            3,
            &nd_shape,
        )
        .expect("in-bounds three-subscript assignment");
        assert_eq!(bounded_nd.base_shape, nd_shape.to_vec());
        let growing_nd = build_assignment_plan(
            &[
                SliceSelector::Scalar(1),
                SliceSelector::Scalar(1),
                SliceSelector::Scalar(4),
            ],
            3,
            &nd_shape,
        )
        .expect("three-subscript assignment grows the third axis");
        assert_eq!(growing_nd.base_shape, vec![2, 1, 4]);

        let collapsed_growth = build_assignment_plan(
            &[SliceSelector::Scalar(2), SliceSelector::Scalar(4)],
            2,
            &nd_shape,
        )
        .expect("fewer subscripts intentionally collapse trailing axes");
        assert_eq!(collapsed_growth.base_shape, vec![2, 4]);
    }

    #[test]
    fn plain_and_expr_linear_range_plans_match() {
        futures::executor::block_on(async {
            let shape = vec![1, 10];
            let numeric = vec![Value::Tensor(
                Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![1, 4]).unwrap(),
            )];
            let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[0],
                    range_params: &[(2.0, 2.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::Sub(
                        Box::new(EndExpr::End),
                        Box::new(EndExpr::Const(1.0)),
                    )],
                    numeric: &[],
                    shape: &shape,
                },
                |dim_len, expr| {
                    let expr = expr.clone();
                    async move {
                        Ok(match &expr {
                            EndExpr::End => dim_len as f64,
                            EndExpr::Const(value) => *value,
                            EndExpr::Sub(lhs, rhs) => {
                                let lhs_val = match lhs.as_ref() {
                                    EndExpr::End => dim_len as f64,
                                    EndExpr::Const(value) => *value,
                                    other => panic!("unsupported lhs expr: {other:?}"),
                                };
                                let rhs_val = match rhs.as_ref() {
                                    EndExpr::Const(value) => *value,
                                    other => panic!("unsupported rhs expr: {other:?}"),
                                };
                                lhs_val - rhs_val
                            }
                            other => panic!("unsupported expr: {other:?}"),
                        })
                    }
                },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
            assert_eq!(plain.properties.full_row, expr.properties.full_row);
            assert_eq!(plain.properties.full_column, expr.properties.full_column);
        })
    }

    #[test]
    fn expr_integer_index_vectors_use_exact_storage_for_all_classes() {
        macro_rules! assert_indices {
            ($storage:expr) => {{
                let indices =
                    Tensor::new_integer($storage, vec![1, 2]).expect("typed integer index tensor");
                let numeric = vec![Value::Tensor(indices)];
                let plan = futures::executor::block_on(build_expr_index_plan(
                    ExprPlanSpec {
                        dims: 1,
                        colon_mask: 0,
                        end_mask: 0,
                        range_dims: &[],
                        range_params: &[],
                        range_start_exprs: &[],
                        range_step_exprs: &[],
                        range_end_exprs: &[],
                        numeric: &numeric,
                        shape: &[1, 2],
                    },
                    |_, _| async { Ok(0.0) },
                ))
                .expect("exact integer vector index plan");
                assert_eq!(plan.indices, vec![0, 1]);
            }};
        }

        assert_indices!(IntegerStorage::I8(vec![1, 2]));
        assert_indices!(IntegerStorage::I16(vec![1, 2]));
        assert_indices!(IntegerStorage::I32(vec![1, 2]));
        assert_indices!(IntegerStorage::I64(vec![1, 2]));
        assert_indices!(IntegerStorage::U8(vec![1, 2]));
        assert_indices!(IntegerStorage::U16(vec![1, 2]));
        assert_indices!(IntegerStorage::U32(vec![1, 2]));
        assert_indices!(IntegerStorage::U64(vec![1, 2]));
    }

    #[test]
    fn expr_integer_index_vectors_reject_wide_values() {
        let indices = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX]), vec![1, 2])
            .expect("typed integer index tensor");
        let numeric = vec![Value::Tensor(indices)];

        let err = futures::executor::block_on(build_expr_index_plan(
            ExprPlanSpec {
                dims: 1,
                colon_mask: 0,
                end_mask: 0,
                range_dims: &[],
                range_params: &[],
                range_start_exprs: &[],
                range_step_exprs: &[],
                range_end_exprs: &[],
                numeric: &numeric,
                shape: &[1, 2],
            },
            |_, _| async { Ok(0.0) },
        ))
        .expect_err("wide exact integer index must not use its float mirror");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn plain_and_expr_column_plans_match_properties() {
        futures::executor::block_on(async {
            let shape = vec![3, 4];
            let numeric = vec![Value::Num(3.0)];
            let plain_selectors = build_slice_selectors(2, 1, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 2, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 1,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.properties.full_column, Some(2));
            assert_eq!(plain.properties.full_column, expr.properties.full_column);
            assert_eq!(plain.properties.full_row, expr.properties.full_row);
        })
    }

    #[test]
    fn expr_linear_range_on_column_vector_uses_source_orientation() {
        futures::executor::block_on(async {
            let plan = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[0],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::Var(0)],
                    numeric: &[],
                    shape: &[10, 1],
                },
                |_dim_len, expr| {
                    let expr = expr.clone();
                    async move {
                        match expr {
                            EndExpr::Var(_) => Ok(6.0),
                            other => panic!("unsupported expr: {other:?}"),
                        }
                    }
                },
            )
            .await
            .unwrap();
            assert_eq!(plan.indices, vec![0, 1, 2, 3, 4, 5]);
            assert_eq!(plan.output_shape, vec![6, 1]);
            assert_eq!(plan.selection_lengths, vec![6]);
        })
    }

    #[test]
    fn expr_empty_linear_range_uses_source_orientation() {
        futures::executor::block_on(async {
            let plan = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[0],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::Var(0)],
                    numeric: &[],
                    shape: &[10, 1],
                },
                |_dim_len, expr| {
                    let expr = expr.clone();
                    async move {
                        match expr {
                            EndExpr::Var(_) => Ok(0.0),
                            other => panic!("unsupported expr: {other:?}"),
                        }
                    }
                },
            )
            .await
            .unwrap();
            assert!(plan.indices.is_empty());
            assert_eq!(plan.output_shape, vec![0, 1]);
            assert_eq!(plan.selection_lengths, vec![0]);
        })
    }

    #[test]
    fn expr_plan_rejects_range_dim_conflicting_with_colon_mask() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0b01,
                    end_mask: 0,
                    range_dims: &[0],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::End],
                    numeric: &[Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("range/colon conflict should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
        })
    }

    #[test]
    fn expr_plan_rejects_range_dim_conflicting_with_end_mask() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0,
                    end_mask: 0b10,
                    range_dims: &[1],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::End],
                    numeric: &[Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("range/end conflict should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
        })
    }

    #[test]
    fn expr_plan_rejects_duplicate_range_dims() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[1, 1],
                    range_params: &[(1.0, 1.0), (1.0, 1.0)],
                    range_start_exprs: &[None, None],
                    range_step_exprs: &[None, None],
                    range_end_exprs: &[EndExpr::End, EndExpr::End],
                    numeric: &[Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("duplicate range dims should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
        })
    }

    #[test]
    fn expr_plan_rejects_out_of_bounds_range_dim() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[2],
                    range_params: &[(1.0, 1.0)],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::End],
                    numeric: &[Value::Num(1.0), Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("out-of-bounds range dim should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorDim"));
        })
    }

    #[test]
    fn expr_plan_rejects_inconsistent_range_metadata_lengths() {
        futures::executor::block_on(async {
            let err = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[1],
                    range_params: &[],
                    range_start_exprs: &[None],
                    range_step_exprs: &[None],
                    range_end_exprs: &[EndExpr::End],
                    numeric: &[Value::Num(1.0)],
                    shape: &[3, 3],
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect_err("inconsistent range metadata should fail");
            assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
        })
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn index_plan_rejects_sparse_sized_indices_beyond_u32_storage() {
        let selectors = vec![SliceSelector::Scalar((u32::MAX as usize) + 2)];
        let err = build_index_plan(&selectors, 1, &[u32::MAX as usize + 2, 1])
            .expect_err("linear index should exceed u32 plan storage");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
        let wide = super::build_virtual_index_plan_with_context(
            None,
            &selectors,
            1,
            &[u32::MAX as usize + 2, 1],
        )
        .expect("virtual metadata uses actual usize indices");
        assert_eq!(&*wide.indices, &[u32::MAX as usize + 1]);
        assert_eq!(&*wide.output_shape, &[1, 1]);
        let exact = (1usize << 53) + 3;
        let wide = super::build_virtual_index_plan_with_context(
            None,
            &[SliceSelector::Scalar(exact)],
            1,
            &[exact, 1],
        )
        .expect("virtual offset never passes through binary64");
        assert_eq!(&*wide.indices, &[exact - 1]);
    }

    #[test]
    fn index_plan_rejects_dimension_product_overflow() {
        let selectors = vec![SliceSelector::Colon];
        let err = build_index_plan(&selectors, 1, &[usize::MAX, 2])
            .expect_err("linearized sparse shape should overflow");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn index_plan_rejects_oversized_linear_colon_before_materialization() {
        let err = build_index_plan(
            &[SliceSelector::Colon],
            1,
            &[super::MAX_INDEX_PLAN_ELEMENTS + 1, 1],
        )
        .expect_err("oversized colon must fail before allocating indices");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
        assert!(err.message().contains("materialization limit"));
    }

    #[test]
    fn index_plan_rejects_oversized_cartesian_product_before_enumeration() {
        let repeated = vec![1; 10_000];
        let selectors = [
            SliceSelector::Indices(repeated.clone().into()),
            SliceSelector::Indices(repeated.into()),
        ];
        let err = build_index_plan(&selectors, 2, &[1, 1])
            .expect_err("oversized Cartesian selection must fail before enumeration");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
        assert!(err.message().contains("materialization limit"));
    }

    #[test]
    fn index_plan_cardinality_overflow_is_deterministic() {
        let err = checked_selection_cardinality(&[usize::MAX, 2])
            .expect_err("overflowing selection cardinality must fail");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
        assert!(err.message().contains("cardinality overflow"));
    }

    #[test]
    fn expr_plan_supports_dims_beyond_mask_width() {
        futures::executor::block_on(async {
            let numeric = vec![Value::Num(1.0); 31];
            let shape = vec![1usize; 33];
            let plan = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 33,
                    colon_mask: 0b1,
                    end_mask: 0b10,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .expect("expr plan for dims beyond mask width");
            assert_eq!(plan.dims, 33);
            assert!(!plan.indices.is_empty());
        })
    }

    #[test]
    fn expr_plan_tensor_selector_length_match_uses_numeric_indices() {
        futures::executor::block_on(async {
            let shape = vec![3, 2];
            let numeric = vec![Value::Tensor(
                Tensor::new(vec![2.0, 1.0, 3.0], vec![1, 3]).expect("selector tensor"),
            )];
            let plain_selectors = build_slice_selectors(2, 0b10, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 2, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0b10,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn expr_plan_logical_selector_remains_logical_mask() {
        futures::executor::block_on(async {
            let shape = vec![3, 2];
            let numeric = vec![Value::LogicalArray(
                LogicalArray::new(vec![0, 1, 1], vec![1, 3]).expect("logical selector"),
            )];
            let plain_selectors = build_slice_selectors(2, 0b10, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 2, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 2,
                    colon_mask: 0b10,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn expr_plan_linear_numeric_vector_selector_uses_source_orientation() {
        futures::executor::block_on(async {
            let shape = vec![1, 10];
            let numeric = vec![Value::Tensor(
                Tensor::new(vec![2.0, 4.0], vec![2, 1]).expect("selector tensor"),
            )];
            let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, vec![1, 2]);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn expr_plan_linear_colon_selector_matches_plain_shape() {
        futures::executor::block_on(async {
            let shape = vec![1, 5];
            let plain = build_index_plan(&[SliceSelector::Colon], 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0b1,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &[],
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn linear_logical_mask_preserves_vector_source_orientation_across_plans() {
        futures::executor::block_on(async {
            for (shape, mask_shape, data, expected) in [
                (vec![1, 5], vec![1, 5], vec![1, 0, 1, 0, 1], vec![1, 3]),
                (vec![1, 5], vec![5, 1], vec![1, 0, 1, 0, 1], vec![1, 3]),
                (vec![5, 1], vec![1, 5], vec![1, 0, 1, 0, 1], vec![3, 1]),
                (vec![2, 3], vec![1, 6], vec![1, 0, 1, 0, 1, 0], vec![3, 1]),
                (vec![1, 6], vec![2, 3], vec![1, 0, 1, 0, 1, 0], vec![3, 1]),
            ] {
                let numeric = vec![Value::LogicalArray(
                    LogicalArray::new(data, mask_shape).expect("logical selector"),
                )];
                let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                    .await
                    .unwrap();
                let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
                let host_selectors = crate::indexing::selectors::build_slice_selectors_host(
                    1, 0, 0, &numeric, &shape,
                )
                .unwrap()
                .expect("host logical selector stays available");
                let host = build_index_plan(&host_selectors, 1, &shape).unwrap();
                let object_selectors = crate::indexing::selectors::build_component_selectors(
                    &[crate::object::indexing::ObjectIndexComponent::Value(
                        numeric[0].clone(),
                    )],
                    &shape,
                    false,
                )
                .await
                .unwrap();
                let object = build_index_plan(&object_selectors, 1, &shape).unwrap();
                let expr = build_expr_index_plan(
                    ExprPlanSpec {
                        dims: 1,
                        colon_mask: 0,
                        end_mask: 0,
                        range_dims: &[],
                        range_params: &[],
                        range_start_exprs: &[],
                        range_step_exprs: &[],
                        range_end_exprs: &[],
                        numeric: &numeric,
                        shape: &shape,
                    },
                    |_dim_len, _expr| async move { unreachable!() },
                )
                .await
                .unwrap();
                for plan in [&plain, &host, &object, &expr] {
                    assert_eq!(plan.output_shape, expected);
                    assert_eq!(plan.indices, plain.indices);
                    assert_eq!(plan.selection_lengths, plain.selection_lengths);
                }
            }
            for (shape, expected) in [(vec![1, 5], vec![1, 0]), (vec![2, 3], vec![0, 1])] {
                let numeric = vec![Value::Bool(false)];
                let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                    .await
                    .unwrap();
                let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
                let host_selectors = crate::indexing::selectors::build_slice_selectors_host(
                    1, 0, 0, &numeric, &shape,
                )
                .unwrap()
                .expect("host scalar logical selector stays available");
                let host = build_index_plan(&host_selectors, 1, &shape).unwrap();
                let object_selectors = crate::indexing::selectors::build_component_selectors(
                    &[crate::object::indexing::ObjectIndexComponent::Value(
                        Value::Bool(false),
                    )],
                    &shape,
                    false,
                )
                .await
                .unwrap();
                let object = build_index_plan(&object_selectors, 1, &shape).unwrap();
                let expr = build_expr_index_plan(
                    ExprPlanSpec {
                        dims: 1,
                        colon_mask: 0,
                        end_mask: 0,
                        range_dims: &[],
                        range_params: &[],
                        range_start_exprs: &[],
                        range_step_exprs: &[],
                        range_end_exprs: &[],
                        numeric: &numeric,
                        shape: &shape,
                    },
                    |_dim_len, _expr| async move { unreachable!() },
                )
                .await
                .unwrap();
                for plan in [&plain, &host, &object, &expr] {
                    assert!(plan.indices.is_empty());
                    assert_eq!(plan.output_shape, expected);
                }
            }
        })
    }

    #[test]
    fn expr_plan_short_linear_logical_mask_matches_plain_column_shape() {
        futures::executor::block_on(async {
            let shape = vec![2, 3];
            let numeric = vec![Value::LogicalArray(
                LogicalArray::new(vec![0, 1, 1], vec![1, 3]).expect("logical selector"),
            )];
            let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert_eq!(plain.indices, vec![1, 2]);
            assert_eq!(plain.indices, expr.indices);
            assert_eq!(plain.output_shape, vec![2, 1]);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }

    #[test]
    fn linear_empty_selector_uses_empty_column_shape() {
        let plan = build_index_plan(&[SliceSelector::Indices(Vec::new().into())], 1, &[1, 5])
            .expect("empty linear selector should build");
        assert!(plan.indices.is_empty());
        assert_eq!(plan.output_shape, vec![0, 1]);
    }

    #[test]
    fn expr_plan_linear_empty_logical_mask_matches_plain_shape() {
        futures::executor::block_on(async {
            let shape = vec![1, 5];
            let numeric = vec![Value::LogicalArray(
                LogicalArray::new(vec![0, 0, 0, 0, 0], vec![1, 5]).expect("logical selector"),
            )];
            let plain_selectors = build_slice_selectors(1, 0, 0, &numeric, &shape)
                .await
                .unwrap();
            let plain = build_index_plan(&plain_selectors, 1, &shape).unwrap();
            let expr = build_expr_index_plan(
                ExprPlanSpec {
                    dims: 1,
                    colon_mask: 0,
                    end_mask: 0,
                    range_dims: &[],
                    range_params: &[],
                    range_start_exprs: &[],
                    range_step_exprs: &[],
                    range_end_exprs: &[],
                    numeric: &numeric,
                    shape: &shape,
                },
                |_dim_len, _expr| async move { unreachable!() },
            )
            .await
            .unwrap();
            assert!(plain.indices.is_empty());
            assert!(expr.indices.is_empty());
            assert_eq!(plain.output_shape, vec![1, 0]);
            assert_eq!(plain.output_shape, expr.output_shape);
            assert_eq!(plain.selection_lengths, expr.selection_lengths);
        })
    }
}
