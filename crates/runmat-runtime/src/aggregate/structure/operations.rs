//! Runtime-selected structure operations over original immutable columns.
//! Destination slots are constructed once in field-major order. Rank-sized
//! planning storage is explicitly temporary; no element-map carrier is built.
use crate::{
    context::{
        host_allocation::{
            host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
            HostAllocationReceipt, HostTypedWorkspace,
        },
        RuntimeContext,
    },
    execution::{
        structure::{construct_generated_structure, replacement},
        RootedValue,
    },
    RuntimeError,
};
use runmat_value::{SharedShape, StructArray, StructValue, Value};

mod assignment;
mod assignment_prefix;
mod concatenate;
#[cfg(test)]
mod tests;
mod transform;
pub use assignment::assign;
pub(crate) use assignment::assign_in_context;
pub(crate) use assignment::assign_initializing_in_context;
pub(crate) use assignment::empty as empty_numeric_in_context;
pub(crate) use assignment_prefix::project_initializing_in_context;
pub(crate) use assignment_prefix::{selected_structure_element, SelectedStructureElement};
pub use concatenate::concatenate;
pub(crate) use concatenate::concatenate_arguments;
pub use transform::{circular_shift, flip, permute, reshape, tile};

enum Workspace {
    Managed(HostTypedWorkspace<usize>),
    Ordinary(Vec<usize>),
}
struct Scratch<'a> {
    storage: Workspace,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}
impl<'a> Scratch<'a> {
    fn new(runtime: Option<&'a RuntimeContext>, count: usize) -> Result<Self, RuntimeError> {
        let bytes =
            HostTypedWorkspace::<usize>::quoted_bytes(count).map_err(|e| allocation(runtime, e))?;
        let plan = HostAllocationPlan::new(bytes, 0).map_err(|e| allocation(runtime, e))?;
        let receipt =
            HostAllocationReceipt::admit(runtime, plan).map_err(|e| allocation(runtime, e))?;
        let storage = match &receipt {
            Some(receipt) => Workspace::Managed(
                receipt
                    .temporary_typed(runtime.expect("managed context"), count)
                    .map_err(|e| allocation(runtime, e))?,
            ),
            None => Workspace::Ordinary(vec![0; count]),
        };
        Ok(Self {
            storage,
            receipt,
            runtime,
        })
    }
    fn values(&self) -> &[usize] {
        match &self.storage {
            Workspace::Managed(v) => v.as_slice(),
            Workspace::Ordinary(v) => v,
        }
    }
    fn values_mut(&mut self) -> &mut [usize] {
        match &mut self.storage {
            Workspace::Managed(v) => v.as_mut_slice(),
            Workspace::Ordinary(v) => v,
        }
    }
    fn checkpoint(&self) -> Result<(), RuntimeError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed context"))
                .map_err(|e| allocation(self.runtime, e))?;
        }
        Ok(())
    }
}
fn allocation(runtime: Option<&RuntimeContext>, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:Struct:Allocation",
        "structure operation allocation failed",
        source,
    )
}
#[derive(Debug, thiserror::Error)]
#[error("invalid structure-array operation")]
struct StructureOperationFailure;
fn invalid(message: impl Into<String>) -> RuntimeError {
    crate::runtime_error::semantic_error_with_source(
        "StructArrayOperation",
        message,
        StructureOperationFailure,
    )
}
/// Re-label only this operation's semantic validation. Resource, cancellation,
/// provider and source-owner failures retain their original typed disposition.
pub fn remap(error: RuntimeError, map: impl FnOnce(&str) -> RuntimeError) -> RuntimeError {
    if std::error::Error::source(&error)
        .is_some_and(|source| source.is::<StructureOperationFailure>())
    {
        let mut mapped = map(error.message());
        mapped.source = Some(Box::new(error));
        mapped
    } else {
        error
    }
}
fn count(shape: &[usize]) -> Result<usize, RuntimeError> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| invalid("structure dimensions overflow"))
}
fn raw(value: RootedValue) -> Value {
    value.into_value_and_lease().0
}
fn array_result(value: RootedValue) -> Result<StructArray, RuntimeError> {
    match raw(value) {
        Value::StructArray(array) => Ok(array),
        _ => Err(invalid("operation unexpectedly normalized a scalar")),
    }
}

/// Caller-supplied indices have already been quoted by their indexing planner.
pub fn select(
    array: &StructArray,
    indices: &[usize],
    shape: &[usize],
) -> Result<Value, RuntimeError> {
    if count(shape)? != indices.len() {
        return Err(invalid("structure selection index or shape is invalid"));
    }
    select_by_index(array, shape, |index| indices.get(index).copied())
}
pub fn select_by_index(
    array: &StructArray,
    shape: &[usize],
    source: impl Fn(usize) -> Option<usize>,
) -> Result<Value, RuntimeError> {
    let runtime = crate::context::legacy::active();
    select_by_index_in_context(runtime.as_ref(), array, shape, source)
}

pub(crate) fn select_by_index_in_context(
    runtime: Option<&RuntimeContext>,
    array: &StructArray,
    shape: &[usize],
    source: impl Fn(usize) -> Option<usize>,
) -> Result<Value, RuntimeError> {
    for index in 0..count(shape)? {
        if source(index).is_none_or(|index| index >= array.len()) {
            return Err(invalid("structure selection index or shape is invalid"));
        }
    }
    let original = Value::StructArray(array.clone());
    let _roots = replacement::lease(&original, runtime)?;
    construct_generated_structure(runtime, array.field_names(), shape, |name, index| {
        array
            .field_values(name)
            .and_then(|values| source(index).and_then(|index| values.get(index)))
            .ok_or_else(|| invalid("structure selection lost a field"))
    })
    .map(raw)
}

pub fn replicate_scalar(value: &StructValue, shape: &[usize]) -> Result<Value, RuntimeError> {
    let runtime = crate::context::legacy::active();
    replicate_scalar_in_context(runtime.as_ref(), value, shape)
}

pub(crate) fn replicate_scalar_in_context(
    runtime: Option<&RuntimeContext>,
    value: &StructValue,
    shape: &[usize],
) -> Result<Value, RuntimeError> {
    let original = Value::Struct(value.clone());
    let _roots = replacement::lease(&original, runtime)?;
    construct_generated_structure(runtime, value.field_names(), shape, |name, _| {
        value
            .fields
            .get(name)
            .ok_or_else(|| invalid("structure replication lost a field"))
    })
    .map(raw)
}
pub fn remove(
    array: &StructArray,
    removed: &std::collections::HashSet<usize>,
    shape: &[usize],
) -> Result<Value, RuntimeError> {
    if removed.iter().any(|&index| index >= array.len())
        || array.len().checked_sub(removed.len()) != Some(count(shape)?)
    {
        return Err(invalid(
            "structure deletion index or output shape is invalid",
        ));
    }
    let runtime = crate::context::legacy::active();
    let mut indices = Scratch::new(runtime.as_ref(), count(shape)?)?;
    for (slot, index) in indices
        .values_mut()
        .iter_mut()
        .zip((0..array.len()).filter(|index| !removed.contains(index)))
    {
        *slot = index;
    }
    indices.checkpoint()?;
    select(array, indices.values(), shape)
}
pub fn empty_like(base: &Value, shape: &[usize]) -> Result<Value, RuntimeError> {
    let runtime = crate::context::legacy::active();
    empty_like_in_context(runtime.as_ref(), base, shape)
}

pub(crate) fn empty_like_in_context(
    runtime: Option<&RuntimeContext>,
    base: &Value,
    shape: &[usize],
) -> Result<Value, RuntimeError> {
    if count(shape)? != 0 {
        return Err(invalid(
            "empty structure destination must have zero elements",
        ));
    }
    let _roots = replacement::lease(base, runtime)?;
    let value = |_: &str, _: usize| Err(invalid("empty structure initializer was called"));
    match base {
        Value::Struct(base) => {
            construct_generated_structure(runtime, base.field_names(), shape, value).map(raw)
        }
        Value::StructArray(base) => {
            construct_generated_structure(runtime, base.field_names(), shape, value).map(raw)
        }
        _ => Err(invalid("structure value expected")),
    }
}

/// Project directly from sorted unique source removals. No survivor table or
/// field/payload copy is needed before the one canonical final constructor.
pub(crate) fn remove_sorted_in_context(
    runtime: Option<&RuntimeContext>,
    array: &StructArray,
    removed: &[usize],
    shape: &[usize],
) -> Result<Value, RuntimeError> {
    if removed.last().is_some_and(|&index| index >= array.len())
        || removed.windows(2).any(|pair| pair[0] >= pair[1])
        || array.len().checked_sub(removed.len()) != Some(count(shape)?)
    {
        return Err(invalid(
            "structure deletion index or output shape is invalid",
        ));
    }
    select_by_index_in_context(runtime, array, shape, |ordinal| {
        // removed[i] - i is the number of retained elements before that
        // removal. Rank-select is logarithmic and repeatable across fields.
        let mut low = 0;
        let mut high = removed.len();
        while low < high {
            let middle = low + (high - low) / 2;
            if removed[middle] - middle <= ordinal {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        ordinal.checked_add(low)
    })
}
/// Owned structural projection for transforms that preserve GC identities
/// (for example string-to-character and missing-value classification). The
/// callback may replace payloads, but MUST NOT create previously unrooted GC
/// identities. Original descendant roots cover the entire recursive walk.
pub fn map_identity_preserving(
    source: &StructArray,
    mut transform: impl FnMut(Value) -> Result<Value, RuntimeError>,
) -> Result<StructArray, RuntimeError> {
    let runtime = crate::context::legacy::active();
    let owner = HostAllocationReceipt::capture_owner(runtime.as_ref())
        .map_err(|source| allocation(runtime.as_ref(), source))?;
    let prepared = replacement::array(source, runtime.as_ref())?;
    let _roots = prepared.1;
    let mut output = prepared.0;
    output
        .try_for_each_value_mut_unique(|slot| {
            let value = std::mem::replace(slot, Value::Num(0.0));
            *slot = transform(value)?;
            Ok::<_, RuntimeError>(())
        })
        .expect("explicit admitted unique columns")?;
    if let Some(owner) = owner {
        owner
            .check(runtime.as_ref().expect("managed context"))
            .map_err(|source| allocation(runtime.as_ref(), source))?;
    }
    Ok(output)
}
fn reindex(
    array: &StructArray,
    shape: &[usize],
    index: impl Fn(usize) -> Result<usize, RuntimeError>,
) -> Result<StructArray, RuntimeError> {
    let runtime = crate::context::legacy::active();
    let original = Value::StructArray(array.clone());
    let _roots = replacement::lease(&original, runtime.as_ref())?;
    let output = construct_generated_structure(
        runtime.as_ref(),
        array.field_names(),
        shape,
        |name, destination| {
            let source = index(destination)?;
            array
                .field_values(name)
                .and_then(|column| column.get(source))
                .ok_or_else(|| invalid("structure reorder index is out of bounds"))
        },
    )?;
    array_result(output)
}
fn shape_owner(
    runtime: Option<&RuntimeContext>,
    shape: &[usize],
) -> Result<SharedShape, RuntimeError> {
    let bytes = SharedShape::requested_bytes_for_rank(shape.len())
        .map_err(|e| allocation(runtime, e.into()))?;
    let bytes =
        u64::try_from(bytes).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?;
    let bytes = HostAllocationPlan::retained_exact_output_bytes(bytes, 1)
        .map_err(|e| allocation(runtime, e))?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(|e| allocation(runtime, e))?;
    let receipt =
        HostAllocationReceipt::admit(runtime, plan).map_err(|e| allocation(runtime, e))?;
    match receipt {
        Some(receipt) => {
            let runtime = runtime.expect("managed context");
            let permit = receipt
                .output_permit(runtime)
                .map_err(|e| allocation(Some(runtime), e))?;
            let shape = SharedShape::new_admitted_from_slice(shape, &permit);
            receipt
                .checkpoint(runtime)
                .map_err(|e| allocation(Some(runtime), e))?;
            shape.map_err(|e| allocation(Some(runtime), e.into()))
        }
        None => Ok(shape.to_vec().into()),
    }
}
