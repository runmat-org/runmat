use super::mex;
use crate::aggregate::structure::operations as struct_ops;
use crate::context::RuntimeContext;
use crate::indexing::plan::{
    admission::{self, Owner, Scratch},
    effective_index_shape_in_context, IndexPlan,
};
use crate::indexing::shape::column_major_strides_in_context;
use crate::RuntimeError;
use runmat_value::Value;

pub(super) fn delete_with_plan(base: Value, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let context = crate::context::legacy::active();
    delete(context.as_ref(), base, plan)
}

pub fn delete_with_plan_in_context(
    context: Option<&RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if !matches!(rhs, Value::Tensor(value) if value.is_empty())
        && !matches!(rhs, Value::ComplexTensor(value) if value.is_empty())
    {
        return Err(mex(
            "InvalidStructDeletionRhs",
            "Structure-array deletion requires an empty right-hand side",
        ));
    }
    delete(context, base, plan)
}

/// Legacy scalar selectors retain their original plan/error policy, but their
/// actual selector and plan storage use the explicit invocation owner.
pub fn delete_scalar_indices_in_context(
    context: Option<&RuntimeContext>,
    base: Value,
    indices: &[usize],
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    let shape: &[usize] = match &base {
        Value::Struct(_) => &[1, 1],
        Value::StructArray(array) => array.shape(),
        _ => {
            return Err(mex(
                "StructAssignmentTypeMismatch",
                "Structure-array deletion requires a structure target",
            ))
        }
    };
    let owner = Owner::new(context)?;
    let _roots = crate::execution::structure::replacement::lease(&base, context)?;
    let selectors =
        crate::context::host_allocation::initialize_host_array(context, indices.len(), |index| {
            Ok(crate::indexing::selectors::SliceSelector::Scalar(
                indices[index],
            ))
        })?;
    let plan = match context {
        Some(context) => crate::indexing::plan::build_index_plan_in_context(
            context,
            &selectors,
            indices.len(),
            shape,
        )?,
        None => crate::indexing::plan::build_index_plan(&selectors, indices.len(), shape)?,
    };
    owner.check()?;
    let result = delete_with_plan_in_context(context, base, &plan, rhs)?;
    owner.check()?;
    Ok(result)
}

fn delete(
    context: Option<&RuntimeContext>,
    base: Value,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let shape: &[usize] = match &base {
        Value::Struct(_) => &[1, 1],
        Value::StructArray(array) => array.shape(),
        _ => {
            return Err(mex(
                "StructAssignmentTypeMismatch",
                "Structure-array deletion requires a structure target",
            ))
        }
    };
    let owner = Owner::new(context)?;
    if plan.indices.is_empty() {
        return Ok(base);
    }
    let _roots = crate::execution::structure::replacement::lease(&base, context)?;
    let mut removal = Scratch::<usize>::new(context, plan.indices.len())?;
    for (ordinal, (&index, slot)) in plan.indices.iter().zip(removal.as_mut_slice()).enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        *slot = index as usize;
    }
    owner.check()?;
    removal.as_mut_slice().sort_unstable();
    let unique = deduplicate(removal.as_mut_slice(), &owner)?;
    let removed = &removal.as_slice()[..unique];
    if plan.dims == 1 {
        let vector = shape.first().copied().unwrap_or(1) == 1
            || shape.iter().skip(1).all(|extent| *extent == 1);
        if !vector {
            return Err(unsupported(
                "Linear structure-array deletion requires a vector",
            ));
        }
        let original_len = checked_shape_len(shape)?;
        validate_bounds(removed, original_len)?;
        let retained = original_len - unique;
        let output = if retained == 0 && original_len <= 1 {
            [0, 0]
        } else if shape.first().copied().unwrap_or(1) == 1 {
            [1, retained]
        } else {
            [retained, 1]
        };
        let result = remove_from_base(context, &base, removed, &output)?;
        owner.check()?;
        return Ok(result);
    }
    let effective = effective_index_shape_in_context(context, shape, plan.dims)?;
    if effective.contains(&0) {
        return Err(mex(
            "IndexOutOfBounds",
            "Cannot remove elements from an empty structure array",
        ));
    }
    let strides = column_major_strides_in_context(context, &effective)?;
    let original_len = checked_shape_len(shape)?;
    // Reuse one coordinate workspace; no HashSet forest or survivor copy.
    let mut coordinates = Scratch::<usize>::new(context, unique)?;
    let mut cartesian = 1usize;
    let mut partial = 0usize;
    let mut removed_dimension = 0;
    let mut removed_extent = 0;
    for (dimension, &extent) in effective.iter().enumerate() {
        for (ordinal, (&linear, coordinate)) in
            removed.iter().zip(coordinates.as_mut_slice()).enumerate()
        {
            if ordinal % 4096 == 0 {
                owner.check()?;
            }
            *coordinate = linear / strides[dimension] % extent;
        }
        coordinates.as_mut_slice().sort_unstable();
        let selected = deduplicate(coordinates.as_mut_slice(), &owner)?;
        cartesian = cartesian
            .checked_mul(selected)
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))?;
        if selected < extent {
            partial += 1;
            removed_dimension = dimension;
            removed_extent = selected;
        }
    }
    if cartesian != unique {
        return Err(unsupported(
            "Structure-array deletion requires a Cartesian subscript selection",
        ));
    }
    if partial > 1 {
        return Err(unsupported(
            "Structure-array deletion requires one indexed dimension and colons in the others",
        ));
    }
    validate_bounds(removed, original_len)?;
    if partial == 0 {
        let result = struct_ops::empty_like_in_context(context, &base, &[0, 0]).map_err(remap)?;
        owner.check()?;
        return Ok(result);
    }
    let output = admission::shape(context, effective.len().max(2), |dimension| {
        let extent = effective.get(dimension).copied().unwrap_or(1);
        if dimension == removed_dimension {
            extent
                .checked_sub(removed_extent)
                .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))
        } else {
            Ok(extent)
        }
    })?;
    // A unique Cartesian selection with one partial axis and complete other
    // axes is precisely the original linear removal set.
    let result = remove_from_base(context, &base, removed, &output)?;
    owner.check()?;
    Ok(result)
}

fn deduplicate(values: &mut [usize], owner: &Owner<'_>) -> Result<usize, RuntimeError> {
    let mut unique = 0;
    for index in 0..values.len() {
        if index % 4096 == 0 {
            owner.check()?;
        }
        if unique == 0 || values[index] != values[unique - 1] {
            values[unique] = values[index];
            unique += 1;
        }
    }
    owner.check()?;
    Ok(unique)
}

fn remove_from_base(
    context: Option<&RuntimeContext>,
    base: &Value,
    removed: &[usize],
    shape: &[usize],
) -> Result<Value, RuntimeError> {
    match base {
        Value::Struct(_) => {
            if removed != [0] {
                return Err(mex(
                    "IndexOutOfBounds",
                    "Structure-array deletion index is out of bounds",
                ));
            }
            struct_ops::empty_like_in_context(context, base, shape)
        }
        Value::StructArray(array) => {
            struct_ops::remove_sorted_in_context(context, array, removed, shape)
        }
        _ => unreachable!("deletion admission checked the structure target"),
    }
    .map_err(remap)
}

fn remap(error: RuntimeError) -> RuntimeError {
    struct_ops::remap(error, |message| mex("StructDeletion", message))
}
fn validate_bounds(removed: &[usize], original_len: usize) -> Result<(), RuntimeError> {
    if removed.last().is_some_and(|&index| index >= original_len) {
        return Err(mex(
            "IndexOutOfBounds",
            "Structure-array deletion index is out of bounds",
        ));
    }
    Ok(())
}
fn checked_shape_len(shape: &[usize]) -> Result<usize, RuntimeError> {
    shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))
    })
}
fn unsupported(message: &str) -> RuntimeError {
    mex("UnsupportedStructDeletion", message)
}
