use super::{mex, zero_based_indices};
use crate::indexing::plan::{effective_index_shape, IndexPlan};
use crate::indexing::shape::column_major_strides;
use crate::RuntimeError;
use runmat_value::{StructArray, Value};
use std::collections::HashSet;

pub(super) fn delete_with_plan(base: Value, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    let (schema, shape) = match &base {
        Value::Struct(structure) => (
            structure.field_names().cloned().collect::<Vec<_>>(),
            vec![1, 1],
        ),
        Value::StructArray(array) => (
            array.field_names().cloned().collect::<Vec<_>>(),
            array.shape().to_vec(),
        ),
        _ => {
            return Err(mex(
                "StructAssignmentTypeMismatch",
                "Structure-array deletion requires a structure target",
            ))
        }
    };
    if plan.indices.is_empty() {
        return Ok(base);
    }
    let removal = zero_based_indices(plan).into_iter().collect::<HashSet<_>>();
    if plan.dims == 1 {
        return delete_linear(base, schema, shape, removal);
    }

    let effective_shape = effective_index_shape(&shape, plan.dims)?;
    let coordinate_sets = selected_coordinates(&removal, &effective_shape)?;
    let cartesian_count = coordinate_sets
        .iter()
        .try_fold(1usize, |total, coordinates| {
            total
                .checked_mul(coordinates.len())
                .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))
        })?;
    if cartesian_count != removal.len() {
        return Err(unsupported(
            "Structure-array deletion requires a Cartesian subscript selection",
        ));
    }
    let partial = coordinate_sets
        .iter()
        .enumerate()
        .filter(|(dimension, selected)| selected.len() < effective_shape[*dimension])
        .map(|(dimension, _)| dimension)
        .collect::<Vec<_>>();
    if partial.is_empty() {
        return StructArray::empty(schema, vec![0, 0])
            .map(Value::StructArray)
            .map_err(|error| mex("StructDeletion", error));
    }
    if partial.len() != 1
        || coordinate_sets
            .iter()
            .enumerate()
            .any(|(dimension, selected)| {
                dimension != partial[0]
                    && (selected.len() != effective_shape[dimension]
                        || !(0..effective_shape[dimension]).all(|index| selected.contains(&index)))
            })
    {
        return Err(unsupported(
            "Structure-array deletion requires one indexed dimension and colons in the others",
        ));
    }
    let removed_dimension = partial[0];
    let removed_coordinates = &coordinate_sets[removed_dimension];
    let strides = column_major_strides(&effective_shape)?;
    let removed_extent = effective_shape[removed_dimension];
    let mut output_shape = effective_shape;
    output_shape[removed_dimension] -= removed_coordinates.len();
    output_shape.resize(output_shape.len().max(2), 1);
    let linear_removal = (0..checked_shape_len(&shape)?)
        .filter(|linear| {
            let coordinate = linear / strides[removed_dimension] % removed_extent;
            removed_coordinates.contains(&coordinate)
        })
        .collect::<HashSet<_>>();
    remove_from_base(base, schema, &linear_removal, output_shape)
}

fn delete_linear(
    base: Value,
    schema: Vec<String>,
    shape: Vec<usize>,
    removal: HashSet<usize>,
) -> Result<Value, RuntimeError> {
    let vector =
        shape.first().copied().unwrap_or(1) == 1 || shape.iter().skip(1).all(|extent| *extent == 1);
    if !vector {
        return Err(unsupported(
            "Linear structure-array deletion requires a vector",
        ));
    }
    let original_len = checked_shape_len(&shape)?;
    let retained_len = original_len.saturating_sub(removal.len());
    let output_shape = if retained_len == 0 && original_len <= 1 {
        vec![0, 0]
    } else if shape.first().copied().unwrap_or(1) == 1 {
        vec![1, retained_len]
    } else {
        vec![retained_len, 1]
    };
    remove_from_base(base, schema, &removal, output_shape)
}

fn remove_from_base(
    base: Value,
    schema: Vec<String>,
    removal: &HashSet<usize>,
    output_shape: Vec<usize>,
) -> Result<Value, RuntimeError> {
    match base {
        Value::Struct(_) => StructArray::empty(schema, output_shape).map(Value::StructArray),
        Value::StructArray(array) => array.remove_linear(removal, output_shape),
        _ => unreachable!("deletion admission checked the structure target"),
    }
    .map_err(|error| mex("StructDeletion", error))
}

fn checked_shape_len(shape: &[usize]) -> Result<usize, RuntimeError> {
    shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| mex("IndexOutOfBounds", "Index dimensions overflow"))
    })
}

fn selected_coordinates(
    removal: &HashSet<usize>,
    shape: &[usize],
) -> Result<Vec<HashSet<usize>>, RuntimeError> {
    if shape.contains(&0) && !removal.is_empty() {
        return Err(mex(
            "IndexOutOfBounds",
            "Cannot remove elements from an empty structure array",
        ));
    }
    let strides = column_major_strides(shape)?;
    let mut selected = vec![HashSet::new(); shape.len()];
    for &linear in removal {
        for dimension in 0..shape.len() {
            selected[dimension].insert(linear / strides[dimension] % shape[dimension]);
        }
    }
    Ok(selected)
}

fn unsupported(message: &str) -> RuntimeError {
    mex("UnsupportedStructDeletion", message)
}
