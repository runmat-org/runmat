use super::{IndexComponent, IndexSelector};
use crate::indexing::plan::{build_assignment_plan, build_index_plan, IndexPlan};
use crate::indexing::selectors::SliceSelector;
use crate::RuntimeError;
use runmat_value::Value;

pub(in crate::builtins::structs::core) fn build_plan(
    value: &Value,
    selector: &IndexSelector,
    assignment: bool,
) -> Result<IndexPlan, RuntimeError> {
    let shape = crate::indexing::value::shape(value).ok_or_else(|| {
        crate::runtime_error::semantic_error(
            "InvalidObjectSubscriptBase",
            "value does not support indexed field access",
        )
    })?;
    let dimensions = selector.components.len();
    let effective = crate::indexing::plan::effective_index_shape(&shape, dimensions)?;
    let selectors = selector
        .components
        .iter()
        .enumerate()
        .map(|(dimension, component)| {
            let bound = if dimensions == 1 {
                effective
                    .iter()
                    .try_fold(1usize, |length, extent| length.checked_mul(*extent))
                    .ok_or_else(|| {
                        crate::runtime_error::semantic_error(
                            "IndexOutOfBounds",
                            "Index dimensions overflow",
                        )
                    })?
            } else {
                effective[dimension]
            };
            component_selector(component, bound, dimensions == 1, &shape)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if assignment {
        build_assignment_plan(&selectors, dimensions, &shape)
    } else {
        build_index_plan(&selectors, dimensions, &shape)
    }
}

fn component_selector(
    component: &IndexComponent,
    bound: usize,
    linear: bool,
    target_shape: &[usize],
) -> Result<SliceSelector, RuntimeError> {
    match component {
        IndexComponent::Scalar(index) => Ok(SliceSelector::Scalar(*index)),
        IndexComponent::End => Ok(SliceSelector::Scalar(bound)),
        IndexComponent::Vector(indices, shape) if linear => Ok(SliceSelector::LinearIndices {
            values: indices.clone(),
            output_shape: numeric_linear_output_shape(target_shape, shape, indices.len()),
        }),
        IndexComponent::Vector(indices, _) => Ok(SliceSelector::Indices(indices.clone())),
        IndexComponent::Logical(mask) => {
            if mask
                .iter()
                .enumerate()
                .any(|(index, selected)| *selected != 0 && index >= bound)
            {
                return Err(crate::runtime_error::semantic_error(
                    "IndexOutOfBounds",
                    "logical index contains a true value outside the selected dimension",
                ));
            }
            let indices = mask
                .iter()
                .enumerate()
                .filter_map(|(index, selected)| (*selected != 0).then_some(index + 1))
                .collect::<Vec<_>>();
            if linear {
                Ok(SliceSelector::LinearIndices {
                    output_shape: logical_linear_output_shape(target_shape, indices.len()),
                    values: indices,
                })
            } else {
                Ok(SliceSelector::Indices(indices))
            }
        }
    }
}

fn numeric_linear_output_shape(
    target_shape: &[usize],
    selector_shape: &[usize],
    selected_len: usize,
) -> Vec<usize> {
    if is_non_scalar_vector(target_shape) && is_vector(selector_shape) {
        return resize_vector(target_shape, selected_len);
    }
    resize_selection_shape(selector_shape, selected_len)
}

fn logical_linear_output_shape(target_shape: &[usize], selected_len: usize) -> Vec<usize> {
    if is_non_scalar_vector(target_shape) {
        resize_vector(target_shape, selected_len)
    } else {
        vec![selected_len, 1]
    }
}

fn resize_selection_shape(shape: &[usize], selected_len: usize) -> Vec<usize> {
    if is_vector(shape) {
        resize_vector(shape, selected_len)
    } else {
        shape.to_vec()
    }
}

fn resize_vector(shape: &[usize], selected_len: usize) -> Vec<usize> {
    if shape.first().copied().unwrap_or(1) == 1 {
        vec![1, selected_len]
    } else {
        vec![selected_len, 1]
    }
}

fn is_non_scalar_vector(shape: &[usize]) -> bool {
    is_vector(shape) && shape.iter().copied().product::<usize>() != 1
}

fn is_vector(shape: &[usize]) -> bool {
    let rows = shape.first().copied().unwrap_or(1);
    let columns = shape.get(1).copied().unwrap_or(1);
    shape.iter().skip(2).all(|extent| *extent == 1) && (rows == 1 || columns == 1)
}
