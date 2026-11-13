use super::{IndexComponent, IndexSelector};
use crate::context::RuntimeContext;
use crate::indexing::plan::{
    build_assignment_plan, build_assignment_plan_in_context, build_index_plan,
    build_index_plan_in_context, IndexPlan,
};
use crate::indexing::selectors::SliceSelector;
use crate::RuntimeError;
use runmat_value::{SharedShape, Value};

pub(in crate::builtins::structs::core) fn build_plan(
    value: &Value,
    selector: &IndexSelector,
    assignment: bool,
) -> Result<IndexPlan, RuntimeError> {
    let context = crate::context::legacy::active();
    let context = context.as_ref();
    let shape = crate::indexing::value::shape_borrowed(value).ok_or_else(|| {
        crate::runtime_error::semantic_error(
            "InvalidObjectSubscriptBase",
            "value does not support indexed field access",
        )
    })?;
    let shape = shape.as_slice();
    let dimensions = selector.components.len();
    if dimensions == 0 {
        return finish_plan(context, &[], dimensions, shape, assignment);
    }
    let selectors =
        crate::context::host_allocation::initialize_host_array(context, dimensions, |dimension| {
            let bound = crate::indexing::shape::selector_extent(shape, dimensions, dimension)?;
            component_selector(
                context,
                &selector.components[dimension],
                bound,
                dimensions == 1,
                shape,
            )
        })?;
    finish_plan(context, &selectors, dimensions, shape, assignment)
}
fn finish_plan(
    context: Option<&RuntimeContext>,
    selectors: &[SliceSelector],
    dimensions: usize,
    shape: &[usize],
    assignment: bool,
) -> Result<IndexPlan, RuntimeError> {
    match (context, assignment) {
        (Some(context), true) => {
            build_assignment_plan_in_context(context, selectors, dimensions, shape)
        }
        (Some(context), false) => {
            build_index_plan_in_context(context, selectors, dimensions, shape)
        }
        (None, true) => build_assignment_plan(selectors, dimensions, shape),
        (None, false) => build_index_plan(selectors, dimensions, shape),
    }
}

fn component_selector(
    context: Option<&RuntimeContext>,
    component: &IndexComponent,
    bound: usize,
    linear: bool,
    target_shape: &[usize],
) -> Result<SliceSelector, RuntimeError> {
    use crate::indexing::plan::admission;
    match component {
        IndexComponent::Scalar(index) => Ok(SliceSelector::Scalar(*index)),
        IndexComponent::End => Ok(SliceSelector::Scalar(bound)),
        IndexComponent::Vector(indices, shape) => {
            let values = admission::array(context, indices.len(), |index| Ok(indices[index]))?;
            if linear {
                Ok(SliceSelector::LinearIndices {
                    values,
                    output_shape: admit_shape(
                        context,
                        numeric_linear_output_shape(target_shape, shape, indices.len()),
                    )?,
                })
            } else {
                Ok(SliceSelector::Indices(values))
            }
        }
        IndexComponent::LogicalScalar(selected) => {
            if *selected && bound == 0 {
                return Err(crate::runtime_error::semantic_error(
                    "IndexOutOfBounds",
                    "logical index contains a true value outside the selected dimension",
                ));
            }
            let values = admission::array(context, usize::from(*selected), |_| Ok(1))?;
            if linear {
                Ok(SliceSelector::LinearIndices {
                    output_shape: crate::indexing::plan::logical_linear_output_shape_in_context(
                        context,
                        target_shape,
                        &[1, 1],
                        values.len(),
                    )?,
                    values,
                })
            } else {
                Ok(SliceSelector::Indices(values))
            }
        }
        IndexComponent::Logical(mask) => {
            if mask
                .data
                .iter()
                .enumerate()
                .any(|(index, selected)| *selected != 0 && index >= bound)
            {
                return Err(crate::runtime_error::semantic_error(
                    "IndexOutOfBounds",
                    "logical index contains a true value outside the selected dimension",
                ));
            }
            let count = mask.data.iter().filter(|selected| **selected != 0).count();
            let mut source = mask
                .data
                .iter()
                .enumerate()
                .filter_map(|(index, selected)| (*selected != 0).then_some(index + 1));
            let values = admission::array(context, count, |_| {
                source.next().ok_or_else(admission::geometry)
            })?;
            if linear {
                Ok(SliceSelector::LinearIndices {
                    output_shape: crate::indexing::plan::logical_linear_output_shape_in_context(
                        context,
                        target_shape,
                        &mask.shape,
                        values.len(),
                    )?,
                    values,
                })
            } else {
                Ok(SliceSelector::Indices(values))
            }
        }
    }
}
enum SelectionShape<'a> {
    Borrowed(&'a [usize]),
    Vector([usize; 2]),
}
impl SelectionShape<'_> {
    fn as_slice(&self) -> &[usize] {
        match self {
            Self::Borrowed(shape) => shape,
            Self::Vector(shape) => shape,
        }
    }
}
fn admit_shape(
    context: Option<&RuntimeContext>,
    shape: SelectionShape<'_>,
) -> Result<SharedShape, RuntimeError> {
    let shape = shape.as_slice();
    crate::indexing::plan::admission::shape(context, shape.len(), |index| Ok(shape[index]))
}
fn numeric_linear_output_shape<'a>(
    target_shape: &[usize],
    selector_shape: &'a [usize],
    selected_len: usize,
) -> SelectionShape<'a> {
    if !is_vector(selector_shape) {
        return SelectionShape::Borrowed(selector_shape);
    }
    let orientation = if is_vector(target_shape) && target_shape.iter().any(|extent| *extent != 1) {
        target_shape
    } else {
        selector_shape
    };
    resize_vector(orientation, selected_len)
}
fn resize_vector(shape: &[usize], selected_len: usize) -> SelectionShape<'static> {
    SelectionShape::Vector(if shape.first().copied().unwrap_or(1) == 1 {
        [1, selected_len]
    } else {
        [selected_len, 1]
    })
}
fn is_vector(shape: &[usize]) -> bool {
    let rows = shape.first().copied().unwrap_or(1);
    let columns = shape.get(1).copied().unwrap_or(1);
    shape.iter().skip(2).all(|extent| *extent == 1) && (rows == 1 || columns == 1)
}
