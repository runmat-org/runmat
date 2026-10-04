//! Shared object subscript planning over an already known array shape.

use super::{ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript};
use crate::RuntimeError;

pub async fn plan_for_shape(
    shape: &[usize],
    step: &ObjectSubscript,
    assignment: bool,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let ObjectIndexSelector::IndexValues { components } = step.selector() else {
        return Err(crate::runtime_error::semantic_error(
            "InvalidObjectIndex",
            "index selector must contain ordinary indexing values",
        ));
    };
    plan_for_components(shape, components, assignment).await
}

pub async fn plan_for_components(
    shape: &[usize],
    components: &[ObjectIndexComponent],
    assignment: bool,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let selectors =
        crate::indexing::selectors::build_component_selectors(components, shape, assignment)
            .await?;
    if assignment {
        crate::indexing::plan::build_assignment_plan(&selectors, components.len(), shape)
    } else {
        crate::indexing::plan::build_index_plan(&selectors, components.len(), shape)
    }
}
