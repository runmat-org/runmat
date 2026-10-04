//! Shared object subscript planning over an already known array shape.

use super::{ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript};
use crate::RuntimeError;

#[derive(Clone, Copy)]
enum PlanMode {
    Read,
    Assignment,
    FixedAssignment,
}

/// Assignment selection for a fixed-size backing. Index admission uses the
/// existing read bounds; result adaptation retains assignment shape rules.
pub async fn plan_for_fixed_assignment(
    shape: &[usize],
    step: &ObjectSubscript,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let ObjectIndexSelector::IndexValues { components } = step.selector() else {
        return Err(crate::runtime_error::semantic_error(
            "InvalidObjectIndex",
            "index selector must contain ordinary indexing values",
        ));
    };
    plan_for_mode(shape, components, PlanMode::FixedAssignment).await
}

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
    let mode = if assignment {
        PlanMode::Assignment
    } else {
        PlanMode::Read
    };
    plan_for_mode(shape, components, mode).await
}

async fn plan_for_mode(
    shape: &[usize],
    components: &[ObjectIndexComponent],
    mode: PlanMode,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let selectors = crate::indexing::selectors::build_component_selectors(
        components,
        shape,
        matches!(mode, PlanMode::Assignment),
    )
    .await?;
    if matches!(mode, PlanMode::Assignment | PlanMode::FixedAssignment) {
        crate::indexing::plan::build_assignment_plan(&selectors, components.len(), shape)
    } else {
        crate::indexing::plan::build_index_plan(&selectors, components.len(), shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::indexing::ObjectIndexComponent;
    use runmat_value::Value;

    fn index(values: &[f64]) -> ObjectSubscript {
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: values
                .iter()
                .map(|value| ObjectIndexComponent::Value(Value::Num(*value)))
                .collect(),
        })
    }

    #[test]
    fn fixed_assignment_rejects_linear_growth_while_ordinary_assignment_preserves_it() {
        let step = index(&[3.0]);
        assert!(futures::executor::block_on(plan_for_fixed_assignment(&[1, 2], &step)).is_err());
        let grown = futures::executor::block_on(plan_for_shape(&[1, 2], &step, true)).unwrap();
        assert_eq!(grown.indices, vec![2]);
    }

    #[test]
    fn fixed_assignment_rejects_dimension_growth_even_when_linear_offset_fits() {
        let step = index(&[3.0, 1.0]);
        assert!(futures::executor::block_on(plan_for_fixed_assignment(&[2, 2], &step)).is_err());
    }

    #[test]
    fn fixed_assignment_preserves_valid_column_major_selection() {
        let planned =
            futures::executor::block_on(plan_for_fixed_assignment(&[2, 3], &index(&[2.0, 3.0])))
                .unwrap();
        assert_eq!(planned.indices, vec![5]);
        assert_eq!(planned.output_shape, vec![1, 1]);
    }
}
