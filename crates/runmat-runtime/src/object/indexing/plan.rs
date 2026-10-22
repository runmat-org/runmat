//! Shared object subscript planning over an already known array shape.
use super::{ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript};
use crate::{context::RuntimeContext, RuntimeError};

#[derive(Clone, Copy)]
enum PlanMode {
    Read,
    Assignment,
    FixedAssignment,
}

pub async fn plan_for_fixed_assignment(
    shape: &[usize],
    step: &ObjectSubscript,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let context = crate::context::legacy::active();
    plan_for_step(context.as_ref(), shape, step, PlanMode::FixedAssignment).await
}
pub async fn plan_for_fixed_assignment_in_context(
    context: &RuntimeContext,
    shape: &[usize],
    step: &ObjectSubscript,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    plan_for_step(Some(context), shape, step, PlanMode::FixedAssignment).await
}
pub async fn plan_for_shape(
    shape: &[usize],
    step: &ObjectSubscript,
    assignment: bool,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let context = crate::context::legacy::active();
    plan_for_step(
        context.as_ref(),
        shape,
        step,
        if assignment {
            PlanMode::Assignment
        } else {
            PlanMode::Read
        },
    )
    .await
}
pub async fn plan_for_shape_in_context(
    context: &RuntimeContext,
    shape: &[usize],
    step: &ObjectSubscript,
    assignment: bool,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    plan_for_step(
        Some(context),
        shape,
        step,
        if assignment {
            PlanMode::Assignment
        } else {
            PlanMode::Read
        },
    )
    .await
}
async fn plan_for_step(
    context: Option<&RuntimeContext>,
    shape: &[usize],
    step: &ObjectSubscript,
    mode: PlanMode,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let ObjectIndexSelector::IndexValues { components } = step.selector() else {
        return Err(crate::runtime_error::semantic_error(
            "InvalidObjectIndex",
            "index selector must contain ordinary indexing values",
        ));
    };
    plan_for_mode(context, shape, components, mode).await
}
pub async fn plan_for_components(
    shape: &[usize],
    components: &[ObjectIndexComponent],
    assignment: bool,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let context = crate::context::legacy::active();
    plan_for_components_with_context(context.as_ref(), shape, components, assignment).await
}
pub async fn plan_for_components_in_context(
    context: &RuntimeContext,
    shape: &[usize],
    components: &[ObjectIndexComponent],
    assignment: bool,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    plan_for_components_with_context(Some(context), shape, components, assignment).await
}
pub(crate) async fn plan_for_components_with_context(
    context: Option<&RuntimeContext>,
    shape: &[usize],
    components: &[ObjectIndexComponent],
    assignment: bool,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    plan_for_mode(
        context,
        shape,
        components,
        if assignment {
            PlanMode::Assignment
        } else {
            PlanMode::Read
        },
    )
    .await
}
async fn plan_for_mode(
    context: Option<&RuntimeContext>,
    shape: &[usize],
    components: &[ObjectIndexComponent],
    mode: PlanMode,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(context)?;
    let selectors = owner
        .await_stage(
            crate::indexing::selectors::build_component_selectors_with_context(
                context,
                components,
                shape,
                matches!(mode, PlanMode::Assignment),
            ),
        )
        .await??;
    let assignment = matches!(mode, PlanMode::Assignment | PlanMode::FixedAssignment);
    let plan = match (context, assignment) {
        (Some(context), true) => crate::indexing::plan::build_assignment_plan_in_context(
            context,
            &selectors,
            components.len(),
            shape,
        )?,
        (Some(context), false) => crate::indexing::plan::build_index_plan_in_context(
            context,
            &selectors,
            components.len(),
            shape,
        )?,
        (None, true) => {
            crate::indexing::plan::build_assignment_plan(&selectors, components.len(), shape)?
        }
        (None, false) => {
            crate::indexing::plan::build_index_plan(&selectors, components.len(), shape)?
        }
    };
    owner.check()?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::indexing::ObjectIndexComponent;
    use runmat_value::Value;

    fn index(values: &[f64]) -> ObjectSubscript {
        ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues {
            components: runmat_value::ExactArrayStorage::shared_unbounded(
                values
                    .iter()
                    .map(|value| ObjectIndexComponent::Value(Value::Num(*value)))
                    .collect::<Vec<_>>(),
            ),
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
