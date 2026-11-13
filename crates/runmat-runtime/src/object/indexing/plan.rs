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
pub fn plan_for_fixed_assignment_in_context<'a>(
    context: &'a RuntimeContext,
    shape: &'a [usize],
    step: &'a ObjectSubscript,
) -> impl std::future::Future<Output = Result<crate::indexing::plan::IndexPlan, RuntimeError>> + 'a
{
    plan_for_step(Some(context), shape, step, PlanMode::FixedAssignment)
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
pub fn plan_for_shape_in_context<'a>(
    context: &'a RuntimeContext,
    shape: &'a [usize],
    step: &'a ObjectSubscript,
    assignment: bool,
) -> impl std::future::Future<Output = Result<crate::indexing::plan::IndexPlan, RuntimeError>> + 'a
{
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
pub fn plan_for_components_in_context<'a>(
    context: &'a RuntimeContext,
    shape: &'a [usize],
    components: &'a [ObjectIndexComponent],
    assignment: bool,
) -> impl std::future::Future<Output = Result<crate::indexing::plan::IndexPlan, RuntimeError>> + 'a
{
    plan_for_components_with_context(Some(context), shape, components, assignment)
}
pub(crate) fn plan_for_components_with_context<'a, S: AsRef<[usize]> + 'a>(
    context: Option<&'a RuntimeContext>,
    shape: S,
    components: &'a [ObjectIndexComponent],
    assignment: bool,
) -> impl std::future::Future<Output = Result<crate::indexing::plan::IndexPlan, RuntimeError>> + 'a
{
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
}
pub(crate) fn plan_for_virtual_components_with_context<'a, S: AsRef<[usize]> + 'a>(
    context: Option<&'a RuntimeContext>,
    shape: S,
    components: &'a [ObjectIndexComponent],
) -> impl std::future::Future<Output = Result<crate::indexing::plan::IndexPlan<usize>, RuntimeError>> + 'a
{
    plan_for_mode_with_output(
        context,
        shape,
        components,
        PlanMode::Read,
        |context, selectors, dims, shape, _| {
            crate::indexing::plan::build_virtual_index_plan_with_context(
                context, selectors, dims, shape,
            )
        },
    )
}

fn plan_for_mode<'a, S: AsRef<[usize]> + 'a>(
    context: Option<&'a RuntimeContext>,
    shape: S,
    components: &'a [ObjectIndexComponent],
    mode: PlanMode,
) -> impl std::future::Future<Output = Result<crate::indexing::plan::IndexPlan, RuntimeError>> + 'a
{
    plan_for_mode_with_output(context, shape, components, mode, physical_plan)
}

async fn plan_for_mode_with_output<S: AsRef<[usize]>, I>(
    context: Option<&RuntimeContext>,
    shape: S,
    components: &[ObjectIndexComponent],
    mode: PlanMode,
    output: impl FnOnce(
        Option<&RuntimeContext>,
        &[crate::indexing::selectors::SliceSelector],
        usize,
        &[usize],
        bool,
    ) -> Result<crate::indexing::plan::IndexPlan<I>, RuntimeError>,
) -> Result<crate::indexing::plan::IndexPlan<I>, RuntimeError> {
    // Own either the caller's borrowed dimensions or its existing inline
    // scalar/sparse shape in this original planner, not a forwarding future.
    let shape = shape.as_ref();
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
    let plan = output(context, &selectors, components.len(), shape, assignment)?;
    owner.check()?;
    Ok(plan)
}
fn physical_plan(
    context: Option<&RuntimeContext>,
    selectors: &[crate::indexing::selectors::SliceSelector],
    dims: usize,
    shape: &[usize],
    assignment: bool,
) -> Result<crate::indexing::plan::IndexPlan, RuntimeError> {
    match (context, assignment) {
        (Some(context), true) => {
            crate::indexing::plan::build_assignment_plan_in_context(context, selectors, dims, shape)
        }
        (Some(context), false) => {
            crate::indexing::plan::build_index_plan_in_context(context, selectors, dims, shape)
        }
        (None, true) => crate::indexing::plan::build_assignment_plan(selectors, dims, shape),
        (None, false) => crate::indexing::plan::build_index_plan(selectors, dims, shape),
    }
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
        let step = index(&[2.0, 3.0]);
        let planned =
            futures::executor::block_on(plan_for_fixed_assignment(&[2, 3], &step)).unwrap();
        assert_eq!(planned.indices, vec![5]);
        assert_eq!(planned.output_shape, vec![1, 1]);

        let ObjectIndexSelector::IndexValues { components } = step.selector() else {
            unreachable!("ordinary test indices")
        };
        use crate::builtins::common::shape::ValueDimensions;
        for shape in [
            ValueDimensions::Stored(&[2, 3]),
            ValueDimensions::Inline([2, 3]),
        ] {
            let read = futures::executor::block_on(plan_for_components_with_context(
                None, shape, components, false,
            ))
            .unwrap();
            assert_eq!(read.indices, planned.indices);
            assert_eq!(read.output_shape, planned.output_shape);
        }
    }
}
