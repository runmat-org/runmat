//! Canonical selector planning composed back to the same original definition.
use super::super::common::ScalarCooperation;
use super::*;
use crate::object::indexing::{
    ObjectIndexComponent as Component, ObjectIndexSelector, ObjectSubscript,
};
use runmat_value::optimization_variable::*;

mod named;

pub(super) async fn read(
    view: &View,
    step: &ObjectSubscript,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let ObjectIndexSelector::IndexValues { components } = step.selector() else {
        return Err(invalid(
            "variable parentheses require ordinary index selectors",
        ));
    };
    let mut selected = runmat_value::exact_capacity_vec(components.len()).map_err(data_error)?;
    for (axis, component) in components.iter().enumerate() {
        let output = match component {
            Component::Colon => Component::Colon,
            Component::End => Component::End,
            Component::Value(value) => {
                if named::is_name_selector(value) {
                    let axis = named::axis(view, components.len(), axis)?;
                    Component::Value(named::indices(view, axis, value, cooperation).await?)
                } else {
                    if !matches!(
                        value,
                        Value::Num(_)
                            | Value::Int(_)
                            | Value::Bool(_)
                            | Value::Tensor(_)
                            | Value::LogicalArray(_)
                            | Value::SparseTensor(_)
                            | Value::Complex(_, _)
                            | Value::GpuTensor(_)
                    ) {
                        return Err(invalid(
                            "indices must be numeric, logical or named selectors",
                        ));
                    }
                    Component::Value(value.clone())
                }
            }
        };
        selected.push(output);
        cooperation.completed(1).await?;
    }
    let plan = crate::object::indexing::plan_for_virtual_components_with_context(
        cooperation.runtime(),
        &**view.selection().shape(),
        &selected,
    )
    .await?;
    let n = plan.indices.len();
    let direct = view.selection().is_whole();
    let mut whole = n == view.definition().element_count();
    let mut indices = if direct || whole {
        None
    } else {
        Some(
            crate::context::host_allocation::prepare_host_array_optional(cooperation.runtime(), n)?,
        )
    };
    for (position, &index) in plan.indices.iter().enumerate() {
        let original = view
            .selection()
            .definition_index(index)
            .ok_or_else(|| invalid("index exceeds the current variable reference"))?;
        whole &= original == position;
        if !direct && indices.is_none() && original != position {
            let mut mapped = crate::context::host_allocation::prepare_host_array_optional(
                cooperation.runtime(),
                n,
            )?;
            for prefix in 0..position {
                mapped
                    .try_push(prefix)
                    .map_err(|_| invalid("variable index extent changed"))?;
                cooperation.completed(1).await?;
            }
            indices = Some(mapped);
        }
        if let Some(indices) = &mut indices {
            indices
                .try_push(original)
                .map_err(|_| invalid("variable index extent changed"))?;
        }
        cooperation.completed(1).await?;
    }
    let rank = plan.output_shape.len();
    let mut sources =
        crate::context::host_allocation::prepare_host_array_optional(cooperation.runtime(), rank)?;
    let vector_axis = if components.len() == 1
        && view.selection().shape().len() == 2
        && view.selection().shape().iter().filter(|&&n| n > 1).count() <= 1
    {
        let original = named::vector_axis(view);
        let output = plan
            .output_shape
            .iter()
            .position(|&n| n > 1)
            .unwrap_or(original.min(rank - 1));
        Some((original, output))
    } else {
        None
    };
    for axis in 0..rank {
        let source = if components.len() > 1
            && axis < components.len()
            && (axis + 1 < components.len() || components.len() >= view.selection().shape().len())
        {
            view.selection()
                .axis_sources()
                .get(axis)
                .copied()
                .unwrap_or(OptimizationVariableAxisSource::Unnamed)
        } else if let Some((original, _)) = vector_axis.filter(|(_, output)| *output == axis) {
            view.selection().axis_sources()[original]
        } else {
            OptimizationVariableAxisSource::Unnamed
        };
        sources
            .try_push(source)
            .map_err(|_| invalid("variable axis-source rank changed"))?;
        cooperation.completed(1).await?;
    }
    let candidate = OptimizationVariableSelectionCandidate {
        definition: view.selection().definition(),
        definition_elements: view.definition().element_count(),
        role: OptimizationVariableReferenceRole::Reference,
        indices: if direct {
            if whole {
                OptimizationVariableIndices::Whole
            } else {
                OptimizationVariableIndices::Indexed(plan.indices)
            }
        } else {
            match indices {
                None => OptimizationVariableIndices::Whole,
                Some(indices) => OptimizationVariableIndices::Indexed(
                    indices
                        .try_finish()
                        .map_err(|_| invalid("variable index extent changed"))?,
                ),
            }
        },
        shape: plan.output_shape,
        axis_sources: sources
            .try_finish()
            .map_err(|_| invalid("variable axis-source rank changed"))?,
    };
    super::construction::publish_reference(candidate, cooperation).await
}
