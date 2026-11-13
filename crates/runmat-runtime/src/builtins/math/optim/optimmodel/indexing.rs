//! Real expression views and immutable element assignment. All numeric/logical
//! selector geometry is supplied by the existing canonical index planner.
use super::super::common::model_indexing as named;
use super::*;
use crate::object::indexing::{
    ObjectIndexComponent as Component, ObjectIndexSelector, ObjectSubscript,
};

impl named::Axes for Node {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn has_names(&self, axis: usize) -> bool {
        self.index_names.get(axis).is_some_and(Option::is_some)
    }
    fn name(&self, axis: usize, position: usize) -> Option<&Utf8Text> {
        self.index_names.get(axis)?.as_ref()?.get(position)
    }
}
async fn plan(
    node: &Node,
    step: &ObjectSubscript,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(crate::indexing::plan::IndexPlan<usize>, usize)> {
    let ObjectIndexSelector::IndexValues { components } = step.selector() else {
        return Err(policy.invalid("expression parentheses require ordinary index selectors"));
    };
    let mut selected = runmat_value::exact_capacity_vec(components.len())
        .map_err(|source| policy.storage(source))?;
    for (axis, component) in components.iter().enumerate() {
        let value = match component {
            Component::Colon => Component::Colon,
            Component::End => Component::End,
            Component::Value(value) if named::is_name_selector(value) => {
                let axis = named::axis(node, components.len(), axis, policy)?;
                Component::Value(named::indices(node, axis, value, cooperation, policy).await?)
            }
            Component::Value(value) => {
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
                    return Err(policy.invalid(
                        "expression indices must be numeric, logical or named selectors",
                    ));
                }
                Component::Value(value.clone())
            }
        };
        selected.push(value);
        cooperation.completed(1).await?;
    }
    let plan = crate::object::indexing::plan_for_virtual_components_with_context(
        cooperation.runtime(),
        &*node.shape,
        &selected,
    )
    .await?;
    Ok((plan, components.len()))
}
pub(super) async fn read(
    value: &Value,
    step: &ObjectSubscript,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let source = owner::reference(value, policy)?;
    let owned = owner::OwnedNode::read(source, policy)?;
    let (plan, count) = plan(owned.node(), step, policy, cooperation).await?;
    let index_names = project_names(owned.node(), &plan, count, policy, cooperation).await?;
    let selected = plan.indices;
    owner::publish(
        Node {
            shape: plan.output_shape,
            index_names,
            kind: Kind::Index {
                source,
                indices: selected,
            },
        },
        policy,
        cooperation,
    )
    .await
}
pub(super) async fn assign(
    value: &Value,
    step: &ObjectSubscript,
    rhs: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let source = owner::reference(value, policy)?;
    let owned = owner::OwnedNode::read(source, policy)?;
    let (plan, _) = plan(owned.node(), step, policy, cooperation).await?;
    let replacement_root = owner::as_expression(rhs, policy, cooperation).await?;
    let replacement = owner::reference(replacement_root.value(), policy)?;
    let rhs = owner::OwnedNode::read(replacement, policy)?;
    let rhs_count = super::evaluation::element_count(&rhs.node().shape, policy)?;
    let _layout = crate::indexing::write_slice::AssignmentRhsLayout::multidimensional(
        &rhs.node().shape,
        rhs_count,
        &plan.selection_lengths,
    )?;
    let selected = plan.indices;
    let index_names = owned.node().index_names.clone();
    owner::publish(
        Node {
            shape: owned.node().shape.clone(),
            index_names,
            kind: Kind::Assign {
                source,
                indices: selected,
                replacement,
                selection_shape: plan.selection_lengths,
            },
        },
        policy,
        cooperation,
    )
    .await
}
async fn project_names(
    node: &Node,
    plan: &crate::indexing::plan::IndexPlan<usize>,
    count: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<runmat_value::optimization_model::ModelAxisNames> {
    let rank = plan.output_shape.len();
    let mut result =
        crate::context::host_allocation::prepare_host_array_optional(cooperation.runtime(), rank)?;
    for axis in 0..rank {
        let source = if plan.indices.is_empty() {
            None
        } else if count > 1 && axis < count && (axis + 1 < count || count >= node.shape.len()) {
            Some(axis)
        } else if count == 1
            && node.shape.len() == 2
            && node.shape.iter().filter(|&&n| n > 1).count() <= 1
        {
            let source = named::vector_axis(node);
            let output_axis = plan
                .output_shape
                .iter()
                .position(|&n| n > 1)
                .unwrap_or(source.min(rank - 1));
            (axis == output_axis).then_some(source)
        } else {
            None
        };
        let Some(source) =
            source.filter(|&source| node.index_names.get(source).is_some_and(Option::is_some))
        else {
            result
                .try_push(None)
                .map_err(|_| policy.invalid("index-name rank changed"))?;
            cooperation.completed(1).await?;
            continue;
        };
        let output_stride = plan
            .output_shape
            .iter()
            .take(axis)
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| policy.invalid("index-name output stride overflow"))?;
        let source_stride = node
            .shape
            .iter()
            .take(source)
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| policy.invalid("index-name source stride overflow"))?;
        let mut labels = crate::context::host_allocation::prepare_host_array_optional(
            cooperation.runtime(),
            plan.output_shape[axis],
        )?;
        for position in 0..plan.output_shape[axis] {
            let ordinal = position
                .checked_mul(output_stride)
                .ok_or_else(|| policy.invalid("index-name output coordinate overflow"))?;
            let index = *plan
                .indices
                .get(ordinal)
                .ok_or_else(|| policy.invalid("index-name output projection is incomplete"))?;
            let coordinate = (index / source_stride) % node.shape[source];
            labels
                .try_push(
                    node.index_names[source]
                        .as_ref()
                        .and_then(|labels| labels.get(coordinate))
                        .ok_or_else(|| {
                            policy.invalid("index-name source projection is incomplete")
                        })?
                        .clone(),
                )
                .map_err(|_| policy.invalid("index-name axis extent changed"))?;
            cooperation.completed(1).await?;
        }
        result
            .try_push(Some(
                labels
                    .try_finish()
                    .map_err(|_| policy.invalid("index-name axis extent changed"))?,
            ))
            .map_err(|_| policy.invalid("index-name rank changed"))?;
    }
    result
        .try_finish()
        .map_err(|_| policy.invalid("index-name rank changed"))
}
