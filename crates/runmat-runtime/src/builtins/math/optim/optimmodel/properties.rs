//! Reachable actual variable definitions and public model property projections.
use super::*;
pub(super) async fn variable_names(
    view: &super::super::optimvar::View,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<runmat_value::optimization_model::ModelAxisNames> {
    let shape = view.selection().shape();
    let mut axes =
        runmat_value::exact_capacity_vec(shape.len()).map_err(|source| policy.storage(source))?;
    for (axis, &extent) in shape.iter().enumerate() {
        let names =
            if extent > 0 && super::super::optimvar::reference_axis_name(view, axis, 0).is_some() {
                let mut names = runmat_value::exact_capacity_vec(extent)
                    .map_err(|source| policy.storage(source))?;
                for position in 0..extent {
                    names.push(
                        super::super::optimvar::reference_axis_name(view, axis, position)
                            .ok_or_else(|| {
                                policy.invalid("variable label projection has incomplete geometry")
                            })?
                            .clone(),
                    );
                    cooperation.completed(1).await?;
                }
                Some(names)
            } else {
                None
            };
        axes.push(names);
        cooperation.completed(1).await?;
    }
    runmat_value::optimization_model::model_axis_names_unbounded(axes)
        .map_err(|source| policy.storage(source))
}

pub(super) async fn names_value(
    names: &[Option<runmat_value::optimization_model::ModelAxisLabels>],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let mut axes =
        runmat_value::exact_capacity_vec(names.len()).map_err(|source| policy.storage(source))?;
    for axis in names {
        let mut values =
            runmat_value::exact_capacity_vec(axis.as_ref().map_or(0, |axis| axis.len()))
                .map_err(|source| policy.storage(source))?;
        if let Some(axis) = axis {
            for label in axis {
                values.push(Value::string(label.clone()));
                cooperation.completed(1).await?;
            }
        }
        let count = values.len();
        axes.push(Value::Cell(
            runmat_value::CellArray::new(values, usize::from(count > 0), count)
                .map_err(|source| policy.invalid(source))?,
        ));
    }
    let count = axes.len();
    RootedValue::in_optional_context(
        Value::Cell(
            runmat_value::CellArray::new(axes, 1, count)
                .map_err(|source| policy.invalid(source))?,
        ),
        cooperation.runtime(),
    )
}
pub(super) async fn get(
    value: &Value,
    property: runmat_builtins::OptimizationExpressionProperty,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let node = owner::OwnedNode::read(owner::reference(value, policy)?, policy)?;
    match property {
        runmat_builtins::OptimizationExpressionProperty::IndexNames => {
            names_value(&node.node().index_names, policy, cooperation).await
        }
        runmat_builtins::OptimizationExpressionProperty::Variables => {
            variables(value, policy, cooperation).await
        }
    }
}
pub(super) async fn set_names(
    value: &Value,
    rhs: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let source = owner::reference(value, policy)?;
    let node = owner::OwnedNode::read(source, policy)?;
    let index_names = super::super::common::model_dimensions::property(
        rhs,
        &node.node().shape,
        cooperation,
        policy,
    )
    .await?;
    owner::publish(
        Node {
            shape: node.node().shape.clone(),
            index_names,
            kind: Kind::Labels { source },
        },
        policy,
        cooperation,
    )
    .await
}
pub(super) async fn variables(
    value: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    variables_roots(&[owner::reference(value, policy)?], policy, cooperation).await
}
pub(super) async fn variables_roots(
    roots: &[runmat_gc::GcHandle],
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let plan = evaluation::Plan::prepare_roots(roots, policy, cooperation).await?;
    evaluation::validate_variable_names(&plan, policy, cooperation).await?;
    let mut originals = std::collections::BTreeMap::new();
    for node in plan.nodes.values() {
        let Kind::Variable(Value::Object(object)) = &node.node().kind else {
            continue;
        };
        let view = super::super::optimvar::View::new(object, cooperation).await?;
        let name = view.definition().name_owner().clone();
        if let std::collections::btree_map::Entry::Vacant(entry) = originals.entry(name) {
            let original =
                super::super::optimvar::original_definition_reference(&view, cooperation).await?;
            entry.insert(original);
        }
        cooperation.completed(1).await?;
    }
    crate::execution::structure::construct_struct_from_fields_in_optional_context(
        cooperation.runtime(),
        originals
            .iter()
            .map(|(name, value)| (name.as_str(), value.value())),
    )
}
