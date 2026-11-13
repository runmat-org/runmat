//! One admitted metadata index of original variable owners and definition edges.
use super::*;
use runmat_value::optimization_model::{ExistingOptimizationNativeInput, OptimizationNativeEdge};
use runmat_value::optimization_variable::OptimizationVariablePayload as Variable;
#[derive(Default)]
pub(super) struct Seen {
    pub walked: bool,
    pub quoted: bool,
}
pub(super) fn insert_object(
    object: &ObjectInstance,
    cursors: &mut Vector<OptimizationNativeAdmission<ExistingOptimizationNativeInput>>,
    seen: &mut Map<SharedStorageIdentity, Seen>,
    storage: &Collections,
    policy: Policy,
) -> BuiltinResult<()> {
    validate(object, policy)?;
    let id = object
        .optimization_native_storage_identity()
        .ok_or_else(|| policy.invalid("native variable owner is absent"))?;
    if seen.get(&id).is_some_and(|entry| entry.walked) {
        return Ok(());
    }
    let additional = usize::from(!seen.contains_key(&id));
    storage.reserve_map(seen, additional)?;
    seen.entry(id).or_default().walked = true;
    storage.reserve_vector(cursors, 1)?;
    cursors.push(ExistingOptimizationNativeInput::new(object.clone()).begin_admission());
    Ok(())
}
fn validate(object: &ObjectInstance, policy: Policy) -> BuiltinResult<()> {
    if !object
        .class_name()
        .is(runmat_types::standard::OPTIMIZATION_VARIABLE)
        || object.selected_class().is_some()
        || object.optimization_variable_payload().is_none()
    {
        return Err(policy.invalid("variable metadata is not an original native class owner"));
    }
    Ok(())
}
pub(super) fn read(edge: OptimizationNativeEdge, policy: Policy) -> BuiltinResult<ObjectInstance> {
    // The real child-root lease protects this handle. No Runtime call occurs
    // inside the checked GC borrow; only the immutable generation is aliased.
    let object = runmat_gc::gc_with_value(&edge.handle, |value| match value {
        Value::Object(object) => Some(object.clone()),
        _ => None,
    })
    .map_err(|source| policy.gc("failed to read variable definition", source))?
    .ok_or_else(|| policy.invalid("variable edge does not identify an object"))?;
    validate(&object, policy)?;
    if !matches!(
        object.optimization_variable_payload(),
        Some(Variable::Definition(_))
    ) {
        return Err(policy.invalid("variable edge does not identify its original definition"));
    }
    Ok(object)
}
pub(super) async fn append_children(
    input: &impl OptimizationNativeSource,
    edges: &mut Vector<OptimizationNativeEdge>,
    storage: &Collections,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let mut index = 0;
    while let Some(edge) = input.edge_at(index) {
        storage.reserve_vector(edges, 1)?;
        edges.push(edge);
        index += 1;
        cooperation.completed(1).await?;
        storage.check()?;
    }
    Ok(())
}
