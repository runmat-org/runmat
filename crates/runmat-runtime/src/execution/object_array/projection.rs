//! Identity-preserving projections into admitted successor slot generations.
use super::*;
/// Identity-preserving successor slots; source roots cover borrowed children
/// until the successor has its own published registration.
pub(crate) fn copy_source(
    source: &ObjectArray,
    runtime: Option<&RuntimeContext>,
) -> Result<RootedValue, RuntimeError> {
    let _roots = crate::execution::structure::replacement::lease(
        &Value::ObjectArray(source.clone()),
        runtime,
    )?;
    let mut output = ObjectArrayConstruction::new_retaining_shape(
        runtime,
        Class::Owner(source.class_name_owner()),
        source.len(),
        source.shape_owner(),
        source.selected_provenance().iter(),
        source.has_unselected_provenance(),
    )?;
    for value in source.data() {
        output.push(value.clone())?;
    }
    output.finish()
}

pub(crate) fn select(
    source: &ObjectArray,
    indices: impl Iterator<Item = usize> + Clone,
    shape: &[usize],
    runtime: Option<&RuntimeContext>,
) -> Result<RootedValue, RuntimeError> {
    let _roots = crate::execution::structure::replacement::lease(
        &Value::ObjectArray(source.clone()),
        runtime,
    )?;
    let count = indices.clone().count();
    let capacity = if count == 0 {
        source
            .len()
            .checked_add(source.selected_provenance().len())
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?
    } else {
        0
    };
    let mut output = ObjectArrayConstruction::with_capacity(
        runtime,
        Class::Owner(source.class_name_owner()),
        count,
        shape,
        capacity,
        source.has_unselected_provenance(),
    )?;
    if count == 0 {
        for facts in source.selected_provenance() {
            output.push_provenance(facts.clone())?;
        }
        for child in source.data() {
            output.checkpoint()?;
            match crate::object::protocol::selected_receiver_facts(child)? {
                Some(facts) => output.push_provenance(facts)?,
                None => output.mark_unselected(),
            }
        }
    }
    for index in indices {
        let child = source.data().get(index).ok_or_else(|| {
            crate::runtime_error::semantic_error(
                "IndexOutOfBounds",
                "Object array index is out of bounds",
            )
        })?;
        output.push(child.clone())?;
    }
    output.finish()
}
