//! Private selected child for a captured initializing structure path. Never
//! construct a grown root merely to read the few elements being assigned.
use super::assignment::{empty, field, names, shape, source_index};
use super::*;
use crate::indexing::plan::IndexPlan;

pub(crate) enum SelectedStructureElement {
    Multiple,
    Existing(usize),
    New,
}

/// The same captured geometry used by projection, not a default field's value.
pub(crate) fn selected_structure_element(
    base: &Value,
    plan: &IndexPlan,
) -> Result<SelectedStructureElement, RuntimeError> {
    let [destination] = plan.indices.as_slice() else {
        return Ok(SelectedStructureElement::Multiple);
    };
    match source_index(*destination as usize, shape(base)?, &plan.base_shape)? {
        Some(source) => Ok(SelectedStructureElement::Existing(source)),
        None => Ok(SelectedStructureElement::New),
    }
}

pub(crate) fn project_initializing_in_context(
    runtime: Option<&RuntimeContext>,
    base: &Value,
    plan: &IndexPlan,
) -> Result<RootedValue, RuntimeError> {
    let owner = crate::indexing::plan::admission::Owner::new(runtime)?;
    let source_shape = shape(base)?;
    let _roots = replacement::lease(base, runtime)?;
    let mut needs_default = false;
    for (ordinal, &index) in plan.indices.iter().enumerate() {
        if ordinal % 4096 == 0 {
            owner.check()?;
        }
        needs_default |= source_index(index as usize, source_shape, &plan.base_shape)?.is_none();
    }
    let default = needs_default.then(|| empty(runtime)).transpose()?;
    let result = construct_generated_structure(
        runtime,
        names(base),
        &plan.output_shape,
        |name, ordinal| {
            let destination = *plan
                .indices
                .get(ordinal)
                .ok_or_else(|| invalid("structure path selection changed"))?
                as usize;
            match source_index(destination, source_shape, &plan.base_shape)? {
                Some(source) => field(base, name, source),
                None => default
                    .as_ref()
                    .ok_or_else(|| invalid("structure growth default is missing")),
            }
        },
    )?;
    owner.check()?;
    Ok(result)
}
