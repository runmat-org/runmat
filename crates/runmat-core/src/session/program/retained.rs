//! Original compiler products retained by successful session publication only.
use super::*;
use std::collections::HashMap;

#[derive(Default)]
pub(super) struct RetainedPrograms {
    owners: HashMap<runmat_types::FunctionId, Rc<ExecutableUnit>>,
}

impl SessionProgramService {
    /// Fixture projection of the actual original compiler owner, not Graph
    /// unit decoding or an invocation grant. Exact origins must be unambiguous.
    #[cfg(test)]
    pub(in crate::session) fn original_for_test(
        &self,
        origin: FunctionBindingOrigin,
    ) -> Option<Rc<ExecutableUnit>> {
        let retained = self.retained.borrow();
        let mut selected = None;
        for owner in retained.owners.values() {
            if owner
                .local_function_for_origin(origin)
                .ok()
                .flatten()
                .is_some()
            {
                if selected
                    .as_ref()
                    .is_some_and(|previous| !Rc::ptr_eq(previous, owner))
                {
                    return None;
                }
                selected = Some(Rc::clone(owner));
            }
        }
        selected
    }

    pub(in crate::session) fn bind_original_dependencies(
        &self,
        unit: ExecutableUnit,
    ) -> Result<ExecutableUnit, String> {
        unit.bind_original_dependencies(&self.retained.borrow().owners)
    }

    pub(in crate::session) fn publish(
        &self,
        unit: &ExecutableUnit,
        registry: &runmat_vm::FunctionRegistry,
    ) {
        let mut retained = self.retained.borrow_mut();
        retained.owners.retain(|id, _| registry.get(*id).is_some());
        let owner = Rc::new(unit.clone());
        for analysis in unit.analysis().functions.iter() {
            let Some(runmat_types::CallableIdentity::BoundFunction(function)) =
                &analysis.callable.identity
            else {
                continue;
            };
            if unit.owns_callable(*function) && registry.get(*function).is_some() {
                retained.owners.insert(*function, owner.clone());
            }
        }
    }

    pub(in crate::session) fn function_modifiers(
        &self,
    ) -> HashMap<runmat_hir::FunctionId, runmat_hir::FunctionModifiers> {
        self.retained
            .borrow()
            .owners
            .iter()
            .filter_map(|(function, owner)| {
                let id = *function;
                owner
                    .mir()
                    .functions
                    .get(&id)
                    .map(|metadata| (id, metadata.modifiers.clone()))
            })
            .collect()
    }

    pub(super) fn resolve<'a>(
        &self,
        caller: &'a ExecutableUnit,
        origin: FunctionBindingOrigin,
    ) -> Result<&'a ExecutableUnit, runmat_runtime::RuntimeError> {
        if caller
            .local_function_for_origin(origin)
            .map_err(program_error)?
            .is_some()
        {
            return Ok(caller);
        }
        let mut selected: Option<&ExecutableUnit> = None;
        for owner in caller.original_dependencies().values() {
            if owner
                .local_function_for_origin(origin)
                .map_err(program_error)?
                .is_some()
            {
                if let Some(previous) = selected {
                    if !std::ptr::eq(previous, owner.as_ref()) {
                        return Err(program_error(
                            "callable origin has ambiguous original products",
                        ));
                    }
                }
                selected = Some(owner.as_ref());
            }
        }
        selected.ok_or_else(|| {
            program_error("callable origin has no retained original compiler product")
        })
    }
}
