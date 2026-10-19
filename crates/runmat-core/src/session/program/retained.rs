//! Original compiler products retained by successful session publication only.
use super::*;
use std::collections::HashMap;

#[derive(Default)]
pub(super) struct RetainedPrograms {
    owners: HashMap<ProgramFunctionId, Rc<ExecutableUnit>>,
}

impl SessionProgramService {
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
        retained.owners.retain(|id, _| {
            registry
                .get(runmat_hir::FunctionId(id.0 as usize))
                .is_some()
        });
        let owner = Rc::new(unit.clone());
        for analysis in unit.analysis().functions.iter() {
            if unit.owns_callable(analysis.function)
                && registry
                    .get(runmat_hir::FunctionId(analysis.function.0 as usize))
                    .is_some()
            {
                retained.owners.insert(analysis.function, owner.clone());
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
                let id = runmat_hir::FunctionId(function.0 as usize);
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
        function: ProgramFunctionId,
    ) -> Result<&'a ExecutableUnit, runmat_runtime::RuntimeError> {
        if caller.owns_callable(function) {
            return Ok(caller);
        }
        caller
            .original_dependencies()
            .get(&function)
            .map(Rc::as_ref)
            .ok_or_else(|| {
                program_error("semantic function has no retained original compiler product")
            })
    }
}
