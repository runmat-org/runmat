use std::cell::RefCell;
use std::rc::Rc;

use runmat_runtime::context::{RuntimeProgramCallable, RuntimeProgramService};
use runmat_types::ProgramFunctionId;

use crate::ExecutableUnit;

#[derive(Default)]
pub(super) struct SessionProgramService {
    active: RefCell<Vec<ExecutableUnit>>,
}

impl SessionProgramService {
    pub(super) fn enter(self: &Rc<Self>, unit: &ExecutableUnit) -> ActiveProgramGuard {
        self.active.borrow_mut().push(unit.clone());
        ActiveProgramGuard {
            service: Rc::clone(self),
        }
    }
}

impl RuntimeProgramService for SessionProgramService {
    fn freeze_bound_callable(
        &self,
        function: ProgramFunctionId,
        display_name: &str,
    ) -> Result<RuntimeProgramCallable, runmat_runtime::RuntimeError> {
        let active = self.active.borrow();
        let unit = active
            .last()
            .ok_or_else(|| program_error("no executable is active"))?;
        let analysis = unit.function_analysis(function).cloned().ok_or_else(|| {
            program_error(format!(
                "active executable does not contain callable function {}",
                function.0
            ))
        })?;
        let function_index = usize::try_from(function.0)
            .map_err(|_| program_error("callable function identity exceeds this host"))?;
        let bytecode = unit
            .functions()
            .get(runmat_hir::FunctionId(function_index))
            .ok_or_else(|| program_error("active executable callable bytecode is absent"))?;
        if bytecode.display_name != display_name {
            return Err(program_error(format!(
                "callable name `{display_name}` does not match active function `{}`",
                bytecode.display_name
            )));
        }
        let executable_source_digest = unit
            .revision()
            .source_digest
            .parse::<runmat_execution::Digest>()
            .map_err(|error| program_error(format!("invalid executable source digest: {error}")))?;
        let executable = unit
            .portable_envelope_for_function(function)
            .map_err(program_error)?;
        Ok(RuntimeProgramCallable {
            owner_identity: unit.source().owner_identity.clone(),
            display_name: display_name.to_string(),
            source_digest: runmat_execution::Digest::sha256(unit.source().text.as_bytes()),
            executable_source_digest,
            function: analysis,
            executable,
        })
    }
}

pub(super) struct ActiveProgramGuard {
    service: Rc<SessionProgramService>,
}

impl Drop for ActiveProgramGuard {
    fn drop(&mut self) {
        self.service
            .active
            .borrow_mut()
            .pop()
            .expect("active program guard stack is balanced");
    }
}

fn program_error(message: impl Into<String>) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(message.into())
        .with_identifier("RunMat:ProgramProduct")
        .build()
}
