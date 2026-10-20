use std::cell::RefCell;
use std::rc::Rc;

mod invocation;
mod retained;

use runmat_execution::FunctionBindingOrigin;
use runmat_runtime::context::{RuntimeProgramCallable, RuntimeProgramService};

use crate::ExecutableUnit;

#[derive(Default)]
pub(super) struct SessionProgramService {
    active: RefCell<Vec<ExecutableUnit>>,
    retained: RefCell<retained::RetainedPrograms>,
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
    fn freeze_executable(
        &self,
        origin: FunctionBindingOrigin,
        invocation_revision: &runmat_execution::ProgramRevision,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, runmat_runtime::RuntimeError> {
        let active = self.active.borrow();
        let caller = active
            .last()
            .ok_or_else(|| program_error("no executable is active"))?;
        if &caller.revision().program_revision != invocation_revision {
            return Err(program_error(
                "invocation revision differs from its active original product",
            ));
        }
        self.resolve(caller, origin)?
            .portable_envelope_for_function(origin.function)
            .map_err(program_error)
    }

    fn freeze_bound_callable(
        &self,
        origin: FunctionBindingOrigin,
        display_name: &str,
    ) -> Result<RuntimeProgramCallable, runmat_runtime::RuntimeError> {
        let active = self.active.borrow();
        let caller = active
            .last()
            .ok_or_else(|| program_error("no executable is active"))?;
        let unit = self.resolve(caller, origin)?;
        let function = unit
            .local_function_for_origin(origin)
            .map_err(program_error)?
            .ok_or_else(|| program_error("original callable is absent"))?;
        let bytecode = unit
            .functions()
            .get(function)
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
            .portable_envelope_for_function(origin.function)
            .map_err(program_error)?;
        Ok(RuntimeProgramCallable {
            owner_identity: unit.source().owner_identity.clone(),
            display_name: display_name.to_string(),
            source_digest: runmat_execution::Digest::sha256(unit.source().text.as_bytes()),
            executable_source_digest,
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

#[cfg(test)]
mod tests;
