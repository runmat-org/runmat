use runmat_types::{ProgramSourceId, SourceId};
use serde::{Deserialize, Serialize};

/// A compiler ordinal and an issued session identity are different authorities.
/// Portable artifacts may contain only `Program`; `Runtime` belongs to a
/// selected execution or a retained session definition.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BytecodeSourceRef {
    Program(ProgramSourceId),
    Runtime(runmat_runtime::source_context::RuntimeSourceRef),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "domain", content = "id", rename_all = "lowercase")]
enum PortableSourceRef {
    Program(ProgramSourceId),
}

impl Serialize for BytecodeSourceRef {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Program(id) => PortableSourceRef::Program(*id).serialize(serializer),
            Self::Runtime(_) => Err(serde::ser::Error::custom(
                "session source authority cannot enter portable bytecode",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for BytecodeSourceRef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match PortableSourceRef::deserialize(deserializer)? {
            PortableSourceRef::Program(id) => Ok(Self::Program(id)),
        }
    }
}

impl BytecodeSourceRef {
    pub fn resolve_in(self, runtime: &runmat_runtime::context::RuntimeContext) -> Option<SourceId> {
        match self {
            Self::Program(id) => runtime.program_source_id(id),
            Self::Runtime(reference) => reference.resolve_in(runtime),
        }
    }

    pub fn program(self) -> Option<ProgramSourceId> {
        match self {
            Self::Program(id) => Some(id),
            Self::Runtime(_) => None,
        }
    }

    pub fn runtime(self) -> Option<runmat_runtime::source_context::RuntimeSourceRef> {
        match self {
            Self::Program(_) => None,
            Self::Runtime(reference) => Some(reference),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bytecode, Instr};
    use std::rc::Rc;

    fn context() -> runmat_runtime::context::RuntimeContext {
        runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
    }

    #[test]
    fn selected_interpreter_rejects_colliding_foreign_source_before_execution() {
        let first = context();
        let second = context();
        let first_lease = runmat_runtime::source_context::lease_source_context_in_context(
            &first,
            "first.m",
            None,
            "answer = 11;",
        )
        .expect("first source");
        let second_lease = runmat_runtime::source_context::lease_source_context_in_context(
            &second,
            "second.m",
            None,
            "answer = 22;",
        )
        .expect("second source");
        assert_eq!(first_lease.source_id(), second_lease.source_id());
        let mut bytecode =
            Bytecode::with_instructions(vec![Instr::LoadConst(11.0), Instr::Return], 0);
        bytecode.source_id = Some(BytecodeSourceRef::Runtime(first_lease.source_ref()));
        let values = Vec::new();
        let execution =
            futures::executor::block_on(crate::interpreter::runner::interpret_workspace(
                &bytecode,
                crate::InterpreterVars::from_owned(values),
                Some("first"),
                second.clone(),
                None,
                None,
            ));
        let rejected = execution
            .outcome
            .expect_err("foreign source owner must fail before executing bytecode");
        assert!(rejected.to_string().contains("another runtime owner"));
        assert!(execution
            .workspace
            .is_none_or(|workspace| workspace.state.vars.is_empty()));
        assert_eq!(
            runmat_runtime::source_context::source_info_in_context(
                &second,
                second_lease.source_id()
            )
            .expect("second text remains selected")
            .text
            .as_ref(),
            "answer = 22;"
        );
    }
}
