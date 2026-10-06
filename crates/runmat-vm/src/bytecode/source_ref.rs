use runmat_types::{ProgramSourceId, SourceId};
use serde::{Deserialize, Serialize};

/// A compiler ordinal and an issued session identity are different authorities.
/// Portable artifacts may contain only `Program`; `Runtime` belongs to a
/// selected execution or a retained session definition.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(tag = "domain", content = "id", rename_all = "lowercase")]
pub enum BytecodeSourceRef {
    Program(ProgramSourceId),
    Runtime(SourceId),
}

impl BytecodeSourceRef {
    pub fn resolve_in(self, runtime: &runmat_runtime::context::RuntimeContext) -> Option<SourceId> {
        match self {
            Self::Program(id) => runtime.program_source_id(id),
            Self::Runtime(id) => Some(id),
        }
    }

    pub fn program(self) -> Option<ProgramSourceId> {
        match self {
            Self::Program(id) => Some(id),
            Self::Runtime(_) => None,
        }
    }

    pub fn runtime(self) -> Option<SourceId> {
        match self {
            Self::Program(_) => None,
            Self::Runtime(id) => Some(id),
        }
    }
}
