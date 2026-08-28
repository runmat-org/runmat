use runmat_types::{ParallelRegionId, ProgramFunctionId};
use serde::{Deserialize, Serialize};

use crate::{ContractError, Digest};

/// Callable identity admitted by an isolated program worker.
///
/// Names are resolved against the frozen program/catalog before submission.
/// Captured values travel as ordinary task inputs, so workers never repeat
/// workspace or path resolution.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramCallable {
    Semantic {
        function: ProgramFunctionId,
        display_name: Option<String>,
    },
    Builtin {
        name: String,
    },
    ParallelRegion {
        region: ParallelRegionId,
    },
}

impl ProgramCallable {
    pub fn semantic(function: ProgramFunctionId, display_name: Option<String>) -> Self {
        Self::Semantic {
            function,
            display_name,
        }
    }

    pub fn builtin(name: impl Into<String>) -> Result<Self, ContractError> {
        let callable = Self::Builtin { name: name.into() };
        callable.validate()?;
        Ok(callable)
    }

    pub fn parallel_region(region: ParallelRegionId) -> Self {
        Self::ParallelRegion { region }
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        match self {
            Self::Semantic { display_name, .. } => {
                if display_name
                    .as_deref()
                    .is_some_and(|name| name.trim().is_empty() || name.contains('\0'))
                {
                    return Err(ContractError::invalid(
                        "program callable",
                        "semantic display name is empty or contains NUL",
                    ));
                }
            }
            Self::Builtin { name } if name.trim().is_empty() || name.contains('\0') => {
                return Err(ContractError::invalid(
                    "program callable",
                    "builtin name is empty or contains NUL",
                ));
            }
            Self::Builtin { .. } => {}
            Self::ParallelRegion { .. } => {}
        }
        Ok(())
    }

    pub fn semantic_function(&self) -> Option<ProgramFunctionId> {
        match self {
            Self::Semantic { function, .. } => Some(*function),
            Self::ParallelRegion { region } => Some(region.0.function),
            Self::Builtin { .. } => None,
        }
    }

    pub fn recipe_entrypoint(&self) -> String {
        match self {
            Self::Semantic { function, .. } => function.0.to_string(),
            Self::Builtin { name } => format!("builtin:{name}"),
            Self::ParallelRegion { region } => format!(
                "parallel-region:{}:{}",
                region.0.function.0, region.0.ordinal
            ),
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            Self::Semantic {
                function,
                display_name,
            } => display_name
                .clone()
                .unwrap_or_else(|| format!("function#{}", function.0)),
            Self::Builtin { name } => name.clone(),
            Self::ParallelRegion { region } => {
                format!("parfor region {}:{}", region.0.function.0, region.0.ordinal)
            }
        }
    }

    pub fn identity_digest(&self) -> Digest {
        Digest::sha256(format!(
            "runmat-program-callable-v1\0{}",
            self.recipe_entrypoint()
        ))
    }
}
