mod dependency;
mod function;
mod program_point;
mod version;

pub use dependency::*;
pub use function::*;
pub use program_point::*;
pub use version::*;

use runmat_hir::FunctionId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{MirDiagnostic, MirLocalId};

use super::MirLocalFact;

/// Complete, portable semantic result for one MIR assembly.
///
/// `program_points` and `functions` are the authoritative products. `mir_locals`
/// is a deterministic final-fact projection retained for consumers migrating in
/// R08; it is not a second inference authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisStore {
    pub revision: AnalysisRevision,
    pub dependencies: Vec<AnalysisDependency>,
    pub program_points: Vec<ProgramPointFacts>,
    pub functions: Vec<FunctionAnalysis>,
    pub classes: Vec<ClassAnalysis>,
    pub mir_locals: BTreeMap<MirLocalKey, MirLocalFact>,
    pub diagnostics: Vec<MirDiagnostic>,
}

impl Default for AnalysisStore {
    fn default() -> Self {
        Self {
            revision: AnalysisRevision::current(),
            dependencies: Vec::new(),
            program_points: Vec::new(),
            functions: Vec::new(),
            classes: Vec::new(),
            mir_locals: BTreeMap::new(),
            diagnostics: Vec::new(),
        }
    }
}

impl AnalysisStore {
    pub fn facts_at(&self, point: runmat_types::ProgramPointId) -> Option<&ProgramPointFacts> {
        self.program_points
            .binary_search_by_key(&point, |facts| facts.point)
            .ok()
            .map(|index| &self.program_points[index])
    }

    pub fn function(&self, function: runmat_types::ProgramFunctionId) -> Option<&FunctionAnalysis> {
        self.functions
            .binary_search_by_key(&function, |facts| facts.function)
            .ok()
            .map(|index| &self.functions[index])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MirLocalKey {
    pub function: FunctionId,
    pub local: MirLocalId,
}

impl Serialize for MirLocalKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{}:{}", self.function.0, self.local.0))
    }
}

impl<'de> Deserialize<'de> for MirLocalKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de;
        let value = String::deserialize(deserializer)?;
        let Some((function, local)) = value.split_once(':') else {
            return Err(de::Error::custom(
                "expected MIR local key as function:local",
            ));
        };
        Ok(Self {
            function: FunctionId(function.parse().map_err(de::Error::custom)?),
            local: MirLocalId(local.parse().map_err(de::Error::custom)?),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum InitFact {
    Unassigned,
    MaybeAssigned,
    DefinitelyAssigned,
}
