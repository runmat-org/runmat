mod dependency;
mod function;
mod program_point;
mod region;
mod version;

pub use dependency::*;
pub use function::*;
pub use program_point::*;
pub use region::*;
pub use version::*;

use crate::MirDiagnostic;
use serde::{Deserialize, Serialize};

/// Complete, portable semantic result for one MIR assembly.
///
/// Program-point facts and function summaries are the sole value-analysis
/// authority. Consumers select facts by stable program identity and point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisStore {
    pub revision: AnalysisRevision,
    pub dependencies: Vec<AnalysisDependency>,
    pub program_points: Vec<ProgramPointFacts>,
    pub regions: Vec<RegionAnalysis>,
    pub functions: Vec<FunctionAnalysis>,
    pub classes: Vec<ClassAnalysis>,
    pub diagnostics: Vec<MirDiagnostic>,
}

impl Default for AnalysisStore {
    fn default() -> Self {
        Self {
            revision: AnalysisRevision::current(),
            dependencies: Vec::new(),
            program_points: Vec::new(),
            regions: Vec::new(),
            functions: Vec::new(),
            classes: Vec::new(),
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

    pub fn region(&self, region: runmat_types::RegionId) -> Option<&RegionAnalysis> {
        self.regions
            .binary_search_by_key(&region, |analysis| analysis.contract.id)
            .ok()
            .map(|index| &self.regions[index])
    }

    pub fn local_value_count(&self, function: Option<runmat_types::ProgramFunctionId>) -> usize {
        self.program_points
            .iter()
            .filter(|point| function.is_none_or(|function| point.point.function == function))
            .flat_map(|point| point.locals.iter().map(|local| local.value))
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum InitFact {
    Unassigned,
    MaybeAssigned,
    DefinitelyAssigned,
}
