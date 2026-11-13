mod class_sources;
mod dependency;
mod function;
mod program_point;
mod projection;
mod region;
mod version;

pub use dependency::*;
pub use function::*;
pub use program_point::*;
pub use region::*;
pub use version::*;

use crate::{MirDiagnostic, MirOperationContract, MirOperationSite};
use serde::{Deserialize, Serialize};

/// Complete, portable semantic result for one MIR assembly.
///
/// Program-point facts and function summaries are the sole value-analysis
/// authority. Operation contracts are the complete per-site effect and
/// capability authority. Consumers select both by stable program identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisStore {
    pub revision: AnalysisRevision,
    pub dependencies: Vec<AnalysisDependency>,
    pub program_points: Vec<ProgramPointFacts>,
    #[serde(default)]
    pub operation_contracts: Vec<MirOperationContract>,
    #[serde(default)]
    pub fusion_stage_contracts: Vec<super::FusionStageContract>,
    pub regions: Vec<RegionAnalysis>,
    pub parallel: runmat_types::ParallelManifest,
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
            operation_contracts: Vec::new(),
            fusion_stage_contracts: Vec::new(),
            regions: Vec::new(),
            parallel: runmat_types::ParallelManifest::empty(),
            functions: Vec::new(),
            classes: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}

impl AnalysisStore {
    /// Canonical payload bytes used by the executable Analysis component and
    /// every downstream projection binding. Keeping the codec at the semantic
    /// owner prevents native consumers from inventing a parallel identity.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, String> {
        let value = serde_json::to_value(self).map_err(|error| error.to_string())?;
        serde_json::to_vec(&value).map_err(|error| error.to_string())
    }

    /// Resolve a compiler coordinate through its canonical self binding, never
    /// by interpreting a portable ordinal as a compiler/session index.
    pub fn function_for_local(&self, local: runmat_types::FunctionId) -> Option<&FunctionAnalysis> {
        let mut matches = self.functions.iter().filter(|function| {
            function.callable.identity == Some(runmat_types::CallableIdentity::BoundFunction(local))
        });
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }

    pub fn fusion_stage_at(
        &self,
        point: runmat_types::ProgramPointId,
    ) -> Option<&super::FusionStageContract> {
        self.fusion_stage_contracts
            .binary_search_by_key(&point, |contract| contract.point)
            .ok()
            .map(|index| &self.fusion_stage_contracts[index])
    }

    pub fn facts_at(&self, point: runmat_types::ProgramPointId) -> Option<&ProgramPointFacts> {
        self.program_points
            .binary_search_by_key(&point, |facts| facts.point)
            .ok()
            .map(|index| &self.program_points[index])
    }

    pub fn operation_contract_at(&self, site: &MirOperationSite) -> Option<&MirOperationContract> {
        self.operation_contracts
            .binary_search_by(|contract| contract.site.cmp(site))
            .ok()
            .map(|index| &self.operation_contracts[index])
    }

    pub(crate) fn local_facts_at(
        &self,
        point: runmat_types::ProgramPointId,
        local_count: usize,
    ) -> Vec<Option<runmat_types::ValueFact>> {
        self.facts_at(point).map_or_else(
            || vec![None; local_count],
            |facts| facts.local_fact_vector(local_count),
        )
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
    /// A dynamic workspace effect may have supplied this slot. Reading it
    /// requires the executor's missing-binding check, not a static proof.
    RuntimeWorkspaceLookup,
}
