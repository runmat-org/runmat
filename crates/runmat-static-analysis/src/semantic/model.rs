use runmat_hir::{BindingId, BindingRole, BindingStorage, WorkspaceVisibility};
use runmat_mir::analysis::{AnalysisRevision, AssignmentFact, ClassAnalysis, FunctionAnalysis};
use runmat_types::{ProgramFunctionId, ProgramPointId, ProgramSpan, RegionValueId, ValueFact};
use serde::{Deserialize, Serialize};

/// Portable, presentation-neutral semantic facts for one immutable source
/// analysis. Native LSP, browser LSP, Desktop, and future tooling consume this
/// same product instead of rebuilding a shape or type environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticDocumentFacts {
    pub revision: AnalysisRevision,
    pub bindings: Vec<SemanticBindingFacts>,
    pub functions: Vec<SemanticFunctionFacts>,
    pub classes: Vec<ClassAnalysis>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticBindingFacts {
    pub binding: BindingId,
    pub name: String,
    pub role: BindingRole,
    pub storage: BindingStorage,
    pub workspace_visibility: WorkspaceVisibility,
    pub declaration: ProgramSpan,
    pub references: Vec<ProgramSpan>,
    pub regions: Vec<SemanticBindingRegion>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticBindingRegion {
    pub value: RegionValueId,
    pub function_span: ProgramSpan,
    pub observations: Vec<SemanticFactObservation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticFactObservation {
    pub point: ProgramPointId,
    pub span: ProgramSpan,
    pub assignment: AssignmentFact,
    pub fact: Option<ValueFact>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticFunctionFacts {
    pub function: ProgramFunctionId,
    pub name: String,
    pub span: ProgramSpan,
    pub parameters: Vec<BindingId>,
    pub outputs: Vec<BindingId>,
    pub analysis: Option<FunctionAnalysis>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticQuickInformation {
    pub revision: AnalysisRevision,
    pub binding: BindingId,
    pub name: String,
    pub role: BindingRole,
    pub storage: BindingStorage,
    pub workspace_visibility: WorkspaceVisibility,
    pub declaration: ProgramSpan,
    pub observation: Option<SemanticFactObservation>,
}

impl SemanticDocumentFacts {
    pub fn validate_current(&self) -> Result<(), runmat_mir::analysis::AnalysisRevisionMismatch> {
        self.revision.validate_current()
    }

    pub fn binding(&self, binding: BindingId) -> Option<&SemanticBindingFacts> {
        self.bindings
            .binary_search_by_key(&binding.0, |facts| facts.binding.0)
            .ok()
            .map(|index| &self.bindings[index])
    }

    pub fn binding_named_at(&self, name: &str, offset: usize) -> Option<&SemanticBindingFacts> {
        let offset = u64::try_from(offset).ok()?;
        self.bindings
            .iter()
            .filter(|binding| binding.name == name)
            .filter(|binding| {
                contains(binding.declaration, offset)
                    || binding
                        .references
                        .iter()
                        .any(|span| contains(*span, offset))
                    || binding
                        .regions
                        .iter()
                        .any(|region| contains(region.function_span, offset))
            })
            .min_by_key(|binding| {
                binding
                    .regions
                    .iter()
                    .filter(|region| contains(region.function_span, offset))
                    .map(|region| {
                        region
                            .function_span
                            .end
                            .saturating_sub(region.function_span.start)
                    })
                    .min()
                    .unwrap_or(u64::MAX)
            })
    }

    pub fn fact_at(&self, binding: BindingId, offset: usize) -> Option<&SemanticFactObservation> {
        self.binding(binding)?.fact_at(offset)
    }

    pub fn quick_information(&self, name: &str, offset: usize) -> Option<SemanticQuickInformation> {
        let binding = self.binding_named_at(name, offset)?;
        Some(SemanticQuickInformation {
            revision: self.revision.clone(),
            binding: binding.binding,
            name: binding.name.clone(),
            role: binding.role.clone(),
            storage: binding.storage.clone(),
            workspace_visibility: binding.workspace_visibility.clone(),
            declaration: binding.declaration,
            observation: binding.fact_at(offset).cloned(),
        })
    }
}

impl SemanticBindingFacts {
    pub fn fact_at(&self, offset: usize) -> Option<&SemanticFactObservation> {
        let offset = u64::try_from(offset).ok()?;
        let region = self
            .regions
            .iter()
            .filter(|region| contains(region.function_span, offset))
            .min_by_key(|region| {
                region
                    .function_span
                    .end
                    .saturating_sub(region.function_span.start)
            })
            .or_else(|| self.regions.first())?;

        // A point whose statement span contains the cursor is the most precise
        // observation. Position n+1 is the state after statement n, which makes
        // assignment targets immediately useful while preserving the same fact
        // for reads that do not mutate the queried binding.
        region
            .observations
            .iter()
            .filter(|observation| contains(observation.span, offset))
            .min_by_key(|observation| {
                (
                    observation.span.end.saturating_sub(observation.span.start),
                    std::cmp::Reverse(observation.point.position),
                )
            })
            .or_else(|| {
                region
                    .observations
                    .iter()
                    .filter(|observation| observation.span.end <= offset)
                    .max_by_key(|observation| {
                        (
                            observation.span.end,
                            observation.point.block,
                            observation.point.position,
                        )
                    })
            })
            .or_else(|| region.observations.first())
    }
}

fn contains(span: ProgramSpan, offset: u64) -> bool {
    span.start <= offset && offset <= span.end
}
