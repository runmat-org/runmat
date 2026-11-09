//! Source-validation workspace of a thermal accuracy evaluation with observations.
//!
//! The worker holds its decoded inputs (input facts) and, beside them, runs in sequence:
//! - the re-derivation of this plan, which decodes a second copy of the observation catalog;
//! - the catalog's `validate_for` over the original workload and mesh.
//!
//! Each input's chunks are joined into one buffer before its typed decode, one input at a
//! time. The workspace is the validation law plus one catalog copy plus the largest join.
use super::*;
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind as Kind, ThermalWorkload},
    planning::resource::FeaResourceDemand,
    results::{accuracy::AccuracyEvidence, observation::ObservationPublicationCatalog},
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub(super) fn workspace(
    request: &AccuracyEvaluationRequest,
    evidence: &[AccuracyEvidence],
    inputs: &[PreparedAnalysisObjects],
    facts: PreparedPhaseResourceFacts,
) -> AnalysisBridgeResult<u64> {
    let unique = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = inputs.iter().filter(|input| input.manifest.kind == kind);
        match (matches.next(), matches.next()) {
            (Some(input), None) => {
                authenticate_accuracy_objects(input)?;
                Ok(input)
            }
            _ => Err(AnalysisBridgeError::Identity(
                "thermal constructor requires unique original inputs",
            )),
        }
    };
    let catalog_input = unique(Kind::ObservationCatalog)?;
    let catalog_copy =
        crate::prepared_input_resource_facts_from_refs(std::iter::once(catalog_input))?
            .decoded_resident_bytes_admission_budget
            .0;
    let join = inputs
        .iter()
        .filter(|input| input.manifest.kind != Kind::ResourcePlan)
        .map(|input| input.manifest.encoded_length)
        .max()
        .unwrap_or(0);
    let beside = catalog_copy
        .checked_add(join)
        .ok_or_else(|| AnalysisBridgeError::Invalid("thermal source workspace overflows".into()))?;
    // Everything but the source workspace: decoded inputs, transport, output and the
    // evaluator's own profile. Validation may spend only what the ceiling leaves.
    let baseline =
        build_accuracy_resource_plan_with_source_workspace(request, evidence, facts, Some(0))?;
    let FeaResourceDemand::AccuracyEvaluation(base) = &baseline.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "thermal accuracy baseline family differs",
        ));
    };
    let mut available = request
        .plan
        .resources
        .resident_byte_budget()
        .checked_sub(baseline.request.memory_bytes)
        .and_then(|bytes| bytes.checked_sub(beside))
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("thermal source baseline exceeds ceiling".into())
        })?;
    // A carried plan bounds the rederivation by the workspace it admitted.
    for input in inputs
        .iter()
        .filter(|input| input.manifest.kind == Kind::ResourcePlan)
    {
        authenticate_accuracy_objects(input)?;
        let plan = FeaResourcePlan::from_canonical_bytes(&input.payload())?;
        let FeaResourceDemand::AccuracyEvaluation(demand) = plan.input.demand else {
            return Err(AnalysisBridgeError::Identity(
                "thermal accuracy carried source family differs",
            ));
        };
        available = available.min(
            demand
                .evaluator_workspace_bytes_upper_bound
                .0
                .checked_sub(base.evaluator_workspace_bytes_upper_bound.0)
                .and_then(|bytes| bytes.checked_sub(beside))
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid("thermal constructor workspace underflows".into())
                })?,
        );
    }
    let workload =
        ThermalWorkload::from_canonical_bytes(&unique(Kind::ThermalWorkload)?.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&unique(Kind::Mesh)?.payload())
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let catalog = ObservationPublicationCatalog::from_canonical_bytes(&catalog_input.payload())
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let validation =
        catalog.thermal_source_validation_resident_bytes_admitted(&workload, &mesh, available)?;
    validation
        .checked_add(beside)
        .ok_or_else(|| AnalysisBridgeError::Invalid("thermal source workspace overflows".into()))
}
