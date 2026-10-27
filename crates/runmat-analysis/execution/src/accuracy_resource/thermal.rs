//! Add original projection/schedule facts to the still-conservative legacy source remainder.
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
    let baseline =
        build_accuracy_resource_plan_with_source_workspace(request, evidence, facts, None)?;
    let FeaResourceDemand::AccuracyEvaluation(base) = &baseline.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "thermal accuracy baseline family differs",
        ));
    };
    // Keep all existing source/decode/output allowances. Only the unreserved ceiling may
    // be spent on new physical-constructor storage; carried plans must reserve that extra.
    let mut available = request
        .plan
        .resources
        .resident_byte_budget()
        .checked_sub(baseline.request.memory_bytes)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("thermal source baseline exceeds ceiling".into())
        })?;
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
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid("thermal constructor workspace underflows".into())
                })?,
        );
    }
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
    let workload =
        ThermalWorkload::from_canonical_bytes(&unique(Kind::ThermalWorkload)?.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&unique(Kind::Mesh)?.payload())
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let catalog = ObservationPublicationCatalog::from_canonical_bytes(
        &unique(Kind::ObservationCatalog)?.payload(),
    )
    .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let constructor =
        catalog.thermal_source_construction_resident_bytes_admitted(&workload, &mesh, available)?;
    let profile = accuracy_evaluation_resource_profile(request, evidence)
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    base.evaluator_workspace_bytes_upper_bound
        .0
        .checked_sub(profile.workspace_bytes_upper_bound)
        .and_then(|v| v.checked_add(constructor))
        .ok_or_else(|| AnalysisBridgeError::Invalid("thermal source workspace overflows".into()))
}
