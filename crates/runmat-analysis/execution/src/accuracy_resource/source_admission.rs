//! One admission boundary for preparation, native execution and cold accuracy import.
//! Metadata and source reference buffers are components; full family reconstruction remains separate.
use super::*;
use runmat_analysis_fea::{
    execution::AnalysisNumericalObjectKind as Kind, planning::resource::FeaResourceDemand,
    results::observation::ObservationPublicationCatalog,
};

fn phase_slot(kind: Kind) -> Option<usize> {
    match kind {
        Kind::AccuracyEvaluationRequest => Some(0),
        Kind::NumericalReceipt => Some(1),
        Kind::ObservationCatalog => Some(2),
        Kind::ResourcePlan => Some(3),
        _ => None,
    }
}

/// Rederive the original-source plan before catalog validation or original-source context certification.
/// Inputs may omit the ResourcePlan during preparation; if carried, it must match exactly.
pub fn validate_admitted_accuracy_source(
    request: &AccuracyEvaluationRequest,
    receipt: &AccuracySourceReceipt,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[PreparedAnalysisObjects],
    admitted: &FeaResourcePlan,
) -> AnalysisBridgeResult<()> {
    validate_accuracy_source(request, receipt, observations, inputs, admitted, true)
}

/// As `validate_admitted_accuracy_source` for a plan the caller has just derived with
/// `build_prepared_accuracy_resource_plan(request, receipt, inputs)` from these exact inputs:
/// the plan is its own rederivation, so it is not derived a second time.
pub(crate) fn validate_derived_accuracy_source(
    request: &AccuracyEvaluationRequest,
    receipt: &AccuracySourceReceipt,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[PreparedAnalysisObjects],
    derived: &FeaResourcePlan,
) -> AnalysisBridgeResult<()> {
    validate_accuracy_source(request, receipt, observations, inputs, derived, false)
}

fn validate_accuracy_source(
    request: &AccuracyEvaluationRequest,
    receipt: &AccuracySourceReceipt,
    observations: Option<&ObservationPublicationCatalog>,
    inputs: &[PreparedAnalysisObjects],
    admitted: &FeaResourcePlan,
    rederive: bool,
) -> AnalysisBridgeResult<()> {
    admitted.validate_identity()?;
    let mut phase_seen = [false; 4];
    for input in inputs {
        authenticate_accuracy_objects(input)?;
        if let Some(slot) = phase_slot(input.manifest.kind) {
            if phase_seen[slot] {
                return Err(AnalysisBridgeError::Identity(
                    "accuracy phase input inventory is duplicated",
                ));
            }
            phase_seen[slot] = true;
        }
        if input.manifest.kind == Kind::ResourcePlan
            && (input.manifest.logical_identity != admitted.identity.digest()
                || FeaResourcePlan::from_canonical_bytes(&input.payload())? != *admitted)
        {
            return Err(AnalysisBridgeError::Identity(
                "accuracy carried source admission differs",
            ));
        }
    }
    if !phase_seen[0] || !phase_seen[1] || phase_seen[2] != observations.is_some() {
        return Err(AnalysisBridgeError::Identity(
            "accuracy phase input inventory differs",
        ));
    }
    if rederive && build_prepared_accuracy_resource_plan(request, receipt, inputs)? != *admitted {
        return Err(AnalysisBridgeError::Identity(
            "accuracy original-source resource rederivation differs",
        ));
    }
    let FeaResourceDemand::AccuracyEvaluation(demand) = &admitted.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "accuracy source admission family differs",
        ));
    };
    let evidence = receipt.evidence(request)?;
    let report = accuracy_evaluation_resource_profile(request, &evidence)
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    let count = inputs
        .iter()
        .filter(|input| phase_slot(input.manifest.kind).is_none())
        .count();
    let references = count
        .checked_mul(std::mem::size_of::<&PreparedAnalysisObjects>())
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("accuracy source reference workspace overflows".into())
        })?;
    let available = demand
        .evaluator_workspace_bytes_upper_bound
        .0
        .checked_sub(report.workspace_bytes_upper_bound)
        .and_then(|v| v.checked_sub(references))
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("accuracy source metadata workspace underflows".into())
        })?;
    if request.source_observations != observations.map(|catalog| catalog.identity()) {
        return Err(AnalysisBridgeError::Identity(
            "accuracy source catalog binding differs",
        ));
    }
    if let Some(catalog) = observations {
        catalog.validate_source_result_admitted(
            request.producer,
            request.source_kind,
            request.source_result,
            available,
        )?;
    }
    let mut context = Vec::new();
    context.try_reserve_exact(count).map_err(|_| {
        AnalysisBridgeError::Invalid("accuracy source reference allocation failed".into())
    })?;
    context.extend(
        inputs
            .iter()
            .filter(|input| phase_slot(input.manifest.kind).is_none()),
    );
    crate::validate_accuracy_source_context(request, receipt, observations, &context)
}
