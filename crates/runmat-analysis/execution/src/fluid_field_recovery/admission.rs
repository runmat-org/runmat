//! Authenticate and admit original wire closures before numerical source decoding.
use super::*;
use runmat_analysis_fea::planning::resource::{FeaResourceDemand, FeaResourcePhase};

pub(super) fn validate_inventory_shape(
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<()> {
    let kinds = [
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        AnalysisNumericalObjectKind::FluidSteadyWorkload,
        AnalysisNumericalObjectKind::Mesh,
        AnalysisNumericalObjectKind::SolverResult,
    ];
    if !(4..=5).contains(&inputs.len())
        || kinds
            .iter()
            .any(|kind| inputs.iter().filter(|i| i.manifest.kind == *kind).count() != 1)
        || inputs
            .iter()
            .filter(|i| i.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
            .count()
            != inputs.len() - 4
    {
        return Err(invalid(
            "fluid recovery requires exact request/workload/mesh/result and optional plan",
        ));
    }
    Ok(())
}

pub fn require_fluid_recovery_source_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    validate_inventory_shape(inputs)?;
    plan.validate_identity()?;
    let FeaResourceDemand::FluidFieldRecovery(demand) = &plan.input.demand else {
        return Err(invalid(
            "fluid recovery source admission requires its recovery resource family",
        ));
    };
    if plan.input.phase != FeaResourcePhase::FluidFieldRecovery(demand.request) {
        return Err(invalid("fluid recovery phase/request differs"));
    }
    let mut encoded = 0_u64;
    for input in inputs {
        validate_inventory(&input.objects, limits)?;
        authenticate_source_objects(input, input.manifest.kind)?;
        if input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan {
            if crate::decode_admitted_resource_plan(input, limits)? != *plan {
                return Err(invalid("fluid recovery carried resource plan differs"));
            }
        } else {
            encoded = encoded
                .checked_add(input.manifest.encoded_length)
                .ok_or_else(|| invalid("fluid recovery source wire overflow"))?;
        }
    }
    // Authenticate/count borrowed chunks before the shared actual-decode profiler joins
    // any payload. Joined wire buffers and subsequent typed decode have separate reserves.
    if encoded != demand.inputs.encoded_payload_bytes.0
        || encoded > demand.inputs.decoded_resident_bytes_admission_budget.0
    {
        return Err(invalid(
            "fluid recovery source exceeds carried wire/decode admission",
        ));
    }
    let facts = prepared_input_resource_facts_without_resource_plan(inputs)?;
    if facts != demand.inputs
        || facts.decoded_resident_bytes_admission_budget.0 < demand.actual_decode_resident_bytes.0
        || unique(inputs, AnalysisNumericalObjectKind::FieldRecoveryRequest)?
            .manifest
            .logical_identity
            != demand.request.digest()
        || unique(inputs, AnalysisNumericalObjectKind::SolverResult)?
            .manifest
            .logical_identity
            != demand.source_result.digest()
    {
        return Err(invalid(
            "fluid recovery source profiles/request/result differ from admitted plan",
        ));
    }
    Ok(())
}
