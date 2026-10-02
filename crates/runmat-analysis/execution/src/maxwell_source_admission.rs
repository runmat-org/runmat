//! Shared streaming source authentication and pre-decode admission for Maxwell.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::{
    execution::AnalysisNumericalObjectKind,
    planning::resource::{FeaResourceDemand, FeaResourcePlan, RESOURCE_PLAN_MAX_ENCODED_BYTES},
};
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};

pub(crate) use crate::portable_source_admission::authenticate_source_objects as authenticate_maxwell_source_objects;

/// Decode only the bounded immutable plan, before numerical source payload materialization.
pub fn decode_admitted_maxwell_resource_plan(
    input: &PreparedAnalysisObjects,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    validate_inventory(&input.objects, limits)?;
    if input.manifest.encoded_length > RESOURCE_PLAN_MAX_ENCODED_BYTES {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell resource input exceeds canonical bound",
        ));
    }
    authenticate_maxwell_source_objects(input, AnalysisNumericalObjectKind::ResourcePlan)?;
    let plan = FeaResourcePlan::from_canonical_bytes(&input.payload())?;
    if input.manifest.logical_identity != plan.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell resource input identity differs",
        ));
    }
    Ok(plan)
}

/// Admit exact original wire counts before decoding workload or mesh.
/// Semantic identities and numerical allocation must still be rederived after decoding.
pub fn require_maxwell_source_decode_budget(
    inputs: &[PreparedAnalysisObjects],
    plan: &FeaResourcePlan,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    plan.validate_identity()?;
    let FeaResourceDemand::MaxwellHarmonic(demand) = &plan.input.demand else {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell task resource family differs",
        ));
    };
    let kinds = [
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
        AnalysisNumericalObjectKind::Mesh,
        AnalysisNumericalObjectKind::ResourcePlan,
    ];
    if inputs.len() != kinds.len()
        || kinds.iter().any(|kind| {
            inputs
                .iter()
                .filter(|input| input.manifest.kind == *kind)
                .count()
                != 1
        })
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell task requires exact original input kinds",
        ));
    }
    let mut encoded = 0_u64;
    for input in inputs {
        validate_inventory(&input.objects, limits)?;
        authenticate_maxwell_source_objects(input, input.manifest.kind)?;
        if input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan {
            if decode_admitted_maxwell_resource_plan(input, limits)? != *plan {
                return Err(AnalysisBridgeError::Identity(
                    "Maxwell immutable resource input differs from admitted plan",
                ));
            }
        } else {
            encoded = encoded
                .checked_add(input.manifest.encoded_length)
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid("Maxwell source length overflow".into())
                })?;
        }
    }
    if encoded != demand.inputs.encoded_payload_bytes.0
        || encoded > demand.inputs.decoded_resident_bytes_admission_budget.0
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell original source exceeds admitted decode budget",
        ));
    }
    Ok(())
}
