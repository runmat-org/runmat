//! Shared streaming source authentication and pre-decode admission for Maxwell.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::{
    execution::AnalysisNumericalObjectKind,
    planning::resource::{FeaResourceDemand, FeaResourcePlan, RESOURCE_PLAN_MAX_ENCODED_BYTES},
};
use runmat_execution::Digest;
use runmat_execution_artifact::{
    object::{validate_inventory, ObjectInventoryLimits},
    ObjectNamespace,
};

/// Authenticate the exact owned closure without joining its payload chunks.
/// Retained results remain ResultValue objects; original inputs remain InputValue objects.
pub(crate) fn authenticate_maxwell_source_objects(
    objects: &PreparedAnalysisObjects,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<()> {
    objects.manifest.validate()?;
    let namespace = if kind == AnalysisNumericalObjectKind::SolverResult {
        ObjectNamespace::ResultValue
    } else {
        ObjectNamespace::InputValue
    };
    if objects.manifest.kind != kind
        || objects.manifest.chunks.len().checked_add(1) != Some(objects.objects.len())
        || objects
            .objects
            .iter()
            .any(|o| o.descriptor.namespace != namespace)
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell source kind, namespace or inventory differs",
        ));
    }
    let root = &objects.objects[0];
    if root.descriptor != objects.root
        || root.descriptor.encoded_length != root.bytes.len() as u64
        || root.bytes != objects.manifest.canonical_bytes()?
        || Digest::sha256(&root.bytes) != objects.root.digest
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell source root differs from canonical manifest",
        ));
    }
    let mut bytes = 0_u64;
    for (object, chunk) in objects.objects[1..].iter().zip(&objects.manifest.chunks) {
        if object.bytes.len() as u64 != chunk.encoded_length
            || object.descriptor.encoded_length != chunk.encoded_length
            || object.descriptor.digest != chunk.digest
            || Digest::sha256(&object.bytes) != chunk.digest
        {
            return Err(AnalysisBridgeError::Identity(
                "Maxwell source chunk differs from immutable digest",
            ));
        }
        bytes = bytes
            .checked_add(chunk.encoded_length)
            .ok_or_else(|| AnalysisBridgeError::Invalid("Maxwell source length overflow".into()))?;
    }
    if bytes != objects.manifest.encoded_length
        || Digest::sha256_parts(objects.objects[1..].iter().map(|o| &o.bytes))
            != objects.manifest.payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell source payload differs from immutable digest",
        ));
    }
    Ok(())
}

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
