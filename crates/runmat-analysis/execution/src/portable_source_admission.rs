//! Shared streaming authentication of immutable numerical-object closures.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::planning::resource::{FeaResourcePlan, RESOURCE_PLAN_MAX_ENCODED_BYTES};
use runmat_execution::Digest;
use runmat_execution_artifact::{
    object::{validate_inventory, ObjectInventoryLimits},
    ObjectNamespace,
};

/// Authenticate and decode the bounded immutable resource plan before any numerical
/// source join. The consuming phase still rederives its exact physical authority.
pub fn decode_admitted_resource_plan(
    input: &PreparedAnalysisObjects,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    validate_inventory(&input.objects, limits)?;
    if input.manifest.encoded_length > RESOURCE_PLAN_MAX_ENCODED_BYTES {
        return Err(AnalysisBridgeError::Identity(
            "resource input exceeds canonical bound",
        ));
    }
    authenticate_source_objects(input, AnalysisNumericalObjectKind::ResourcePlan)?;
    let plan = FeaResourcePlan::from_canonical_bytes(&input.payload())?;
    if input.manifest.logical_identity != plan.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "resource input identity differs",
        ));
    }
    Ok(plan)
}
/// Authenticate the exact owned closure without joining its payload chunks.
/// Retained results remain ResultValue objects; original inputs remain InputValue objects.
pub(crate) fn authenticate_source_objects(
    objects: &PreparedAnalysisObjects,
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<()> {
    objects.manifest.validate()?;
    let namespace = if matches!(
        kind,
        AnalysisNumericalObjectKind::SolverResult
            | AnalysisNumericalObjectKind::NonlinearSolidBlockArtifact
    ) {
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
            "portable source kind, namespace or inventory differs",
        ));
    }
    let root = &objects.objects[0];
    if root.descriptor != objects.root
        || root.descriptor.encoded_length != root.bytes.len() as u64
        || root.bytes != objects.manifest.canonical_bytes()?
        || Digest::sha256(&root.bytes) != objects.root.digest
    {
        return Err(AnalysisBridgeError::Identity(
            "portable source root differs from canonical manifest",
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
                "portable source chunk differs from immutable digest",
            ));
        }
        bytes = bytes.checked_add(chunk.encoded_length).ok_or_else(|| {
            AnalysisBridgeError::Invalid("portable source length overflow".into())
        })?;
    }
    if bytes != objects.manifest.encoded_length
        || Digest::sha256_parts(objects.objects[1..].iter().map(|o| &o.bytes))
            != objects.manifest.payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "portable source payload differs from immutable digest",
        ));
    }
    Ok(())
}
