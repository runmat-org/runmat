//! Retained source closure admission, separate from expensive scientific authentication.
use super::*;
use runmat_analysis_fea::execution::RetainedJouleHeatSource;
use runmat_analysis_fea::planning::resource::{ByteCount, ThermalFieldSourceResourceFacts};

/// Structural decoding and count/profile reservation only. Neither `bind` nor `validate_for`
/// runs here: the worker must first compare/admit the freshly rederived complete resource plan.
pub(super) fn decode(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    inputs: &[PreparedAnalysisObjects],
    resource_plan_present: bool,
) -> AnalysisBridgeResult<Option<RetainedJouleHeatSource>> {
    let count = |kind| {
        inputs
            .iter()
            .filter(|input| input.manifest.kind == kind)
            .count()
    };
    let source_count = count(AnalysisNumericalObjectKind::FieldSourceBinding);
    if source_count > 1
        || inputs.len() != 2 + usize::from(resource_plan_present) + source_count
        || count(AnalysisNumericalObjectKind::ThermalWorkload) != 1
        || count(AnalysisNumericalObjectKind::Mesh) != 1
        || count(AnalysisNumericalObjectKind::ResourcePlan) != usize::from(resource_plan_present)
    {
        return Err(AnalysisBridgeError::Invalid(
            "thermal task requires exact workload/mesh/resource plan and at most one retained field source".into(),
        ));
    }
    let workload_identity = workload.identity()?.digest();
    if !inputs.iter().any(|input| {
        input.manifest.kind == AnalysisNumericalObjectKind::ThermalWorkload
            && input.manifest.logical_identity == workload_identity
    }) || !inputs.iter().any(|input| {
        input.manifest.kind == AnalysisNumericalObjectKind::Mesh
            && input.manifest.logical_identity == Digest::from_bytes(*mesh.canonical_digest.bytes())
    }) {
        return Err(AnalysisBridgeError::Identity(
            "thermal source closure differs from decoded workload or mesh",
        ));
    }
    let Some(objects) = inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::FieldSourceBinding)
    else {
        if RetainedJouleHeatSource::requires_binding(workload) {
            return Err(AnalysisBridgeError::Invalid(
                "thermal retained field is missing its original source binding".into(),
            ));
        }
        return Ok(None);
    };
    let payload = objects.payload();
    if payload.len() as u64 != objects.manifest.encoded_length
        || Digest::sha256(&payload) != objects.manifest.payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal field-source payload differs from its original input",
        ));
    }
    let source = RetainedJouleHeatSource::from_canonical_bytes(&payload)?;
    if source.identity()? != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "thermal field-source logical identity differs",
        ));
    }
    Ok(Some(source))
}

pub(super) fn facts(
    source: Option<&RetainedJouleHeatSource>,
    mesh: &SolverMeshArtifact,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<Option<ThermalFieldSourceResourceFacts>> {
    source
        .map(|source| {
            let objects = inputs
                .iter()
                .find(|input| {
                    input.manifest.kind == AnalysisNumericalObjectKind::FieldSourceBinding
                })
                .ok_or_else(|| {
                    AnalysisBridgeError::Invalid("thermal source objects missing".into())
                })?;
            Ok(ThermalFieldSourceResourceFacts {
                identity: source.identity()?,
                encoded_payload_bytes: ByteCount(objects.manifest.encoded_length),
                validation_workspace_bytes: ByteCount(source.validation_workspace_bytes(mesh)?),
            })
        })
        .transpose()
}

/// Scientific authentication after resource-plan equality/admission. Rebuilds only the bound
/// target from original retained fields and checks exact equality; never executes electrical physics.
pub fn validate_thermal_task_source(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<Option<RetainedJouleHeatSource>> {
    let source = decode(workload, mesh, inputs, true)?;
    if let Some(source) = &source {
        source.validate_for(workload, mesh)?;
    }
    Ok(source)
}
