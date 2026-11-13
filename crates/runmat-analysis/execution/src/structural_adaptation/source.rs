//! Authenticate immutable producing contracts before decoding physical data.
use super::*;

pub(super) struct OriginalSource {
    pub workload: StructuralLinearWorkload,
    pub mesh: SolverMeshArtifact,
    pub result: StructuralLinearWorkloadResult,
    pub receipt: StructuralLinearNumericalReceiptArtifact,
    pub identity: SolverResultIdentity,
}

pub(super) fn decode(
    input: StructuralAdaptationNativeSource<'_>,
) -> AnalysisBridgeResult<OriginalSource> {
    input.host.validate()?;
    if input.host.workload.kind != AnalysisWorkKind::Solve {
        return Err(AnalysisBridgeError::Identity(
            "adaptation requires original structural solve producers",
        ));
    }
    for (kind, objects) in [
        (Kind::SolverInvocation, input.invocation),
        (Kind::Mesh, input.mesh),
    ] {
        crate::portable_source_admission::authenticate_source_objects(objects, kind)?;
        let mut original = input.host.workload.inputs.iter().filter(|i| i.kind == kind);
        if !matches!(original.next(), Some(i) if i.logical_identity == objects.manifest.logical_identity
            && i.payload_digest == objects.manifest.payload_digest
            && i.manifest_digest == objects.root.digest
            && i.manifest_encoded_length == objects.root.encoded_length)
            || original.next().is_some()
        {
            return Err(AnalysisBridgeError::Identity(
                "adaptive source differs from the original producing inputs",
            ));
        }
    }
    crate::portable_source_admission::authenticate_source_objects(
        input.result,
        Kind::SolverResult,
    )?;
    crate::accuracy_resource::authenticate_accuracy_objects(input.receipt)?;
    if input.receipt.manifest.kind != Kind::NumericalReceipt {
        return Err(AnalysisBridgeError::Identity(
            "adaptive source receipt has the wrong object kind",
        ));
    }
    let (identity, result) = crate::structural::result_source::decode_structural_result_source(
        input.host,
        input.result,
    )?;
    let receipt =
        StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(&input.receipt.payload())?;
    if receipt.identity().digest() != input.receipt.manifest.logical_identity
        || receipt.source_workload != input.host.workload.identity
        || receipt.source_payload_digest != input.result.manifest.payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "adaptive receipt differs from its original producer or result",
        ));
    }
    receipt.validate_against(identity, &result)?;
    let workload = StructuralLinearWorkload::from_canonical_bytes(&input.invocation.payload())?;
    if workload.state_input.is_some()
        || !workload.procedure_step()?.state_inputs.is_empty()
        || workload.backend != result.run.requested_backend
    {
        return Err(AnalysisBridgeError::Invalid(
            "native solid adaptation currently requires original fresh static solves".into(),
        ));
    }
    let mesh = SolverMeshArtifact::canonical_decode(&input.mesh.payload())
        .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
    if Digest::from_bytes(*mesh.canonical_digest.bytes()) != input.mesh.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "adaptive mesh digest differs from its original object",
        ));
    }
    Ok(OriginalSource {
        workload,
        mesh,
        result,
        receipt,
        identity,
    })
}
