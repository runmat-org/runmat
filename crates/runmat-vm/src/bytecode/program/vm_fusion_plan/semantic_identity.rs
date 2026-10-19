//! Immutable semantic learning identity for reconciled VM fusion regions.
//!
//! The projection is explicit and excludes invocation values and installation
//! nonces. Execution authorization remains owned by the installed-region right.

use runmat_execution::Digest;
use runmat_runtime::context::VmFusionSemanticIdentity;

use super::super::Bytecode;

/// Fingerprint one already-reconciled executable group exactly once per VM
/// installation. Per-group identities add the exact source-plan digest and
/// ordinal to the executable-wide owner projection.
pub(crate) fn vm_fusion_semantic_identity(
    executable: Digest,
    group: &runmat_accelerate::FusionGroupPlan,
) -> Option<VmFusionSemanticIdentity> {
    let source_plan = runmat_accelerate::fusion::exact_source_plan_identity(group).ok()?;
    let digest = Digest::sha256_serialized(&(
        "runmat.vm-fusion-executable-region.v1",
        executable,
        source_plan,
        group.index,
    ))
    .ok()?;
    Some(VmFusionSemanticIdentity::from_validated_digest(digest))
}

/// Hash the immutable executable-wide fields used by VM region admission.
/// `Instr` has a typed serde representation whose floating payloads use the
/// canonical bit-preserving adapters; runtime `Value` payloads are not part of
/// this stream.
pub(crate) fn vm_fusion_executable_identity(bytecode: &Bytecode) -> Option<Digest> {
    let resume_points = bytecode.resume_points.iter().collect::<Vec<_>>();
    let stage_source = bytecode
        .fusion_metadata
        .stage_projection
        .as_ref()
        .map(|projection| projection.source_fingerprint.as_str());
    Digest::sha256_serialized(&(
        "runmat.vm-fusion-executable-owner.v1",
        &bytecode.instructions,
        &bytecode.catch_scopes,
        &resume_points,
        bytecode.var_count,
        &bytecode.var_types,
        super::super::semantic_fingerprint::semantic_source_revision(bytecode.source_id),
        bytecode.active_function,
        &bytecode.active_class_method_owner,
        stage_source,
    ))
    .ok()
}
