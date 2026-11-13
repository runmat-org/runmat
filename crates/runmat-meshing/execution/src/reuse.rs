//! Accepted meshing stage receipts. Storage and lookup belong to execution adapters;
//! this module owns semantic binding and rebinding into current artifact authority.
use crate::{
    import_result_publication, prepare_result_publication, MeshingExecutionError,
    MeshingExecutionResult, MeshingHostWorkload,
};
use runmat_execution::value::ValuePayload;
use runmat_execution::Digest;
use runmat_execution_artifact::{
    cache::{CacheExport, CacheImport},
    object::ObjectInventoryLimits,
    ArtifactAccess,
};
use runmat_execution_runner::AttemptSuccess;
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingPartitionIdentity, MeshingStageKind, MeshingStageResultKind,
    MESHING_IDENTITY_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedMeshingStageReceipt {
    schema_version: u16,
    key: Digest,
    access: ArtifactAccess,
    root: runmat_execution::value::ValueRef,
    result_objects: Vec<runmat_execution::value::ValueRef>,
}

/// The full semantic workload includes partition, prerequisites and capabilities.
/// The implementation fingerprint must cover this stage's executable dependency closure.
pub fn meshing_stage_reuse_key(
    host: &MeshingHostWorkload,
    implementation: Digest,
) -> MeshingExecutionResult<Digest> {
    host.validate()?;
    if host.workload.stage == MeshingStageKind::Publication
        || implementation.bytes().iter().all(|byte| *byte == 0)
    {
        return Err(MeshingExecutionError::Identity(
            "publication or unidentified implementations cannot be reused",
        ));
    }
    let mut bytes = b"runmat-accepted-meshing-stage/v1\0".to_vec();
    bytes.extend_from_slice(implementation.bytes());
    bytes.extend_from_slice(&host.workload.canonical_encode()?);
    Ok(Digest::sha256(bytes))
}

impl AcceptedMeshingStageReceipt {
    /// Immutable object identities used by the native cache lifecycle owner.
    pub fn object_digests(&self) -> impl Iterator<Item = Digest> + '_ {
        self.result_objects
            .iter()
            .map(|reference| reference.logical_digest)
    }

    pub fn retain(
        source: &impl CacheImport,
        destination: &mut impl CacheExport,
        host: &MeshingHostWorkload,
        implementation: Digest,
        success: &AttemptSuccess,
        limits: ObjectInventoryLimits,
    ) -> MeshingExecutionResult<Self> {
        let Some((outputs, result_objects)) = success.values() else {
            return Err(MeshingExecutionError::Identity(
                "accepted stage requires a value result",
            ));
        };
        let [ValuePayload::Object(root)] = outputs else {
            return Err(MeshingExecutionError::Identity(
                "accepted stage requires one object root",
            ));
        };
        let receipt = Self {
            schema_version: 1,
            key: meshing_stage_reuse_key(host, implementation)?,
            access: host.artifact_access.clone(),
            root: root.as_ref().clone(),
            result_objects: result_objects.to_vec(),
        };
        receipt.restore_inner(source, destination, host, implementation, limits, false)?;
        Ok(receipt)
    }

    /// Original observations retain original timing/resource values. A cache hit is
    /// logged by the adapter, never synthesized as fresh numerical execution.
    pub fn restore(
        &self,
        source: &impl CacheImport,
        destination: &mut impl CacheExport,
        host: &MeshingHostWorkload,
        implementation: Digest,
        limits: ObjectInventoryLimits,
    ) -> MeshingExecutionResult<AttemptSuccess> {
        self.restore_inner(source, destination, host, implementation, limits, true)
    }

    fn restore_inner(
        &self,
        source: &impl CacheImport,
        destination: &mut impl CacheExport,
        host: &MeshingHostWorkload,
        implementation: Digest,
        limits: ObjectInventoryLimits,
        record_reuse: bool,
    ) -> MeshingExecutionResult<AttemptSuccess> {
        if self.schema_version != 1 || self.key != meshing_stage_reuse_key(host, implementation)? {
            return Err(MeshingExecutionError::Identity(
                "stage receipt differs from requested dependencies or implementation",
            ));
        }
        rebind_stage_result(
            source,
            destination,
            host,
            AcceptedStageSource {
                root: &self.root,
                access: &self.access,
                result_objects: &self.result_objects,
            },
            limits,
            record_reuse.then_some(
                runmat_meshing_core::ReusedMeshingStageOrigin::ImplementationCache {
                    implementation_digest: runmat_meshing_core::StableDigest::from_bytes(
                        *implementation.bytes(),
                    ),
                },
            ),
        )
    }
}

/// A pool replay is bound to an already accepted invocation, not to the
/// driver's local implementation. Preserve that distinct origin while sharing
/// exactly the same closure validation and observation rebinding as the cache.
pub fn rebind_committed_meshing_stage(
    source: &impl CacheImport,
    destination: &mut impl CacheExport,
    host: &MeshingHostWorkload,
    success: &AttemptSuccess,
    invocation_identity_digest: runmat_meshing_core::StableDigest,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<AttemptSuccess> {
    host.validate()?;
    invocation_identity_digest.validate_nonzero("accepted meshing invocation digest")?;
    let Some((outputs, result_objects)) = success.values() else {
        return Err(MeshingExecutionError::Identity(
            "accepted stage requires a value result",
        ));
    };
    let [ValuePayload::Object(root)] = outputs else {
        return Err(MeshingExecutionError::Identity(
            "accepted stage requires one object root",
        ));
    };
    rebind_stage_result(
        source,
        destination,
        host,
        AcceptedStageSource {
            root,
            access: &host.artifact_access,
            result_objects,
        },
        limits,
        Some(
            runmat_meshing_core::ReusedMeshingStageOrigin::AcceptedInvocation {
                invocation_identity_digest,
            },
        ),
    )
}

struct AcceptedStageSource<'a> {
    root: &'a runmat_execution::value::ValueRef,
    access: &'a ArtifactAccess,
    result_objects: &'a [runmat_execution::value::ValueRef],
}

fn rebind_stage_result(
    source: &impl CacheImport,
    destination: &mut impl CacheExport,
    host: &MeshingHostWorkload,
    accepted: AcceptedStageSource<'_>,
    limits: ObjectInventoryLimits,
    origin: Option<runmat_meshing_core::ReusedMeshingStageOrigin>,
) -> MeshingExecutionResult<AttemptSuccess> {
    let publication =
        import_result_publication(source, accepted.root, accepted.access.clone(), limits)?;
    let references = accepted
        .result_objects
        .iter()
        .map(|reference| (reference.logical_digest, reference))
        .collect::<std::collections::BTreeMap<_, _>>();
    if accepted.result_objects.len() != publication.result_objects().len() + 1
        || references.len() != accepted.result_objects.len()
        || !publication.result_objects().iter().all(|reference| {
            references
                .get(&reference.logical_digest)
                .is_some_and(|original| *original == reference)
        })
    {
        return Err(MeshingExecutionError::Identity(
            "stage receipt does not contain its exact accepted closure",
        ));
    }
    let partition = MeshingPartitionIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: host.workload.stage_identity_digest,
        partition: host.workload.partition.clone(),
    };
    let objects = publication.stage_objects();
    let expected_kind = match host.workload.partition.kind {
        runmat_meshing_core::MeshingPartitionKind::WholeStage => MeshingStageResultKind::WholeStage,
        runmat_meshing_core::MeshingPartitionKind::DeterministicJoin => {
            MeshingStageResultKind::DeterministicJoin
        }
        _ => MeshingStageResultKind::Partition,
    };
    let mut prerequisites = host
        .workload
        .inputs
        .iter()
        .map(|input| input.digest)
        .collect::<Vec<_>>();
    prerequisites.sort_unstable();
    if objects.result_identity.producer_identity_digest != partition.canonical_digest()?
        || objects.result_identity.stage != host.workload.stage
        || objects.result_identity.result_kind != expected_kind
        || objects.manifest.prerequisite_manifest_digests != prerequisites
    {
        return Err(MeshingExecutionError::Identity(
            "cached stage has the wrong producer or prerequisites",
        ));
    }
    let mut evidence = crate::stage_evidence_object::import_stage_evidence_observation_with_access(
        source,
        host,
        accepted.access,
        accepted.result_objects,
        limits,
    )?;
    if let Some(origin) = origin {
        evidence.reused_from = Some(runmat_meshing_core::ReusedMeshingStageObservation {
            original_observation_digest: evidence
                .reused_from
                .as_ref()
                .map_or(evidence.canonical_digest()?, |origin| {
                    origin.original_observation_digest
                }),
            origin,
        });
    }
    let observation =
        crate::stage_evidence_object::prepare_stage_evidence_observation(host, &evidence, limits)?;
    crate::stage_evidence_object::validate_complete_result_inventory(
        &objects.objects,
        &observation.object,
        limits,
    )?;
    for object in &objects.objects {
        destination.write_verified(object)?;
    }
    destination.write_verified(&observation.object)?;
    let rebound = prepare_result_publication(
        publication.into_stage_objects(),
        host.artifact_access.clone(),
        limits,
    )?;
    let mut success = rebound.attempt_success();
    success
        .push_result_object(observation.reference)
        .map_err(MeshingExecutionError::Invalid)?;
    Ok(success)
}

/// Compilation settings of the concrete numerical dependency closure.
/// Source identities and semantic inputs are bound separately by the native cache.
pub fn meshing_build_configuration(stage: MeshingStageKind) -> Digest {
    let mut configurations = vec![
        crate::BUILD_CONFIGURATION,
        runmat_meshing_core::BUILD_CONFIGURATION,
        runmat_meshing_curve::BUILD_CONFIGURATION,
        runmat_meshing_surface::BUILD_CONFIGURATION,
        runmat_geometry_core::BUILD_CONFIGURATION,
    ];
    if !matches!(
        stage,
        MeshingStageKind::Sizing | MeshingStageKind::CurveMesh | MeshingStageKind::SurfaceMesh
    ) {
        configurations.push(runmat_meshing_tetrahedron::BUILD_CONFIGURATION);
    }
    configurations.extend(runmat_meshing_core::numerical_dependency_build_configuration());
    let mut bytes = b"runmat-meshing-stage-build-configurations/v1\0".to_vec();
    for configuration in configurations {
        bytes.extend_from_slice(configuration.as_bytes());
    }
    Digest::sha256(bytes)
}
