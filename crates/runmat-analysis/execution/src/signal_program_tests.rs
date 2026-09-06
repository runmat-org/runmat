use std::collections::BTreeSet;

use runmat_analysis_core::GeneratorProvenance;
use runmat_analysis_fea::execution::{GeneratorIdentity, SignalMaterializationIdentity};
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::{
    ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableIdentity, ExecutableUnitManifest, ProgramEnvironment,
    ProgramRevision, EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_execution_artifact::BundleCallable;
use runmat_types::{
    CapabilitySet, InteropManifest, ParallelManifest, ProgramFunctionId,
    INTEROP_MANIFEST_SCHEMA_VERSION, PARALLEL_MANIFEST_SCHEMA_VERSION,
    REGION_CONTRACT_SCHEMA_VERSION,
};

use super::*;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("graph"),
        digest("project-source"),
        ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab").unwrap(),
    )
    .unwrap()
}

fn admitted() -> AdmittedSignalGenerator {
    AdmittedSignalGenerator {
        program_revision: revision(),
        callable: BundleCallable {
            owner_identity: "package@1".into(),
            qualified_name: "loads.pulse".into(),
            source_digest: digest("loads.pulse source"),
        },
        executable_source_digest: digest("executable-source"),
        function: ProgramFunctionId(3),
        generator_identity: GeneratorIdentity::new(
            digest("callable"),
            digest("captures"),
            digest("contract"),
        )
        .unwrap(),
        materialization_identity: SignalMaterializationIdentity::new(
            digest("signal"),
            None,
            digest("sampling"),
            Vec::new(),
        )
        .unwrap(),
        sample_points: vec![0.0, 1.0],
        required_capabilities: BTreeSet::new(),
        provenance: GeneratorProvenance {
            project_revision: "project".into(),
            callable_id: "callable".into(),
            frozen_input_digest: "inputs".into(),
        },
    }
}

fn executable(admitted: &AdmittedSignalGenerator) -> ExecutableUnitEnvelope {
    let payloads = ExecutableComponentKind::REQUIRED
        .into_iter()
        .map(|kind| {
            ExecutableComponentPayload::new(kind, format!("{kind:?}").into_bytes()).unwrap()
        })
        .collect::<Vec<_>>();
    let components = payloads
        .iter()
        .map(|payload| {
            ExecutableComponentDescriptor::from_payload(payload.kind, 1, &payload.bytes).unwrap()
        })
        .collect();
    ExecutableUnitEnvelope::new(
        ExecutableUnitManifest {
            schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
            identity: ExecutableIdentity {
                program: admitted.program_revision.clone(),
                root_package: admitted.callable.owner_identity.clone(),
                entrypoint: admitted.callable.qualified_name.clone(),
                entrypoint_function: admitted.function,
                entrypoint_kind: ExecutableEntrypointKind::Function,
                source_digest: admitted.executable_source_digest,
            },
            revisions: ExecutableComponentRevisions {
                catalog_schema: 1,
                catalog_fingerprint: *admitted.program_revision.catalog_fingerprint(),
                contract_schema: 1,
                contract_fingerprint: digest("contracts"),
                analysis_schema: 1,
                mir_schema: 1,
                bytecode_schema: 1,
                vm_layout_schema: 1,
                function_registry_schema: 1,
                source_map_schema: 1,
                region_schema: REGION_CONTRACT_SCHEMA_VERSION,
                interop_schema: INTEROP_MANIFEST_SCHEMA_VERSION,
                parallel_schema: PARALLEL_MANIFEST_SCHEMA_VERSION,
            },
            components,
            capabilities: CapabilitySet::default(),
            regions: Vec::new(),
            interop: InteropManifest::empty(),
            parallel: ParallelManifest::empty(),
            optional_sections: Vec::new(),
        },
        payloads,
    )
    .unwrap()
}

fn chunks() -> Vec<SignalGeneratorChunk> {
    vec![SignalGeneratorChunk {
        ordinal: 0,
        first_sample: 0,
        sample_points: vec![0.0_f64.to_bits(), 1.0_f64.to_bits()],
        arguments: vec![ValuePayload::Inline(Box::new(InlineValue::F64Bits(
            1.0_f64.to_bits(),
        )))],
        identity: digest("chunk"),
    }]
}

#[test]
fn portable_program_binds_exact_callable_revision_and_chunk_arguments() {
    let admitted = admitted();
    let prepared =
        prepare_signal_generator_program(&admitted, executable(&admitted), chunks(), Vec::new())
            .unwrap();
    assert_eq!(prepared.chunks.len(), 1);
    assert_eq!(prepared.chunks[0].request.function, 3);
    assert_eq!(
        prepared.chunks[0].request.arguments,
        prepared.chunks[0].plan.arguments
    );
    assert_eq!(
        prepared.descriptor.artifact.form,
        ExecutableForm::ExecutableUnitV3
    );

    let mut substituted = executable(&admitted);
    substituted.manifest.identity.source_digest = digest("different source");
    assert!(
        prepare_signal_generator_program(&admitted, substituted, chunks(), Vec::new()).is_err()
    );
}

#[test]
fn task_identity_is_logical_and_independent_of_pool_placement() {
    let admitted = admitted();
    let prepared =
        prepare_signal_generator_program(&admitted, executable(&admitted), chunks(), Vec::new())
            .unwrap();
    let context = |scope, pool| SignalGeneratorExecutionContext {
        scope_id: scope,
        pool_id: pool,
        resources: ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 64 * 1024 * 1024,
            scratch_bytes: 0,
            max_wall_millis: 10_000,
            max_artifact_bytes: 64 * 1024 * 1024,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
        },
        retry: RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    };
    let first = build_signal_generator_task_submission(
        &admitted,
        &prepared.chunks[0],
        BTreeSet::new(),
        context(
            ExecutionScopeId::derive(&[b"scope-a"]),
            PoolId::derive(&[b"pool-a"]),
        ),
    )
    .unwrap();
    assert_eq!(first.request.callable.qualified_name, "loads.pulse");
    let second = build_signal_generator_task_submission(
        &admitted,
        &prepared.chunks[0],
        BTreeSet::new(),
        context(
            ExecutionScopeId::derive(&[b"scope-b"]),
            PoolId::derive(&[b"pool-b"]),
        ),
    )
    .unwrap();
    assert_eq!(first.request.id, second.request.id);
    assert_ne!(first.request.scope_id, second.request.scope_id);
    assert_ne!(first.request.pool_id, second.request.pool_id);
}
