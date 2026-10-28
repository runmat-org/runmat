use std::collections::BTreeSet;

use runmat_analysis_core::GeneratorProvenance;
use runmat_analysis_fea::execution::{GeneratorIdentity, SignalMaterializationIdentity};
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{InlineValue, ValuePayload, ValueRefKind};
use runmat_execution::{
    ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableEntrypointKind, ExecutableIdentity,
    ExecutableUnitManifest, ExecutionScopeId, PoolId, ProgramEnvironment, ProgramRevision,
    EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_execution_artifact::{ArtifactAccess, BundleCallable, ProgramExecutionObject};
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
        schema_version: 2,
        value_layout: crate::CanonicalGeneratorValueLayout::SampleMajorRowMajor,
        program_revision: revision(),
        callable: BundleCallable {
            owner_identity: "package@1".into(),
            qualified_name: "loads.pulse".into(),
            source_digest: digest("loads.pulse source"),
        },
        executable_source_digest: digest("executable-source"),
        compiled_binding: crate::named_callable::CompiledCallableBinding::fixture(
            "loads.pulse",
            "loads.pulse",
            ProgramFunctionId(3),
            digest("executable-source"),
        ),
        function: ProgramFunctionId(3),
        frozen_inputs: crate::FrozenGeneratorInputs::DirectCaptures(Vec::new()),
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
        )
        .unwrap(),
        sample_points: runmat_types::physical::Quantity::try_from_descriptor(
            runmat_analysis_core::GeneratorDomain::TimeSeconds
                .axis_descriptor(None)
                .unwrap(),
            vec![0.0, 1.0],
        )
        .unwrap(),
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
                declaration_class: None,
                program: admitted.program_revision.clone(),
                root_package: admitted.callable.owner_identity.clone(),
                entrypoint: admitted.callable.qualified_name.clone(),
                entrypoint_function: Some(admitted.function),
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
            dependencies: Vec::new(),
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

fn chunks() -> Vec<GeneratorChunk> {
    vec![GeneratorChunk {
        ordinal: 0,
        first_sample: 0,
        sample_points: vec![0.0_f64.to_bits(), 1.0_f64.to_bits()],
        arguments: vec![ValuePayload::Inline(Box::new(InlineValue::F64Bits(
            1.0_f64.to_bits(),
        )))],
        identity: digest("chunk"),
    }]
}

fn external_input(scope: &str) -> ProgramExecutionObject {
    ProgramExecutionObject::externalized_value(
        &ArtifactAccess {
            authorization_scope: scope.into(),
            encryption_context: digest(&format!("{scope}-encryption")),
        },
        ValueRefKind::DriverObject,
        ValuePayload::Inline(Box::new(InlineValue::F64Bits(17.0_f64.to_bits()))),
    )
    .unwrap()
}

/// The host a chunk's compiled program requires, as production derives it.
fn chunk_host(chunk: &PreparedGeneratorChunk) -> runmat_execution::host::ExecutionHostRequirement {
    chunk
        .execution_host_requirement(BTreeSet::from([
            runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        ]))
        .unwrap()
}

fn task_context(
    host: runmat_execution::host::ExecutionHostRequirement,
    scope: &[u8],
    pool: &[u8],
) -> GeneratorExecutionContext {
    GeneratorExecutionContext {
        host,
        scope_id: ExecutionScopeId::derive(&[scope]),
        pool_id: PoolId::derive(&[pool]),
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
    }
}

#[test]
fn descriptor_preserves_exact_callable_authority_without_preparing_chunks() {
    let admitted = admitted();
    let executable = executable(&admitted);
    let descriptor = prepare_generator_descriptor(&admitted, executable.clone()).unwrap();
    assert_eq!(
        descriptor.recipe.entrypoint,
        admitted.callable.qualified_name
    );
    assert_eq!(descriptor.recipe.interop, executable.manifest.interop);
    let prepared = prepare_generator_program(&admitted, executable.clone(), chunks(), &[]).unwrap();
    assert_eq!(descriptor, prepared.descriptor);
    assert!(prepare_generator_program(&admitted, executable.clone(), Vec::new(), &[]).is_err());

    let mut foreign = admitted.clone();
    foreign.callable.owner_identity = "foreign@1".into();
    assert!(prepare_generator_descriptor(&foreign, executable.clone()).is_err());
    let mut foreign = admitted.clone();
    foreign.callable.qualified_name = "loads.foreign".into();
    assert!(prepare_generator_descriptor(&foreign, executable.clone()).is_err());
    let mut foreign = admitted.clone();
    foreign.function = ProgramFunctionId(4);
    assert!(prepare_generator_descriptor(&foreign, executable.clone()).is_err());
    let mut foreign = admitted.clone();
    foreign.executable_source_digest = digest("foreign-executable-source");
    assert!(prepare_generator_descriptor(&foreign, executable.clone()).is_err());
    let mut foreign = admitted.clone();
    foreign.program_revision = ProgramRevision::new(
        digest("foreign-graph"),
        digest("foreign-project-source"),
        ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab").unwrap(),
    )
    .unwrap();
    assert!(prepare_generator_descriptor(&foreign, executable.clone()).is_err());
    let mut foreign = admitted.clone();
    foreign
        .required_capabilities
        .insert(runmat_types::CapabilityRequirement::Filesystem);
    assert!(prepare_generator_descriptor(&foreign, executable.clone()).is_err());
    let mut tampered = executable;
    tampered.payloads[0].bytes.push(0);
    assert!(prepare_generator_descriptor(&admitted, tampered).is_err());
}

#[test]
fn portable_program_binds_exact_callable_revision_and_chunk_arguments() {
    let admitted = admitted();
    let prepared =
        prepare_generator_program(&admitted, executable(&admitted), chunks(), &[]).unwrap();
    assert_eq!(prepared.chunks.len(), 1);
    assert_eq!(
        prepared.chunks[0]
            .direct_request()
            .unwrap()
            .callable
            .semantic_function(),
        Some(ProgramFunctionId(3))
    );
    assert_eq!(
        prepared.chunks[0].direct_request().unwrap().arguments,
        prepared.chunks[0].plan.arguments
    );
    assert_eq!(
        prepared.descriptor.artifact.form,
        ExecutableForm::ExecutableUnitV4
    );

    let mut substituted = executable(&admitted);
    substituted.manifest.identity.source_digest = digest("different source");
    assert!(prepare_generator_program(&admitted, substituted, chunks(), &[]).is_err());
}

#[test]
fn portable_program_requires_an_exact_immutable_input_inventory() {
    let admitted = admitted();
    let input = external_input("signal-run");
    let mut referenced_chunks = chunks();
    referenced_chunks[0].arguments = vec![ValuePayload::Object(Box::new(input.reference.clone()))];

    assert!(prepare_generator_program(
        &admitted,
        executable(&admitted),
        referenced_chunks.clone(),
        &[],
    )
    .is_err());

    let prepared = prepare_generator_program(
        &admitted,
        executable(&admitted),
        referenced_chunks,
        std::slice::from_ref(&input),
    )
    .unwrap();
    assert_eq!(prepared.input_references, vec![input.reference.clone()]);

    assert!(
        prepare_generator_program(&admitted, executable(&admitted), chunks(), &[input],).is_err()
    );
}

#[test]
fn ready_input_plan_retains_transport_authority_without_local_object_bytes() {
    use runmat_analysis_core::{
        GeneratorArgument, GeneratorEffectPolicy, PhysicalSignal, SignalArtifactDomain,
        SignalArtifactRef, SignalRepresentation,
    };
    use runmat_execution::value::{
        ValueAdmissionOwnership, ValueAdmissionReceipt, ValueAdmissionTransfer,
        ValueCopyAccounting, VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION,
    };
    use runmat_types::{
        MutationFact, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    let admitted = admitted();
    let input = external_input("signal-run");
    let mut authored = crate::callable::executable::tests::signal();
    let PhysicalSignal::Force(signal) = &mut authored else {
        unreachable!();
    };
    let SignalRepresentation::Generator(generator) = &mut signal.representation else {
        unreachable!();
    };
    generator.invocation.effects = GeneratorEffectPolicy::DeclaredImmutableArtifacts;
    generator.invocation.arguments = vec![GeneratorArgument::Artifact(SignalArtifactRef {
        artifact_id: "immutable-scale".into(),
        content_digest: input.reference.logical_digest.to_string(),
        schema: input.reference.value_schema.clone(),
        domain: SignalArtifactDomain::Constant,
        topology_digest: None,
        sample_count: 1,
    })];
    let mut target = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    target.mutation = MutationFact::Immutable;
    let planned = crate::plan_signal_generator_chunks(&authored, &admitted, None, 8192, |_| {
        Ok(crate::ResolvedSignalArtifact {
            value: ValuePayload::Object(Box::new(input.reference.clone())),
            admission: ValueAdmissionReceipt {
                schema_version: VALUE_ADMISSION_RECEIPT_SCHEMA_VERSION,
                source: target.clone(),
                target: target.clone(),
                ownership: ValueAdmissionOwnership::Moved,
                conversion: None,
                transfer: ValueAdmissionTransfer::None,
                copies: ValueCopyAccounting::default(),
            },
        })
    })
    .unwrap();
    let mut prepared = prepare_signal_materialization(
        authored,
        admitted.clone(),
        executable(&admitted),
        planned.chunks,
        vec![input],
        planned.input_admissions,
        None,
    )
    .unwrap();
    let original = crate::InputMaterializationPlan::from(
        crate::PreparedInputMaterialization::Signal(Box::new(prepared.clone())),
    );
    let expectation = original.expectation().unwrap();
    let descriptor = original.descriptor().unwrap();
    // Native transport registers immutable objects separately and retains only their references
    // in the decoded prepared input. Contract inspection must not demand duplicate local bytes.
    prepared.input_objects.clear();
    let transported = crate::InputMaterializationPlan::from(
        crate::PreparedInputMaterialization::Signal(Box::new(prepared)),
    );
    transported.validate().unwrap();
    assert!(transported.validate_closed().is_err());
    assert_eq!(transported.input_references().len(), 1);
    assert_eq!(transported.expectation().unwrap(), expectation);
    assert_eq!(transported.descriptor().unwrap(), descriptor);
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet4,
    );
    assert_eq!(
        transported.resolve(&mesh).unwrap().expectation().unwrap(),
        expectation.resolve(&mesh).unwrap()
    );
}

#[test]
fn task_identity_is_logical_and_independent_of_pool_placement() {
    let admitted = admitted();
    let prepared =
        prepare_generator_program(&admitted, executable(&admitted), chunks(), &[]).unwrap();
    let first = build_generator_task_submission(
        &admitted,
        &prepared.chunks[0],
        BTreeSet::new(),
        task_context(chunk_host(&prepared.chunks[0]), b"scope-a", b"pool-a"),
    )
    .unwrap();
    assert_eq!(first.request.callable.qualified_name(), "loads.pulse");
    let second = build_generator_task_submission(
        &admitted,
        &prepared.chunks[0],
        BTreeSet::new(),
        task_context(chunk_host(&prepared.chunks[0]), b"scope-b", b"pool-b"),
    )
    .unwrap();
    assert_eq!(first.request.id, second.request.id);
    assert_ne!(first.request.scope_id, second.request.scope_id);
    assert_ne!(first.request.pool_id, second.request.pool_id);
}

#[test]
fn task_identity_tracks_materialization_and_generator_while_callable_is_the_compiled_function() {
    let baseline = admitted();
    let prepared =
        prepare_generator_program(&baseline, executable(&baseline), chunks(), &[]).unwrap();
    let submit = |admitted: &AdmittedSignalGenerator| {
        build_generator_task_submission(
            admitted,
            &prepared.chunks[0],
            BTreeSet::new(),
            task_context(chunk_host(&prepared.chunks[0]), b"scope", b"pool"),
        )
        .unwrap()
    };
    let baseline_task = submit(&baseline);

    let mut sampling_change = baseline.clone();
    sampling_change.materialization_identity = SignalMaterializationIdentity::new(
        digest("signal"),
        Some(baseline.generator_identity),
        digest("changed-sampling"),
    )
    .unwrap();
    let sampling_task = submit(&sampling_change);
    assert_ne!(baseline_task.request.id, sampling_task.request.id);
    assert_eq!(
        baseline_task.request.callable.entrypoint_digest(),
        sampling_task.request.callable.entrypoint_digest()
    );

    let mut generator_change = baseline.clone();
    generator_change.function = ProgramFunctionId(4);
    generator_change.callable.qualified_name = "loads.alternate".into();
    generator_change.generator_identity = GeneratorIdentity::new(
        digest("changed-callable"),
        digest("captures"),
        digest("contract"),
    )
    .unwrap();
    generator_change.materialization_identity = SignalMaterializationIdentity::new(
        digest("signal"),
        Some(generator_change.generator_identity),
        digest("sampling"),
    )
    .unwrap();
    // A new qualified label/function is a new compiled binding, not a
    // permissible rewrite of the baseline admission's immutable authority.
    assert!(
        prepare_generator_program(
            &generator_change,
            executable(&generator_change),
            chunks(),
            &[],
        )
        .is_err(),
        "changing the callable must not reuse the baseline compiled binding"
    );
    generator_change.compiled_binding = crate::named_callable::CompiledCallableBinding::fixture(
        "loads.alternate",
        "loads.alternate",
        ProgramFunctionId(4),
        generator_change.executable_source_digest,
    );
    let generator_program = prepare_generator_program(
        &generator_change,
        executable(&generator_change),
        chunks(),
        &[],
    )
    .unwrap();
    let generator_task = build_generator_task_submission(
        &generator_change,
        &generator_program.chunks[0],
        BTreeSet::new(),
        task_context(chunk_host(&generator_program.chunks[0]), b"scope", b"pool"),
    )
    .unwrap();
    assert_ne!(baseline_task.request.id, generator_task.request.id);
    assert_ne!(
        baseline_task.request.callable.entrypoint_digest(),
        generator_task.request.callable.entrypoint_digest()
    );
}
