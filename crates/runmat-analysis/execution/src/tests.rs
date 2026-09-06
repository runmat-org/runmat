use std::collections::BTreeMap;
use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AccuracyIdentity, AnalysisNumericalObjectKind, AnalysisWorkKind, ConstraintPlanIdentity,
    DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity,
    ProblemIdentity, SequenceIdentity, StepIdentity, StudyIdentity,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ExecutionScopeId, PoolId, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace, ProgramExecutionObjectLocation};

use super::*;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn execution_identities() -> (StudyIdentity, ExecutionPlanIdentity) {
    let study = StudyIdentity::new(digest("study"), Some(digest("project"))).unwrap();
    let sequence = SequenceIdentity::new(study, digest("sequence")).unwrap();
    let step = StepIdentity::new(sequence, digest("step")).unwrap();
    let constraints =
        ConstraintPlanIdentity::new(step, digest("mesh"), digest("constraints")).unwrap();
    let problem =
        ProblemIdentity::new(step, digest("model"), None, constraints, Vec::new()).unwrap();
    let discretization =
        DiscretizationIdentity::new(problem, digest("mesh"), digest("formulation")).unwrap();
    let operator = OperatorIdentity::new(discretization, digest("operator")).unwrap();
    let accuracy = AccuracyIdentity::new(step, digest("accuracy")).unwrap();
    let numerical = NumericalPlanIdentity::new(operator, digest("solver"), accuracy).unwrap();
    (
        study,
        ExecutionPlanIdentity::new(numerical, digest("execution")).unwrap(),
    )
}

fn access() -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: "study:test".into(),
        encryption_context: digest("encryption"),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("graph"),
        digest("source"),
        ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab").unwrap(),
    )
    .unwrap()
}

#[test]
fn object_closure_and_host_request_round_trip_without_inlining_payloads() {
    let payload = vec![7_u8; 65];
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        digest("mesh-identity"),
        &payload,
        16,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(objects.objects.len(), 6);
    let root = objects
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let (study, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
        execution_plan,
        AnalysisWorkKind::Solve,
        vec![AnalysisWorkloadInput {
            kind: objects.manifest.kind,
            logical_identity: objects.manifest.logical_identity,
            payload_digest: objects.manifest.payload_digest,
            manifest_digest: objects.root.digest,
            manifest_encoded_length: objects.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let host = AnalysisHostWorkload::new(workload, access()).unwrap();
    let request = host.program_request(revision(), &[root]).unwrap();
    assert!(request.artifact.executable_bytes.len() < payload.len() * 20);
    assert_eq!(
        AnalysisHostWorkload::from_program_request(&request).unwrap(),
        host
    );
}

#[test]
fn portable_program_packages_every_manifest_and_chunk_under_one_access_scope() {
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        digest("portable-mesh"),
        b"a manifest-plus-chunk mesh payload",
        7,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let (study, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
        execution_plan,
        AnalysisWorkKind::Solve,
        vec![AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let host = AnalysisHostWorkload::new(workload, access()).unwrap();
    let prepared =
        prepare_analysis_program(&host, revision(), std::slice::from_ref(&input)).unwrap();
    assert_eq!(prepared.objects.len(), input.objects.len());
    assert!(prepared.objects.iter().all(|object| {
        object.reference.authorization_scope == host.artifact_access.authorization_scope
            && object.reference.kind == ValueRefKind::DriverObject
    }));
    let locations = prepared
        .objects
        .iter()
        .enumerate()
        .map(|(index, object)| ProgramExecutionObjectLocation {
            reference: object.reference.clone(),
            artifact_id: format!("input-artifact-{index}"),
        })
        .collect();
    let inputs = prepared.inputs(locations).unwrap();
    assert_eq!(inputs.arguments, prepared.request.arguments);
    inputs.validate().unwrap();

    let mut wrong = input;
    wrong.manifest.logical_identity = digest("substituted");
    assert!(prepare_analysis_program(&host, revision(), &[wrong]).is_err());
}

#[test]
fn workload_rejects_identity_and_access_tampering() {
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Signal,
        digest("signal-identity"),
        b"signal samples",
        4,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = objects
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let (study, execution_plan) = execution_identities();
    let mut workload = AnalysisWorkloadRequest::new(
        study,
        execution_plan,
        AnalysisWorkKind::MaterializeSignal,
        vec![AnalysisWorkloadInput {
            kind: objects.manifest.kind,
            logical_identity: objects.manifest.logical_identity,
            payload_digest: objects.manifest.payload_digest,
            manifest_digest: objects.root.digest,
            manifest_encoded_length: objects.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::Signal],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    workload.inputs[0].payload_digest = digest("tampered");
    assert!(workload.validate().is_err());

    workload.inputs[0].payload_digest = objects.manifest.payload_digest;
    let host = AnalysisHostWorkload::new(workload, access()).unwrap();
    let mut wrong_root = root;
    wrong_root.authorization_scope = "other-study".into();
    assert!(host.program_request(revision(), &[wrong_root]).is_err());
}

#[test]
fn worker_capability_inventory_uses_generic_execution_capabilities() {
    let formulation = digest("formulation");
    let capabilities =
        analysis_worker_capabilities([formulation], ["gpu.compute".to_string()], true);
    assert!(
        capabilities.contains(&runmat_execution::resource::Capability::Accelerator(
            "gpu.compute".into()
        ))
    );
    assert!(
        capabilities.contains(&runmat_execution::resource::Capability::Custom(format!(
            "runmat.analysis.formulation:{formulation}"
        )))
    );
}

#[test]
fn logical_task_identity_is_independent_of_pool_placement() {
    let (study, execution_plan) = execution_identities();
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::AccuracyEvidence,
        digest("accuracy-input"),
        b"accuracy evidence input",
        4096,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = input
        .root_reference(&access(), ValueRefKind::DriverObject)
        .unwrap();
    let workload = AnalysisWorkloadRequest::new(
        study,
        execution_plan,
        AnalysisWorkKind::EvaluateAccuracy,
        vec![AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::AccuracyEvidence],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let capabilities = workload.generic_capabilities().unwrap();
    let context = |pool: &[u8]| AnalysisExecutionContext {
        scope_id: ExecutionScopeId::derive(&[b"study-scope"]),
        pool_id: PoolId::derive(&[pool]),
        program_artifact_id: runmat_execution::identity::ArtifactId::derive(&[b"analysis-program"]),
        resources: runmat_execution::resource::ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 1024 * 1024,
            scratch_bytes: 1024,
            max_wall_millis: 60_000,
            max_artifact_bytes: 1024 * 1024,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: capabilities.clone(),
        },
        retry: runmat_execution::task::RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    };
    let first = build_analysis_task_submission(
        &workload,
        vec![root.clone()],
        Vec::new(),
        BTreeSet::new(),
        context(b"pool-a"),
    )
    .unwrap();
    let second = build_analysis_task_submission(
        &workload,
        vec![root],
        Vec::new(),
        BTreeSet::new(),
        context(b"pool-b"),
    )
    .unwrap();
    assert_ne!(first.request.pool_id, second.request.pool_id);
    assert_eq!(first.request.id, second.request.id);
}

#[derive(Default)]
struct MemoryObjects(BTreeMap<Digest, Vec<u8>>);

impl runmat_execution_artifact::cache::CacheImport for MemoryObjects {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
}

impl runmat_execution_artifact::cache::CacheExport for MemoryObjects {
    fn write_verified(
        &mut self,
        object: &runmat_execution_artifact::LogicalObject,
    ) -> runmat_execution_artifact::ArtifactResult<()> {
        object.validate()?;
        if self
            .0
            .insert(object.descriptor.digest, object.bytes.clone())
            .is_some_and(|prior| prior != object.bytes)
        {
            return Err(runmat_execution_artifact::ArtifactError::Identity(
                "memory object collision".into(),
            ));
        }
        Ok(())
    }
}

struct FieldKernel;

impl AnalysisKernel for FieldKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::Solve);
        assert_eq!(invocation.inputs[0].payload(), b"canonical mesh bytes");
        Ok(vec![AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::Field,
            logical_identity: digest("field-identity"),
            payload: b"canonical field bytes".to_vec(),
        }])
    }
}

#[test]
fn serial_execution_imports_closed_inputs_and_publishes_closed_results() {
    let access = access();
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        digest("mesh-identity"),
        b"canonical mesh bytes",
        5,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = input
        .root_reference(&access, ValueRefKind::DriverObject)
        .unwrap();
    let mut store = MemoryObjects::default();
    for object in &input.objects {
        runmat_execution_artifact::cache::CacheExport::write_verified(&mut store, object).unwrap();
    }
    let (study, execution_plan) = execution_identities();
    let workload = AnalysisWorkloadRequest::new(
        study,
        execution_plan,
        AnalysisWorkKind::Solve,
        vec![AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        }],
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )
    .unwrap();
    let host = AnalysisHostWorkload::new(workload, access).unwrap();
    let request = host.program_request(revision(), &[root]).unwrap();
    let completed = execute_serial_analysis(
        &request,
        &mut store,
        &FieldKernel,
        6,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(completed.outputs[0].payload(), b"canonical field bytes");
    let response = AnalysisHostResponse::completed(&host, &completed).unwrap();
    response.validate_against(&host).unwrap();
    assert!(matches!(
        response.program_response(),
        runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess { outputs, result_objects }
            if outputs.len() == 1 && result_objects.len() == 5
    ));
}

#[test]
fn serial_execution_rejects_missing_and_tampered_chunks() {
    let access = access();
    let input = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        digest("mesh-identity"),
        b"canonical mesh bytes",
        5,
        ObjectNamespace::InputValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let root = input
        .root_reference(&access, ValueRefKind::DriverObject)
        .unwrap();
    let mut store = MemoryObjects::default();
    for object in &input.objects {
        runmat_execution_artifact::cache::CacheExport::write_verified(&mut store, object).unwrap();
    }
    let missing = input.objects[2].descriptor.digest;
    store.0.remove(&missing);
    assert!(import_analysis_objects(
        &store,
        &root,
        input.manifest.kind,
        input.manifest.logical_identity,
        input.manifest.payload_digest,
        &access,
        ObjectInventoryLimits::default(),
    )
    .is_err());
}

struct StructuralResultKernel(Vec<u8>);

impl AnalysisKernel for StructuralResultKernel {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        assert_eq!(invocation.kind, AnalysisWorkKind::Solve);
        Ok(vec![AnalysisKernelOutput {
            kind: AnalysisNumericalObjectKind::SemanticState,
            logical_identity: invocation.host.workload.execution_plan.digest(),
            payload: self.0.clone(),
        }])
    }
}

#[test]
fn prepared_structural_study_closes_and_revalidates_its_real_result() {
    use runmat_analysis_fea::execution::{
        StructuralLinearWorkload, StructuralLinearWorkloadResult,
    };
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
    use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
    use runmat_analysis_fea::{
        run_linear_static_with_options, ComputeBackend, LinearStaticSolveOptions,
    };
    use runmat_execution::value::ValuePayload;
    use runmat_execution_artifact::ProgramExecutionResponse;
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest,
    };

    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let solve = StructuralLinearWorkload {
        schema_version: 1,
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
    };
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: if index == 0 { "root" } else { "tip" }.into(),
            assembly_path: vec!["root".into()],
        });
        face.provenance.sort();
    }
    let assignment = solve.model.material_assignments.first().unwrap();
    let region = PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: assignment.region_id.clone(),
        assembly_path: vec!["root".into()],
    };
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access(),
        workload: solve.clone(),
        mesh: mesh.clone(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    assert_eq!(prepared.inputs.len(), 2);
    assert_eq!(prepared.program.request.arguments.len(), 2);
    assert!(prepared
        .program
        .request
        .arguments
        .iter()
        .all(|input| matches!(input, ValuePayload::Object(_))));

    let raw = run_linear_static_with_options(
        &solve.model,
        solve.backend,
        LinearStaticSolveOptions {
            formulation_requirement: solve.formulation_requirement,
            preconditioner_kind: solve.preconditioner,
            algebra_backend_kind: solve.algebra_backend,
            solver_mesh_artifact_path: None,
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: solve.require_solver_mesh_for_solid,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        },
    )
    .unwrap();
    let encoded = StructuralLinearWorkloadResult {
        schema_version: 1,
        run: raw.clone(),
    }
    .canonical_bytes()
    .unwrap();
    let mut store = MemoryObjects::default();
    for input in &prepared.inputs {
        for object in &input.objects {
            runmat_execution_artifact::cache::CacheExport::write_verified(&mut store, object)
                .unwrap();
        }
    }
    let completed = execute_serial_analysis(
        &prepared.program.request,
        &mut store,
        &StructuralResultKernel(encoded),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&prepared.host, &completed)
        .unwrap()
        .program_response();
    let decoded = import_structural_linear_result(
        &store,
        response.clone(),
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(decoded.run, raw);

    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        mut result_objects,
    } = response
    else {
        unreachable!()
    };
    result_objects.push(result_objects[0].clone());
    assert!(import_structural_linear_result(
        &store,
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        },
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .is_err());

    let access_variant = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "another-run-authority".into(),
            encryption_context: digest("another-encryption-context"),
        },
        workload: solve.clone(),
        mesh: mesh.clone(),
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    assert_eq!(prepared.host.workload, access_variant.host.workload);
    assert_eq!(
        prepared.program.request.recipe,
        access_variant.program.request.recipe
    );
    assert_eq!(
        prepared.program.request.artifact,
        access_variant.program.request.artifact
    );
    assert_ne!(
        prepared.program.request.arguments,
        access_variant.program.request.arguments
    );

    let mut backend_variant = solve.clone();
    backend_variant.backend = ComputeBackend::Gpu;
    let backend_variant = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access(),
        workload: backend_variant,
        mesh,
        chunk_bytes: 4096,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    assert_ne!(
        prepared.host.workload.execution_plan,
        backend_variant.host.workload.execution_plan
    );

    let mut wrong_backend = raw;
    wrong_backend.requested_backend = ComputeBackend::Gpu;
    let encoded = StructuralLinearWorkloadResult {
        schema_version: 1,
        run: wrong_backend,
    }
    .canonical_bytes()
    .unwrap();
    let completed = execute_serial_analysis(
        &prepared.program.request,
        &mut store,
        &StructuralResultKernel(encoded),
        4096,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&prepared.host, &completed)
        .unwrap()
        .program_response();
    assert!(import_structural_linear_result(
        &store,
        response,
        &prepared,
        ObjectInventoryLimits::default(),
    )
    .is_err());
}
