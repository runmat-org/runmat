use super::*;
use runmat_analysis_fea::execution::StructuralLinearWorkload;
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{
    resolve_meshing_request, GeometryTolerancePolicy, MeshingRequestSettings,
};

pub(crate) fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

pub(crate) fn access(authority: &str) -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: authority.into(),
        encryption_context: Digest::sha256(authority.as_bytes()),
    }
}

fn workload() -> StructuralLinearWorkload {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let step_id = model.sequence.steps[0].step_id.clone();
    StructuralLinearWorkload {
        schema_version: 2,
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: false,
        state_input: None,
    }
}

fn workloads(
) -> std::collections::BTreeMap<runmat_analysis_core::ProcedureStepId, StructuralLinearWorkload> {
    let workload = workload();
    std::collections::BTreeMap::from([(workload.step_id.clone(), workload)])
}

pub(crate) fn prepared(
    authority: &str,
    expected_source: Digest,
    authored_study_digest: Digest,
) -> PreparedExactStructuralStudyRoot {
    prepare_exact_structural_study_root(preparation(
        authority,
        expected_source,
        authored_study_digest,
    ))
    .unwrap()
}

fn preparation(
    authority: &str,
    expected_source: Digest,
    authored_study_digest: Digest,
) -> ExactStructuralStudyRootPreparation {
    preparation_with_revision(
        authority,
        expected_source,
        authored_study_digest,
        revision(),
    )
}

fn preparation_with_revision(
    authority: &str,
    expected_source: Digest,
    authored_study_digest: Digest,
    revision: ProgramRevision,
) -> ExactStructuralStudyRootPreparation {
    let source = b"portable exact CAD source";
    let access = access(authority);
    let import_options = ExactCadImportOptions::default();
    let meshing_request = resolve_meshing_request(
        GeometryTolerancePolicy {
            source_tolerance_m: runmat_types::physical::Length::new(0.0),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-12),
            model_relative_term: 1.0e-12,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-4),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
        },
        MeshingRequestSettings::default(),
    )
    .unwrap();
    let exact_meshing = crate::prepare_exact_meshing_root(crate::ExactMeshingRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        source_name: "part.brep",
        source_bytes: source,
        import_options,
        contact_definitions: Vec::new(),
        meshing_request,
        preferred_edges_per_partition: 8,
        preferred_faces_per_partition: 8,
        maximum_chunk_bytes: 8,
    })
    .unwrap();
    ExactStructuralStudyRootPreparation {
        program_revision: revision,
        artifact_access: access,
        authored_study_digest,
        frozen_project_digest: Some(Digest::sha256(b"project")),
        geometry_source_digest: expected_source,
        workloads: runmat_analysis_execution::StructuralStudyWorkloads::LinearSequence {
            workloads: workloads(),
        },
        analysis_accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        input_materializations: Vec::new(),
        exact_meshing,
    }
}

#[tokio::test]
async fn exact_root_retains_deferred_intent_and_derives_mesh_bound_expectations() {
    use runmat_analysis_core::{
        AuthoredSignalFieldLocation, AuthoredSignalTopologySelector, GeneratorCallable,
        GeneratorDomain, GeneratorEffectPolicy, GeneratorInvocation, PhysicalSignal,
        SignalExtrapolation, SignalGenerator, SignalInterpolation, SignalRepresentation,
        SignalSamplingPolicy, SignalShape,
    };
    use runmat_analysis_execution::{
        FrozenDeferredSignal, InputMaterializationExpectation, InputMaterializationPlan,
        InputMaterializationPlanExpectation, StructuralStudyWorkloads,
    };
    use std::collections::BTreeSet;

    // This is a root transport/authority test, not numerical field-consumer qualification.
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("runmat.toml"),
        "[package]\nname = \"exact-deferred-input\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    // Keep this transport-only callable genuinely admissible under Pure.
    // reshape is cooperatively suspending and must not be relabeled pure by
    // this fixture. The axis identity is intentionally not a numerical field
    // generator: this test never invokes it or certifies its output shape.
    // Deferred topology resolution and field-consumer execution are separate.
    let source_text = "function y = field(t)\narguments\n    t (:,1) double\nend\ny = t;\nend\n";
    let source_path = directory.path().join("src/field.m");
    std::fs::write(&source_path, source_text).unwrap();
    let project = runmat_package::build_frozen_project(
        &directory.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let project_revision = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_revision.graph_digest.bytes()),
        Digest::from_bytes(*project_revision.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let symbols = runmat_package::source_symbols_from_frozen(&project, &source_path);
    let definition = symbols
        .definitions
        .iter()
        .find(|definition| definition.name == "field")
        .unwrap();
    let source_id = definition.source_id.as_ref().unwrap();
    let owner = source_id.package_instance.to_string();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
        .await
        .unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::frozen(
                owner.clone(),
                source_id.relative_path.as_str(),
                source_path.to_string_lossy(),
                source_text,
            ),
            Some(revision.clone()),
        )
        .await
        .unwrap();
    let executable = unit.portable_envelope_for(Some("field")).unwrap();
    let mut preparation = preparation_with_revision(
        "deferred-authority",
        Digest::sha256(b"portable exact CAD source"),
        Digest::sha256(b"deferred-study"),
        revision.clone(),
    );
    let StructuralStudyWorkloads::LinearSequence { workloads } = &mut preparation.workloads else {
        unreachable!()
    };
    let workload = workloads.values_mut().next().unwrap();
    let mut authored = workload.model.signals[0].clone();
    let PhysicalSignal::Force(force) = &mut authored else {
        panic!("fixture force signal")
    };
    force.evidence.sample_count = 0;
    force.representation = SignalRepresentation::Generator(SignalGenerator {
        invocation: GeneratorInvocation {
            callable: GeneratorCallable::ProjectFunction {
                qualified_name: "loads.field".into(),
            },
            arguments: Vec::new(),
            effects: GeneratorEffectPolicy::Pure,
        },
        domain: GeneratorDomain::TimeSeconds,
        sampling: SignalSamplingPolicy::Uniform {
            start: 0.0,
            end: 1.0,
            sample_count: 3,
        },
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    authored = authored.with_shape(SignalShape::DeferredTopologyField {
        selector: AuthoredSignalTopologySelector::StudySolverMesh {
            location: AuthoredSignalFieldLocation::Node,
        },
        component_count: 3,
    });
    let frozen = FrozenDeferredSignal::prepare(
        authored.clone(),
        revision,
        runmat_execution_artifact::BundleCallable {
            owner_identity: owner,
            qualified_name: "loads.field".into(),
            source_digest: Digest::sha256(source_text),
        },
        executable,
        BTreeSet::new(),
        8192,
    )
    .unwrap();
    workload.model.signals[0] = authored.clone();
    preparation.input_materializations = vec![InputMaterializationPlan::DeferredSignal(Box::new(
        frozen.clone(),
    ))];
    let prepared = prepare_exact_structural_study_root(preparation).unwrap();
    let frozen_descriptor = frozen.descriptor().unwrap();
    assert!(prepared.required_programs().unwrap().iter().any(|program| {
        program.recipe == frozen_descriptor.recipe && program.artifact == frozen_descriptor.artifact
    }));
    let mut store =
        crate::FilesystemObjectStore::open(directory.path().join("store"), 64 * 1024 * 1024)
            .unwrap();
    prepared.write_to(&mut store).unwrap();
    let decoded = decode_exact_structural_study_root(prepared.request(), &store).unwrap();
    assert_eq!(
        decoded.input_materializations,
        vec![InputMaterializationPlan::DeferredSignal(Box::new(
            frozen.clone()
        ))]
    );
    assert_eq!(
        &decoded.workloads.authored_model().unwrap().signals[0],
        &authored
    );
    let expected = prepared.result_expectation();
    let InputMaterializationPlanExpectation::DeferredSignal(retained) =
        &expected.input_materializations[0]
    else {
        panic!("original deferred authority")
    };
    retained.validate_for(&frozen).unwrap();
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet4,
    );
    let foreign = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet10,
    );
    let resolved = expected.input_materializations[0].resolve(&mesh).unwrap();
    let InputMaterializationExpectation::Signal(signal) = &resolved else {
        panic!("resolved signal")
    };
    assert_ne!(signal.authored, authored);
    assert_eq!(
        signal.admitted.generator_identity,
        frozen.admitted().generator_identity
    );
    signal.validate_for_mesh(&mesh).unwrap();
    assert!(signal.validate_for_mesh(&foreign).is_err());
    frozen.validate_resolved_expectation(signal, &mesh).unwrap();
    assert!(frozen
        .validate_resolved_expectation(signal, &foreign)
        .is_err());
    let mut rewritten = expected;
    rewritten.input_materializations[0] =
        InputMaterializationPlanExpectation::Ready(Box::new(resolved));
    assert!(
        rewritten.validate().is_err(),
        "resolved declaration cannot replace retained original source authority"
    );
}

#[test]
fn whole_study_root_binds_accelerator_requests_to_the_authored_backend() {
    let source_digest = Digest::sha256(b"portable exact CAD source");
    let mut request = preparation("run-accelerator", source_digest, Digest::sha256(b"study"));
    request.analysis_accelerator_requests = vec![runmat_execution::resource::AcceleratorRequest {
        class: runmat_execution::resource::AcceleratorClass::new("gpu.test").unwrap(),
        count: 1,
        minimum_allocation_bytes: 1024,
        provider: None,
        required_features: Default::default(),
    }];
    assert!(prepare_exact_structural_study_root(request).is_err());

    let mut request = preparation(
        "run-missing-accelerator",
        source_digest,
        Digest::sha256(b"gpu-study"),
    );
    let runmat_analysis_execution::StructuralStudyWorkloads::LinearSequence { workloads } =
        &mut request.workloads
    else {
        unreachable!("test preparation is linear")
    };
    for workload in workloads.values_mut() {
        workload.backend = ComputeBackend::Gpu;
        workload.algebra_backend = LinearAlgebraBackendKind::Accelerator;
    }
    assert!(prepare_exact_structural_study_root(request).is_err());
}

#[test]
fn whole_study_root_is_path_free_and_reuses_fixed_programs() {
    let source_digest = Digest::sha256(b"portable exact CAD source");
    let first = prepared("run-a", source_digest, Digest::sha256(b"study"));
    let second = prepared("run-b", source_digest, Digest::sha256(b"study"));
    assert_eq!(
        first.required_programs().unwrap(),
        second.required_programs().unwrap()
    );
    assert_eq!(first.request().recipe, second.request().recipe);
    assert_ne!(first.request().arguments, second.request().arguments);
    assert_eq!(first.required_programs().unwrap().len(), 4);
    assert_eq!(
        first.transport_objects().unwrap().len(),
        first.objects().len()
    );

    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path(),
        64 * 1024 * 1024,
    )
    .unwrap();
    first.write_to(&mut store).unwrap();
    let decoded = decode_exact_structural_study_root(first.request(), &store).unwrap();
    assert_eq!(decoded.geometry_source_digest, source_digest);
    assert_eq!(
        decoded.workloads,
        runmat_analysis_execution::StructuralStudyWorkloads::LinearSequence {
            workloads: workloads(),
        }
    );
    assert_eq!(decoded.artifact_access, access("run-a"));
}

#[test]
fn whole_study_root_rejects_a_substituted_source_identity() {
    let prepared = prepared(
        "run",
        Digest::sha256(b"different source"),
        Digest::sha256(b"study"),
    );
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path(),
        64 * 1024 * 1024,
    )
    .unwrap();
    prepared.write_to(&mut store).unwrap();
    assert!(decode_exact_structural_study_root(prepared.request(), &store).is_err());
}
