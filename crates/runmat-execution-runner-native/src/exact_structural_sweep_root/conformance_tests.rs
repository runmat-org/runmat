use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_analysis_execution::analysis_worker_capabilities;
use runmat_analysis_fea::execution::{StructuralLinearWorkload, StudyIdentity, SweepIdentity};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::ComputeBackend;
use runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT;
use runmat_execution::resource::ResourceInventory;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ExecutionBundleBuilder};
use runmat_execution_runner::{PoolSpec, WorkerSpec};
use runmat_execution_transport_native::control::ResourceRequest;
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
};
use runmat_meshing_core::{MeshingRequest, MetricTensor3};
use runmat_meshing_execution::exact_meshing_worker_capabilities;

use crate::{
    import_exact_structural_sweep_result, native_meshing_kernel_dispatcher,
    prepare_exact_meshing_root, prepare_exact_structural_study_root,
    run_remote_meshing_worker_quic, ExactMeshingRootPreparation,
    ExactStructuralStudyRootPreparation, ExactStructuralSweepTerminal, QuicRemoteWorkerChannel,
    RemoteMeshingWorkerQuicRequest, RemotePoolDriver, RemoteWorkerChannelConfig,
    RemoteWorkerQuicRequest,
};

use super::{prepare_exact_structural_sweep_root, ExactStructuralSweepRootPreparation};

const BOX: &[u8] = include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep");

#[tokio::test(flavor = "current_thread")]
async fn exact_structural_sweep_reaches_ordered_admitted_results_over_encrypted_quic() {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"remote-exact-structural-study\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let project = runmat_package::build_frozen_project(
        &project_root.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let project_identity = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_identity.graph_digest.bytes()),
        Digest::from_bytes(*project_identity.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let access = ArtifactAccess {
        authorization_scope: "remote-exact-structural-study-run".into(),
        encryption_context: Digest::sha256(b"remote-exact-structural-study-encryption"),
    };
    let meshing_request = coarse_box_request();
    let exact_meshing = prepare_exact_meshing_root(ExactMeshingRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        source_name: "box.brep",
        source_bytes: BOX,
        import_options: ExactCadImportOptions::default(),
        contact_definitions: Vec::new(),
        meshing_request: meshing_request.clone(),
        preferred_edges_per_partition: 3,
        preferred_faces_per_partition: 2,
        maximum_chunk_bytes: 1024,
    })
    .unwrap();
    let source_digest = Digest::sha256(BOX);
    let workload = box_structural_workload();
    let first_study_digest = Digest::sha256(b"box structural study A");
    let second_study_digest = Digest::sha256(b"box structural study B");
    let invalid_study_digest = Digest::sha256(b"box structural study invalid boundary");
    let first = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        authored_study_digest: first_study_digest,
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: source_digest,
        workload_template: workload.clone(),
        exact_meshing: exact_meshing.clone(),
    })
    .unwrap();
    let second = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        authored_study_digest: second_study_digest,
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: source_digest,
        workload_template: workload,
        exact_meshing: exact_meshing.clone(),
    })
    .unwrap();
    let mut invalid_workload = box_structural_workload();
    invalid_workload.model.constraints[0].visit_targets_mut(|target| {
        *target = runmat_analysis_core::RegionTarget::boundary("face_999999");
    });
    let invalid = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        authored_study_digest: invalid_study_digest,
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: source_digest,
        workload_template: invalid_workload,
        exact_meshing,
    })
    .unwrap();
    let frozen_project_digest = Some(Digest::from_bytes(*project_identity.graph_digest.bytes()));
    let identity = SweepIdentity::new(
        Digest::sha256(b"encrypted structural sweep"),
        vec![
            StudyIdentity::new(first_study_digest, frozen_project_digest).unwrap(),
            StudyIdentity::new(second_study_digest, frozen_project_digest).unwrap(),
            StudyIdentity::new(invalid_study_digest, frozen_project_digest).unwrap(),
        ],
        Digest::sha256(b"collect all outcomes"),
    )
    .unwrap();
    let prepared = prepare_exact_structural_sweep_root(ExactStructuralSweepRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        identity,
        fail_fast: false,
        studies: vec![first, second, invalid],
    })
    .unwrap();
    let expectation = prepared.result_expectation();

    let mut builder = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_compiled_package_closure();
    for program in prepared.required_programs() {
        builder = builder.with_materialized_program(
            program.recipe,
            program.artifact.form,
            program.artifact.executable_bytes,
        );
    }
    let mut bundle = Vec::new();
    write_bundle(
        &builder.build().unwrap(),
        &mut bundle,
        ArchiveLimits::default(),
    )
    .unwrap();

    let imported = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    let document = imported.geometry_document().unwrap();
    let cohort = crate::meshing_platform::native_meshing_capability_cohort();
    let mut capabilities =
        exact_meshing_worker_capabilities(&document, &meshing_request, Some(&cohort)).unwrap();
    capabilities.extend(analysis_worker_capabilities(
        std::iter::empty(),
        std::iter::empty(),
        false,
    ));
    let resources = ResourceRequest {
        cpu_millicores: 500,
        memory_bytes: 1 << 30,
        scratch_bytes: 1 << 30,
        accelerator_count: 0,
        accelerator_class: None,
        accelerator_memory_bytes: 0,
        maximum_wall_millis: 180_000,
    };
    let inventory = ResourceInventory {
        cpu_millicores: 1_000,
        memory_bytes: resources.memory_bytes,
        scratch_bytes: resources.scratch_bytes,
        accelerators: Vec::new(),
        capabilities,
    };
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"exact-study-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"exact-study-pool"]);
    let worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[b"exact-study-worker"]),
        pool_id,
        resources: inventory.clone(),
    };

    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let certificate_der = cert.der().to_vec();
    let listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key.serialize_der(),
        FrameLimits::default(),
    )
    .unwrap();
    let authority = listener.local_addr().unwrap();
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([53; 32]).unwrap();
    let server = run_remote_meshing_worker_quic(RemoteMeshingWorkerQuicRequest {
        worker: RemoteWorkerQuicRequest {
            listener,
            run_identity: access.authorization_scope.clone(),
            worker: worker.clone(),
            driver_fence: 59,
            session_id: [61; 16],
            run_key: run_key.clone(),
            limits: FrameLimits::default(),
        },
        kernel: Arc::new(native_meshing_kernel_dispatcher()),
        meshing_limits: crate::NativeMeshingHostLimits::default(),
    });
    let client = async {
        let channel = QuicRemoteWorkerChannel::connect(
            RemoteWorkerChannelConfig {
                run_identity: access.authorization_scope.clone(),
                node_identity: "remote-exact-structural-study-node".into(),
                worker: worker.clone(),
                driver_fence: 59,
                session_id: [61; 16],
                run_key,
                limits: FrameLimits::default(),
            },
            &PinnedQuicEndpoint {
                authority,
                server_name: "runmat.execution".into(),
                certificate_der,
            },
        )
        .await
        .unwrap();
        let pool = RemotePoolDriver::new_with_value_scope(
            scope_id,
            PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: 1,
                max_in_flight: 2,
                resource_limit: inventory,
            },
            59,
            bundle,
            access.authorization_scope.clone(),
        )
        .unwrap();
        pool.add_worker(channel.clone()).await.unwrap();
        for object in prepared.transport_objects() {
            pool.register_execution_object(object.reference, object.bytes)
                .unwrap();
        }

        let outcome = tokio::time::timeout(
            Duration::from_secs(180),
            crate::remote::exact_structural_sweep::execute(
                Arc::clone(&pool),
                prepared.into_request(),
                access.authorization_scope.clone(),
                resources,
            ),
        )
        .await
        .expect("remote exact structural sweep timed out")
        .unwrap();
        let crate::remote::pool_execution::RemotePoolExecutionOutcome::Completed(completed) =
            outcome
        else {
            panic!("remote exact structural sweep did not complete")
        };
        let mut wrong_request = expectation.clone();
        wrong_request.studies[0]
            .expected_meshing_request
            .deterministic_seed ^= 1;
        assert!(import_exact_structural_sweep_result(
            pool.as_ref(),
            completed.response.clone(),
            &wrong_request,
            ObjectInventoryLimits::default(),
        )
        .is_err());
        let mut incomplete = completed.response.clone();
        if let runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
            result_objects,
            ..
        } = &mut incomplete
        {
            result_objects.pop();
        }
        assert!(import_exact_structural_sweep_result(
            pool.as_ref(),
            incomplete,
            &expectation,
            ObjectInventoryLimits::default(),
        )
        .is_err());
        let sweep = import_exact_structural_sweep_result(
            pool.as_ref(),
            completed.response,
            &expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        assert_eq!(sweep.identity, identity);
        assert_eq!(sweep.entries.len(), 3);
        let mut entries = sweep.entries.into_iter();
        let first = entries.next().unwrap();
        let ExactStructuralSweepTerminal::Succeeded(result) = first.terminal else {
            panic!("first remote structural sweep entry did not succeed")
        };
        let second = entries.next().unwrap();
        let ExactStructuralSweepTerminal::Succeeded(second_result) = second.terminal else {
            panic!(
                "second remote structural sweep entry did not succeed: {:?}",
                second.terminal
            )
        };
        assert_eq!(
            second_result.analysis.run.identity.family(),
            runmat_analysis_core::PhysicsFamily::Structural
        );
        let invalid = entries.next().unwrap();
        assert!(matches!(
            invalid.terminal,
            ExactStructuralSweepTerminal::Failed(message)
                if message.contains("face_999999")
        ));
        assert!(!result.mesh.artifact.topology.volume_elements.is_empty());
        result
            .mesh
            .evidence
            .validate(&result.mesh.artifact)
            .unwrap();
        result.analysis.validate().unwrap();
        assert!(!result.analysis.run.fields_are_empty());
        let displacement = result
            .analysis
            .run
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .expect("remote structural result must contain displacement");
        assert_eq!(
            displacement.shape,
            vec![result.mesh.artifact.topology.nodes.len(), 3]
        );
        assert_eq!(
            result.analysis.run.identity.family(),
            runmat_analysis_core::PhysicsFamily::Structural
        );
        assert_eq!(
            result.analysis.run.identity.procedure(),
            runmat_analysis_core::AnalysisProcedure::Static
        );
        assert!(result
            .boundary_resolution
            .aliases
            .contains_key("face_000001"));
        let constrained_face = &result.boundary_resolution.aliases["face_000001"];
        let loaded_face = &result.boundary_resolution.aliases["face_000002"];
        assert_eq!(
            result.resolved_workload.model.constraints[0]
                .target()
                .unwrap()
                .region_id,
            constrained_face.analysis_selection_id().unwrap()
        );
        let runmat_analysis_core::ExcitationTarget::Region(load_target) =
            &result.resolved_workload.model.excitations[0]
                .excitation()
                .target
        else {
            panic!("remote structural fixture must retain its region excitation")
        };
        assert_eq!(
            load_target.region_id,
            loaded_face.analysis_selection_id().unwrap()
        );
        for selected in [constrained_face, loaded_face] {
            assert!(result
                .mesh
                .artifact
                .topology
                .boundary_faces
                .iter()
                .flat_map(|face| &face.provenance)
                .any(|identity| identity == selected));
        }
        let mut missing_alias = box_structural_workload();
        missing_alias.model.constraints[0].visit_targets_mut(|target| {
            *target = runmat_analysis_core::RegionTarget::boundary("face_999999");
        });
        let aliases = runmat_analysis_core::BoundaryRegionAliases::new(
            result.boundary_resolution.aliases.clone(),
        )
        .unwrap();
        assert!(matches!(
            runmat_analysis_core::resolve_boundary_region_aliases(&missing_alias.model, &aliases),
            Err(runmat_analysis_core::BoundaryRegionResolutionError::UnresolvedImporterAlias {
                alias
            }) if alias == "face_999999"
        ));
        pool.remove_worker(worker.id, false).await.unwrap();
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.unwrap();
}

fn box_structural_workload() -> StructuralLinearWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_id = "box".into();
    model.geometry_revision = 1;
    model.material_assignments.clear();
    model.constraints[0].visit_targets_mut(|target| {
        *target = runmat_analysis_core::RegionTarget::boundary("face_000001");
    });
    if let runmat_analysis_core::ExcitationTarget::Region(target) =
        &mut model.excitations[0].excitation_mut().target
    {
        *target = runmat_analysis_core::RegionTarget::boundary("face_000002");
    } else {
        panic!("cantilever fixture must use a region excitation");
    }
    StructuralLinearWorkload {
        schema_version: 1,
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
    }
}

fn coarse_box_request() -> MeshingRequest {
    let mut request = crate::resolve_native_meshing_request(
        &ExactCadImportOptions::default(),
        runmat_meshing_core::MeshingRequestSettings::default(),
    )
    .unwrap();
    request.metric.global_metric = MetricTensor3::isotropic_length_m(10.0).unwrap();
    request.quality.curve.maximum_chordal_deviation_m = 0.1;
    request.quality.curve.maximum_tangent_change_degrees = 180.0;
    request.quality.curve.minimum_metric_edge_length = 0.01;
    request.quality.curve.maximum_metric_edge_length = 10.0;
    request.quality.surface.minimum_metric_angle_degrees = 0.1;
    request.quality.surface.maximum_physical_aspect_ratio = 1_000.0;
    request.quality.surface.maximum_chordal_deviation_m = 0.1;
    request.quality.surface.maximum_normal_deviation_degrees = 180.0;
    request.quality.volume.maximum_metric_edge_length = 2.0;
    request.quality.volume.maximum_radius_edge_ratio = 10.0;
    request.quality.volume.minimum_scaled_jacobian = 0.01;
    request.resources.maximum_nodes = 10_000;
    request.resources.maximum_elements = 10_000;
    request.resources.maximum_search_work = 1_000_000;
    request.resources.maximum_iterations = 1_000_000;
    request.resources.maximum_wall_time_ms = 180_000;
    request
}
