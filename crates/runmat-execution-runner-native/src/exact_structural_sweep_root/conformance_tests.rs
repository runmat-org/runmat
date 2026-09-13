use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_analysis_core::{
    GeneratorDomain, GeneratorEffectPolicy, PhysicalSignal, SignalExtrapolation, SignalGenerator,
    SignalInterpolation, SignalRepresentation, SignalSamplingPolicy,
};
use runmat_analysis_execution::{
    admit_named_signal_generator, analysis_worker_capabilities, plan_signal_generator_chunks,
    prepare_signal_materialization, PreparedSignalMaterialization, ResolvedSignalArtifact,
};
use runmat_analysis_fea::execution::{
    SignalMaterializationIdentity, StructuralLinearWorkload, StructuralTransientBlockPlan,
    StructuralTransientWorkload, StudyIdentity, SweepCompositionPolicy, SweepDefinition,
};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy;
use runmat_analysis_fea::ComputeBackend;
use runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT;
use runmat_execution::resource::{Capability, ResourceInventory};
use runmat_execution::value::ValueLimits;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, BundleCallable, ExecutionBundleBuilder};
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
    import_exact_structural_study_result, import_exact_structural_sweep_result,
    native_meshing_kernel_dispatcher, prepare_exact_meshing_root,
    prepare_exact_structural_study_root, run_remote_meshing_worker_quic,
    ExactMeshingRootPreparation, ExactStructuralStudyRootPreparation, ExactStructuralSweepTerminal,
    QuicRemoteWorkerChannel, RemoteMeshingWorkerQuicRequest, RemotePoolDriver,
    RemoteWorkerChannelConfig, RemoteWorkerQuicRequest, StructuralTransientRecoveryPublisher,
};

use super::{prepare_exact_structural_sweep_root, ExactStructuralSweepRootPreparation};

const BOX: &[u8] = include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep");

#[derive(Default)]
struct RecordingRecoveryPublisher {
    closures: Vec<runmat_analysis_execution::StructuralTransientRecoveryClosure>,
}

#[async_trait::async_trait]
impl StructuralTransientRecoveryPublisher for RecordingRecoveryPublisher {
    async fn publish(
        &mut self,
        closure: runmat_analysis_execution::StructuralTransientRecoveryClosure,
    ) -> crate::NativeExecutionResult<()> {
        self.closures.push(closure);
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn exact_structural_sweep_reaches_ordered_admitted_results_over_encrypted_quic() {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"remote-exact-structural-study\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let generator_source = "function y = pulse(t, magnitude)\narguments\n    t (:,1) double\n    magnitude (:,1) double\nend\ny = t * [0, 0, 0];\ny(:, 2) = magnitude(1);\nend\n";
    std::fs::write(project_root.path().join("src/pulse.m"), generator_source).unwrap();
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
    let (workload, signal_materialization) = generated_box_workload(
        &project,
        revision.clone(),
        project_root.path().join("src/pulse.m"),
        generator_source,
        &access,
    )
    .await;
    let first_study_digest = Digest::sha256(b"box structural study A");
    let second_study_digest = Digest::sha256(b"box structural study B");
    let invalid_study_digest = Digest::sha256(b"box structural study invalid boundary");
    let first = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        authored_study_digest: first_study_digest,
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: source_digest,
        workloads: runmat_analysis_execution::StructuralStudyWorkloads::LinearSequence {
            workloads: independent_structural_workloads(&workload),
        },
        analysis_accelerator_requests: Vec::new(),
        signal_materializations: vec![signal_materialization.clone()],
        exact_meshing: exact_meshing.clone(),
    })
    .unwrap();
    let second = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        authored_study_digest: second_study_digest,
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: source_digest,
        workloads: runmat_analysis_execution::StructuralStudyWorkloads::LinearSequence {
            workloads: std::collections::BTreeMap::from([(
                workload.step_id.clone(),
                workload.clone(),
            )]),
        },
        analysis_accelerator_requests: Vec::new(),
        signal_materializations: vec![signal_materialization.clone()],
        exact_meshing: exact_meshing.clone(),
    })
    .unwrap();
    let direct_study_digest = Digest::sha256(b"box direct transient study");
    let direct_workload = box_direct_transient_workload();
    let direct = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        authored_study_digest: direct_study_digest,
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: source_digest,
        workloads: runmat_analysis_execution::StructuralStudyWorkloads::DirectTransient {
            workload: Box::new(direct_workload),
            block_plan: StructuralTransientBlockPlan::new(1).unwrap(),
        },
        analysis_accelerator_requests: Vec::new(),
        signal_materializations: Vec::new(),
        exact_meshing: exact_meshing.clone(),
    })
    .unwrap();
    let direct_expectation = direct.result_expectation();
    let mut invalid_workload = workload;
    invalid_workload.model.constraints[0].visit_targets_mut(|target| {
        *target = runmat_analysis_core::RegionTarget::boundary("face_999999");
    });
    let invalid = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        authored_study_digest: invalid_study_digest,
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: source_digest,
        workloads: runmat_analysis_execution::StructuralStudyWorkloads::LinearSequence {
            workloads: std::collections::BTreeMap::from([(
                invalid_workload.step_id.clone(),
                invalid_workload,
            )]),
        },
        analysis_accelerator_requests: Vec::new(),
        signal_materializations: vec![signal_materialization],
        exact_meshing: exact_meshing.clone(),
    })
    .unwrap();
    let definition = SweepDefinition::new(
        Digest::sha256(b"encrypted structural sweep"),
        vec![
            StudyIdentity::from_authored_digest(first_study_digest, None).unwrap(),
            StudyIdentity::from_authored_digest(second_study_digest, None).unwrap(),
            StudyIdentity::from_authored_digest(invalid_study_digest, None).unwrap(),
        ],
        SweepCompositionPolicy::CollectAll,
    )
    .unwrap();
    let identity = definition.identity().unwrap();
    let prepared = prepare_exact_structural_sweep_root(ExactStructuralSweepRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        definition,
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
    capabilities.insert(Capability::ProcessIsolation);
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
                resource_limit: inventory.clone(),
            },
            59,
            bundle.clone(),
            access.authorization_scope.clone(),
        )
        .unwrap();
        pool.add_worker(channel.clone()).await.unwrap();
        for object in prepared.transport_objects() {
            if object.is_externalized_value() {
                pool.register_value_object(object.reference, object.bytes)
                    .unwrap();
            } else {
                pool.register_execution_object(object.reference, object.bytes)
                    .unwrap();
            }
        }

        let outcome = tokio::time::timeout(
            Duration::from_secs(180),
            crate::remote::exact_structural_sweep::execute(
                Arc::clone(&pool),
                prepared.into_request(),
                access.authorization_scope.clone(),
                resources.clone(),
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
        assert_eq!(
            pool.snapshot()
                .tasks
                .values()
                .filter(|task| task.submission.request.callable.qualified_name == "loads.pulse")
                .count(),
            1,
            "identical signal materializations must execute once across a sweep"
        );
        let diagnostic_sweep = import_exact_structural_sweep_result(
            pool.as_ref(),
            completed.response.clone(),
            &expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        assert!(
            matches!(
                diagnostic_sweep.entries[0].terminal,
                ExactStructuralSweepTerminal::Succeeded(_)
            ),
            "first remote structural sweep entry did not succeed: {:?}",
            diagnostic_sweep.entries[0].terminal
        );
        assert!(
            matches!(
                diagnostic_sweep.entries[1].terminal,
                ExactStructuralSweepTerminal::Succeeded(_)
            ),
            "second remote structural sweep entry did not succeed: {:?}",
            diagnostic_sweep.entries[1].terminal
        );
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
        let mut wrong_signal = expectation.clone();
        wrong_signal.studies[0].signal_materializations[0]
            .admitted
            .materialization_identity = SignalMaterializationIdentity::new(
            Digest::sha256(b"substituted signal"),
            None,
            Digest::sha256(b"substituted sampling"),
        )
        .unwrap();
        assert!(import_exact_structural_sweep_result(
            pool.as_ref(),
            completed.response.clone(),
            &wrong_signal,
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
        let crate::ExactStructuralStudyAnalysisResult::LinearSequence {
            resolved_workloads,
            sequence,
        } = &result.analysis
        else {
            panic!("first remote structural sweep entry returned a non-linear result")
        };
        assert_eq!(sequence.steps.len(), 2);
        assert_eq!(sequence.states.len(), 2);
        assert_eq!(sequence.steps[1].step_id.as_str(), "step_static_second");
        let second = entries.next().unwrap();
        let ExactStructuralSweepTerminal::Succeeded(second_result) = second.terminal else {
            panic!(
                "second remote structural sweep entry did not succeed: {:?}",
                second.terminal
            )
        };
        let crate::ExactStructuralStudyAnalysisResult::LinearSequence {
            sequence: second_sequence,
            ..
        } = &second_result.analysis
        else {
            panic!("second remote structural sweep entry returned a non-linear result")
        };
        assert_eq!(
            second_sequence.steps[0].result.run.identity.family(),
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
        let analysis = &sequence.steps[0].result;
        let resolved_workload = resolved_workloads.values().next().unwrap();
        analysis.validate().unwrap();
        assert!(!analysis.run.fields_are_empty());
        let displacement = analysis
            .run
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .expect("remote structural result must contain displacement");
        assert_eq!(
            displacement.shape,
            vec![result.mesh.artifact.topology.nodes.len(), 3]
        );
        assert_eq!(
            analysis.run.identity.family(),
            runmat_analysis_core::PhysicsFamily::Structural
        );
        assert_eq!(
            analysis.run.identity.procedure(),
            runmat_analysis_core::AnalysisProcedure::Static
        );
        assert!(resolved_workload
            .model
            .signals
            .iter()
            .all(|signal| signal.generator().is_none()));
        assert!(result
            .boundary_resolution
            .aliases
            .contains_key("face_000001"));
        let constrained_face = &result.boundary_resolution.aliases["face_000001"];
        let loaded_face = &result.boundary_resolution.aliases["face_000002"];
        assert_eq!(
            resolved_workload.model.constraints[0]
                .target()
                .unwrap()
                .region_id,
            constrained_face.analysis_selection_id().unwrap()
        );
        let runmat_analysis_core::ExcitationTarget::Region(load_target) =
            &resolved_workload.model.excitations[0].excitation().target
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
    run_direct_exact_study_conformance(
        direct,
        direct_expectation,
        bundle,
        inventory,
        resources,
        access,
    )
    .await;
}

async fn run_direct_exact_study_conformance(
    direct: crate::PreparedExactStructuralStudyRoot,
    expectation: crate::ExactStructuralStudyResultExpectation,
    bundle: Vec<u8>,
    inventory: ResourceInventory,
    resources: ResourceRequest,
    access: ArtifactAccess,
) {
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"direct-exact-study-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"direct-exact-study-pool"]);
    let worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[b"direct-exact-study-worker"]),
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
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([67; 32]).unwrap();
    let server = run_remote_meshing_worker_quic(RemoteMeshingWorkerQuicRequest {
        worker: RemoteWorkerQuicRequest {
            listener,
            run_identity: access.authorization_scope.clone(),
            worker: worker.clone(),
            driver_fence: 71,
            session_id: [73; 16],
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
                node_identity: "direct-exact-structural-study-node".into(),
                worker: worker.clone(),
                driver_fence: 71,
                session_id: [73; 16],
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
            71,
            bundle,
            access.authorization_scope.clone(),
        )
        .unwrap();
        pool.add_worker(channel).await.unwrap();
        for object in direct.transport_objects().unwrap() {
            if object.is_externalized_value() {
                pool.register_value_object(object.reference, object.bytes)
                    .unwrap();
            } else {
                pool.register_execution_object(object.reference, object.bytes)
                    .unwrap();
            }
        }
        let mut recovery = RecordingRecoveryPublisher::default();
        let response = tokio::time::timeout(
            Duration::from_secs(180),
            crate::remote::exact_structural_study::execute_for_test(
                Arc::clone(&pool),
                direct.request(),
                &access.authorization_scope,
                &resources,
                &mut recovery,
            ),
        )
        .await
        .expect("remote direct-transient exact study timed out")
        .unwrap()
        .unwrap_or_else(|message| panic!("remote direct-transient exact study failed: {message}"));
        let imported = import_exact_structural_study_result(
            pool.as_ref(),
            response,
            &expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let crate::ExactStructuralStudyAnalysisResult::DirectTransient {
            resolved_workload,
            result,
            numerical_receipt,
        } = imported.analysis
        else {
            panic!("direct-transient exact study returned a linear result")
        };
        assert_eq!(resolved_workload.step_id.as_str(), "transient");
        assert_eq!(result.samples().len(), 3);
        assert_eq!(numerical_receipt.blocks.len(), 2);
        numerical_receipt.validate_for_result(&result).unwrap();
        assert_eq!(recovery.closures.len(), 2);
        pool.remove_worker(worker.id, false).await.unwrap();
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.unwrap();
}

fn independent_structural_workloads(
    template: &StructuralLinearWorkload,
) -> std::collections::BTreeMap<runmat_analysis_core::ProcedureStepId, StructuralLinearWorkload> {
    let first = template
        .model
        .sequence
        .steps
        .first()
        .expect("structural fixture must contain a procedure step")
        .clone();
    let second = runmat_analysis_core::ProcedureStep::structural_static("step_static_second");
    let mut model = template.model.clone();
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "two-independent-static-steps",
        vec![first.clone(), second.clone()],
    );
    let mut first_workload = template.clone();
    first_workload.model = model.clone();
    first_workload.step_id = first.step_id.clone();
    let mut second_workload = template.clone();
    second_workload.model = model;
    second_workload.step_id = second.step_id.clone();
    std::collections::BTreeMap::from([
        (first.step_id, first_workload),
        (second.step_id, second_workload),
    ])
}

async fn generated_box_workload(
    project: &runmat_package::FrozenProject,
    revision: ProgramRevision,
    source_path: std::path::PathBuf,
    source_text: &str,
    access: &ArtifactAccess,
) -> (StructuralLinearWorkload, PreparedSignalMaterialization) {
    let mut workload = box_structural_workload();
    let mut authored = workload
        .model
        .signals
        .first()
        .expect("cantilever fixture must contain its force signal")
        .clone();
    let PhysicalSignal::Force(force) = &mut authored else {
        panic!("cantilever fixture signal must be force")
    };
    force.signal_id = runmat_analysis_core::SignalId("generated_force_history".into());
    let admitted_input = runmat_runtime::analysis::value_admission::admit_owned_real64(
        runmat_value::Value::Tensor(runmat_value::Tensor::new(vec![-1_000.0], vec![1, 1]).unwrap()),
        runmat_runtime::analysis::value_admission::AnalysisValueAdmissionPolicy::exact_host(),
    )
    .unwrap();
    let (immutable_value, input_admission) = admitted_input.into_value_payload().unwrap();
    let immutable_object = runmat_execution_artifact::ProgramExecutionObject::externalized_value(
        access,
        runmat_execution::value::ValueRefKind::DriverObject,
        immutable_value,
    )
    .unwrap();
    force.representation = SignalRepresentation::Generator(SignalGenerator {
        callable: runmat_analysis_core::GeneratorCallable::ProjectFunction {
            qualified_name: "loads.pulse".into(),
        },
        arguments: vec![runmat_analysis_core::GeneratorArgument::Artifact(
            runmat_analysis_core::SignalArtifactRef {
                artifact_id: "magnitude".into(),
                content_digest: immutable_object.reference.logical_digest.to_string(),
                schema: "runmat-value-payload-v1".into(),
                domain: runmat_analysis_core::SignalArtifactDomain::Constant,
                topology_digest: None,
                sample_count: 1,
            },
        )],
        domain: GeneratorDomain::TimeSeconds,
        sampling: SignalSamplingPolicy::Uniform {
            start: 0.0,
            end: 1.0,
            sample_count: 5,
        },
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
        effects: GeneratorEffectPolicy::DeclaredImmutableArtifacts,
    });
    force.evidence.sample_count = 0;
    // The executable envelope is currently a static structural solve. Keep its referenced force
    // constant: a static procedure must not silently choose a point from a time-domain history.
    // The additional generated history still exercises shared materialization and deduplication.
    workload.model.signals.push(authored.clone());

    let symbols = runmat_package::source_symbols_from_frozen(project, &source_path);
    let definition = symbols
        .definitions
        .iter()
        .find(|definition| definition.name == "pulse")
        .expect("frozen pulse definition");
    let source_id = definition
        .source_id
        .as_ref()
        .expect("frozen pulse source id");
    let owner_identity = source_id.package_instance.to_string();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
        .unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::frozen(
                owner_identity.clone(),
                source_id.relative_path.as_str(),
                source_path.to_string_lossy(),
                source_text,
            ),
            Some(revision.clone()),
        )
        .await
        .unwrap();
    let executable = unit.portable_envelope_for(Some("pulse")).unwrap();
    let function = unit
        .function_analysis(executable.manifest.identity.entrypoint_function)
        .cloned()
        .expect("pulse static facts");
    let executable_source_digest = unit.revision().source_digest.parse::<Digest>().unwrap();
    let admitted = admit_named_signal_generator(
        &authored,
        revision,
        BundleCallable {
            owner_identity,
            qualified_name: "loads.pulse".into(),
            source_digest: Digest::sha256(source_text),
        },
        executable_source_digest,
        &function,
        &BTreeSet::new(),
    )
    .unwrap();
    let plan = plan_signal_generator_chunks(
        &authored,
        &admitted,
        None,
        usize::try_from(ValueLimits::default().max_inline_bytes).unwrap(),
        |_| {
            Ok(ResolvedSignalArtifact {
                value: runmat_execution::value::ValuePayload::Object(Box::new(
                    immutable_object.reference.clone(),
                )),
                admission: input_admission.clone(),
            })
        },
    )
    .unwrap();
    let prepared = prepare_signal_materialization(
        authored,
        admitted,
        executable,
        plan.chunks,
        vec![immutable_object],
        plan.input_admissions,
    )
    .unwrap();
    (workload, prepared)
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
    let step_id = model.sequence.steps[0].step_id.clone();
    StructuralLinearWorkload {
        schema_version: 1,
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        initial_guess_transition: None,
    }
}

fn box_direct_transient_workload() -> StructuralTransientWorkload {
    let mut model = box_structural_workload().model;
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "box-direct-transient-sequence",
        vec![runmat_analysis_core::ProcedureStep::structural_transient(
            "transient",
        )],
    );
    StructuralTransientWorkload {
        schema_version: 1,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
        residual_target: 1.0e-8,
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
