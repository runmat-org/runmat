use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_analysis_core::{
    DirectionalCombination, GeneratorDomain, GeneratorEffectPolicy, IndependentDomain,
    ModalCombination, ModalTruncationCorrection, ModalTruncationPolicy as SpectrumTruncationPolicy,
    PhysicalSignal, ProcedureStep, ProcedureStepId, ReferenceStatePolicy, ResponseSpectrum,
    ResponseSpectrumId, RigidResponsePolicy, SequenceDependency, SequenceDependencyReason,
    SignalExtrapolation, SignalGenerator, SignalInterpolation, SignalRepresentation,
    SignalSamplingPolicy, SpectrumAbscissa, SpectrumConstruction, SpectrumDamping,
    SpectrumDirection, SpectrumEnvelope, SpectrumEvidence, SpectrumExtrapolation,
    SpectrumInterpolation, SpectrumResponse, StateId, StateTransition, StateTransitionMethod,
    TopologyTransitionPolicy, TransitionId,
};
use runmat_analysis_execution::{
    admit_named_signal_generator, analysis_worker_capabilities, plan_signal_generator_chunks,
    prepare_signal_materialization, PreparedSignalMaterialization, ResolvedSignalArtifact,
};
use runmat_analysis_fea::execution::{
    SignalMaterializationIdentity, StructuralBucklingSolverPolicy, StructuralBucklingTemplate,
    StructuralFrequencySolverPlan, StructuralFrequencyWorkload, StructuralFrequencyWorkloadInput,
    StructuralLinearWorkload, StructuralModalResponseSpectrumTemplate,
    StructuralModalTransientTemplate, StructuralModalWorkload, StructuralModalWorkloadInput,
    StructuralPrestressedFrequencySolverTemplate, StructuralPrestressedFrequencyTemplate,
    StructuralPrestressedModalTemplate, StructuralStepWorkload, StructuralTransientBlockPlan,
    StructuralTransientWorkload, StudyIdentity, SweepCompositionPolicy, SweepDefinition,
};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::frequency::FrequencyResponsePlan;
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy;
use runmat_analysis_fea::ComputeBackend;
use runmat_analysis_fea::FEA_FIELD_STRUCTURAL_DISPLACEMENT;
use runmat_analysis_fea::{ModalConvergencePolicy, ModalTruncationPolicy};
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

mod prestressed_frequency;
mod prestressed_modal;
mod random_response;
mod response_spectrum;

const BOX: &[u8] = include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep");
const EXACT_SWEEP_CONFORMANCE_STACK_BYTES: usize = 16 * 1024 * 1024;

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

#[test]
fn exact_structural_sweep_reaches_ordered_admitted_results_over_encrypted_quic() {
    std::thread::Builder::new()
        .name("exact-structural-sweep-conformance".into())
        .stack_size(EXACT_SWEEP_CONFORMANCE_STACK_BYTES)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("exact structural sweep conformance runtime")
                .block_on(run_exact_structural_sweep_conformance());
        })
        .expect("exact structural sweep conformance thread")
        .join()
        .expect("exact structural sweep conformance thread panicked");
}

async fn run_exact_structural_sweep_conformance() {
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
        import_options: ExactCadImportOptions {
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            ..ExactCadImportOptions::default()
        },
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
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        input_materializations: vec![
            runmat_analysis_execution::PreparedInputMaterialization::Signal(Box::new(
                signal_materialization.clone(),
            ))
            .into(),
        ],
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
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        input_materializations: vec![
            runmat_analysis_execution::PreparedInputMaterialization::Signal(Box::new(
                signal_materialization.clone(),
            ))
            .into(),
        ],
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
        workloads: runmat_analysis_execution::StructuralStudyWorkloads::StructuralTransient {
            workload: Box::new(direct_workload),
            block_plan: StructuralTransientBlockPlan::new(1).unwrap(),
        },
        analysis_accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        input_materializations: Vec::new(),
        exact_meshing: exact_meshing.clone(),
    })
    .unwrap();
    let direct = Box::new(direct);
    let direct_expectation = direct.result_expectation();
    let frequency_study_digest = Digest::sha256(b"box structural frequency study");
    let frequency = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        authored_study_digest: frequency_study_digest,
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: source_digest,
        workloads: runmat_analysis_execution::StructuralStudyWorkloads::StructuralFrequency {
            workload: Box::new(box_structural_frequency_workload()),
        },
        analysis_accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        input_materializations: Vec::new(),
        exact_meshing: exact_meshing.clone(),
    })
    .unwrap();
    let frequency = Box::new(frequency);
    let frequency_expectation = frequency.result_expectation();
    let modal_study_digest = Digest::sha256(b"box structural modal study");
    let modal = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        authored_study_digest: modal_study_digest,
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: source_digest,
        workloads: runmat_analysis_execution::StructuralStudyWorkloads::StructuralModal {
            workload: Box::new(box_structural_modal_workload()),
        },
        analysis_accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        input_materializations: Vec::new(),
        exact_meshing: exact_meshing.clone(),
    })
    .unwrap();
    let modal = Box::new(modal);
    let modal_expectation = modal.result_expectation();
    let modal_transient_study_digest = Digest::sha256(b"box modal transient study");
    let modal_transient =
        prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access.clone(),
            authored_study_digest: modal_transient_study_digest,
            frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
            geometry_source_digest: source_digest,
            workloads: box_modal_transient_workloads(),
            analysis_accelerator_requests: Vec::new(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            input_materializations: Vec::new(),
            exact_meshing: exact_meshing.clone(),
        })
        .unwrap();
    let modal_transient = Box::new(modal_transient);
    let modal_transient_expectation = modal_transient.result_expectation();
    let buckling_study_digest = Digest::sha256(b"box structural buckling study");
    let buckling = Box::new(
        prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access.clone(),
            authored_study_digest: buckling_study_digest,
            frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
            geometry_source_digest: source_digest,
            workloads: box_buckling_workloads(),
            analysis_accelerator_requests: Vec::new(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            input_materializations: Vec::new(),
            exact_meshing: exact_meshing.clone(),
        })
        .unwrap(),
    );
    let buckling_expectation = buckling.result_expectation();
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
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        input_materializations: vec![
            runmat_analysis_execution::PreparedInputMaterialization::Signal(Box::new(
                signal_materialization,
            ))
            .into(),
        ],
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

    let mut builder = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_project_runtime_policy_object(crate::remote::runtime_settings::fixture_policy(
            &revision,
        ))
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
        &ExactCadImportOptions {
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            ..ExactCadImportOptions::default()
        },
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
        cpu_millicores: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial()
            .checked_cpu_millicores()
            .unwrap()
            .into(),
        memory_bytes: 1 << 30,
        scratch_bytes: 1 << 30,
        accelerators: Vec::new(),
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
        host: crate::test_support::host_inventory(),
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
        let Ok(completed) = outcome else {
            panic!("remote exact structural sweep did not complete")
        };
        assert_eq!(
            pool.snapshot()
                .tasks
                .values()
                .filter(|task| task.submission.request.callable.qualified_name() == "loads.pulse")
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
        let runmat_analysis_execution::InputMaterializationPlanExpectation::Ready(input) =
            &mut wrong_signal.studies[0].input_materializations[0]
        else {
            panic!("expected ready input materialization");
        };
        let runmat_analysis_execution::InputMaterializationExpectation::Signal(signal) =
            input.as_mut()
        else {
            panic!("expected signal materialization");
        };
        signal.admitted.materialization_identity = SignalMaterializationIdentity::new(
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
        let constrained_face = &box_selectors().fixed_face;
        let loaded_face = &box_selectors().loaded_face;
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
        pool.remove_worker(worker.id, false).await.unwrap();
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.unwrap();
    run_direct_exact_study_conformance(DirectExactStudyConformance {
        direct,
        direct_expectation,
        frequency,
        frequency_expectation,
        modal,
        modal_expectation,
        modal_transient,
        modal_transient_expectation,
        buckling,
        buckling_expectation,
        bundle,
        inventory,
        resources,
        access,
    })
    .await;
}

struct DirectExactStudyConformance {
    direct: Box<crate::PreparedExactStructuralStudyRoot>,
    direct_expectation: crate::ExactStructuralStudyResultExpectation,
    frequency: Box<crate::PreparedExactStructuralStudyRoot>,
    frequency_expectation: crate::ExactStructuralStudyResultExpectation,
    modal: Box<crate::PreparedExactStructuralStudyRoot>,
    modal_expectation: crate::ExactStructuralStudyResultExpectation,
    modal_transient: Box<crate::PreparedExactStructuralStudyRoot>,
    modal_transient_expectation: crate::ExactStructuralStudyResultExpectation,
    buckling: Box<crate::PreparedExactStructuralStudyRoot>,
    buckling_expectation: crate::ExactStructuralStudyResultExpectation,
    bundle: Vec<u8>,
    inventory: ResourceInventory,
    resources: ResourceRequest,
    access: ArtifactAccess,
}

async fn run_direct_exact_study_conformance(input: DirectExactStudyConformance) {
    let DirectExactStudyConformance {
        direct,
        direct_expectation,
        frequency,
        frequency_expectation,
        modal,
        modal_expectation,
        modal_transient,
        modal_transient_expectation,
        buckling,
        buckling_expectation,
        bundle,
        inventory,
        resources,
        access,
    } = input;
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"direct-exact-study-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"direct-exact-study-pool"]);
    let worker = WorkerSpec {
        host: crate::test_support::host_inventory(),
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
        for object in frequency.transport_objects().unwrap() {
            if object.is_externalized_value() {
                pool.register_value_object(object.reference, object.bytes)
                    .unwrap();
            } else {
                pool.register_execution_object(object.reference, object.bytes)
                    .unwrap();
            }
        }
        for prepared in [modal.as_ref(), modal_transient.as_ref(), buckling.as_ref()] {
            for object in prepared.transport_objects().unwrap() {
                if object.is_externalized_value() {
                    pool.register_value_object(object.reference, object.bytes)
                        .unwrap();
                } else {
                    pool.register_execution_object(object.reference, object.bytes)
                        .unwrap();
                }
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
            &direct_expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let crate::ExactStructuralStudyAnalysisResult::StructuralTransient {
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

        let response = tokio::time::timeout(
            Duration::from_secs(180),
            crate::remote::exact_structural_study::execute_for_test(
                Arc::clone(&pool),
                frequency.request(),
                &access.authorization_scope,
                &resources,
                &mut RecordingRecoveryPublisher::default(),
            ),
        )
        .await
        .expect("remote structural-frequency exact study timed out")
        .unwrap()
        .unwrap_or_else(|message| {
            panic!("remote structural-frequency exact study failed: {message}")
        });
        let imported = import_exact_structural_study_result(
            pool.as_ref(),
            response,
            &frequency_expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let crate::ExactStructuralStudyAnalysisResult::StructuralFrequency(frequency) =
            imported.analysis
        else {
            panic!("structural-frequency exact study returned another result kind")
        };
        let crate::ExactStructuralFrequencyAnalysisResult {
            resolved_workload,
            result,
            numerical_receipt,
            entries,
            histories,
            resource_usage,
            ..
        } = *frequency;
        assert_eq!(resolved_workload.step_id().as_str(), "frequency");
        assert_eq!(result.samples().len(), 2);
        assert_eq!(entries.len(), 2);
        assert_eq!(histories.len(), 1);
        let finalized = histories[0]
            .finalization
            .as_ref()
            .expect("frequency history finalization");
        let series = finalized
            .result
            .series_reduction
            .as_ref()
            .expect("frequency peak series");
        let evidence = runmat_analysis_fea::results::assessment::scalar_series_evidence(series)
            .expect("admitted scalar frequency evidence");
        assert!(evidence.value.is_finite() && evidence.value > 0.0);
        assert!(matches!(
            evidence.sample,
            Some(
                runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(
                    20.0 | 40.0
                )
            )
        ));
        numerical_receipt.validate_for_result(&result).unwrap();
        assert!(entries.iter().all(|entry| {
            !entry.recovered_fields.manifest.fields.is_empty() && entry.observations.is_some()
        }));
        assert_eq!(
            resource_usage.evidence.result_digest,
            result.manifest().identity().digest()
        );
        assert!(resource_usage.evidence.overruns.is_empty());

        let response = tokio::time::timeout(
            Duration::from_secs(180),
            crate::remote::exact_structural_study::execute_for_test(
                Arc::clone(&pool),
                modal.request(),
                &access.authorization_scope,
                &resources,
                &mut RecordingRecoveryPublisher::default(),
            ),
        )
        .await
        .expect("remote structural-modal exact study timed out")
        .unwrap()
        .unwrap_or_else(|message| panic!("remote structural-modal exact study failed: {message}"));
        let imported = import_exact_structural_study_result(
            pool.as_ref(),
            response,
            &modal_expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let crate::ExactStructuralStudyAnalysisResult::StructuralModal {
            resolved_workload,
            result,
            numerical_receipt,
        } = imported.analysis
        else {
            panic!("structural-modal exact study returned another result kind")
        };
        assert_eq!(resolved_workload.step_id().as_str(), "modes");
        assert!(!result.basis().modes().is_empty());
        numerical_receipt
            .validate_against(numerical_receipt.source_result(), &result)
            .unwrap();

        let mut recovery = RecordingRecoveryPublisher::default();
        let response = tokio::time::timeout(
            Duration::from_secs(180),
            crate::remote::exact_structural_study::execute_for_test(
                Arc::clone(&pool),
                modal_transient.request(),
                &access.authorization_scope,
                &resources,
                &mut recovery,
            ),
        )
        .await
        .expect("remote modal-transient exact study timed out")
        .unwrap()
        .unwrap_or_else(|message| panic!("remote modal-transient exact study failed: {message}"));
        let imported = import_exact_structural_study_result(
            pool.as_ref(),
            response,
            &modal_transient_expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let crate::ExactStructuralStudyAnalysisResult::StructuralModalTransient(result) =
            imported.analysis
        else {
            panic!("modal-transient exact study returned another result kind")
        };
        assert_eq!(result.modal_workload.step_id().as_str(), "modes");
        assert_eq!(result.transient_workload.step_id.as_str(), "response");
        assert_eq!(
            result.basis_state.source_result,
            result.modal_numerical_receipt.source_result()
        );
        result
            .transient_numerical_receipt
            .validate_for_result(&result.transient_result)
            .unwrap();
        assert_eq!(recovery.closures.len(), 2);

        let response = tokio::time::timeout(
            Duration::from_secs(180),
            crate::remote::exact_structural_study::execute_for_test(
                Arc::clone(&pool),
                buckling.request(),
                &access.authorization_scope,
                &resources,
                &mut RecordingRecoveryPublisher::default(),
            ),
        )
        .await
        .expect("remote structural-buckling exact study timed out")
        .unwrap()
        .unwrap_or_else(|message| panic!("remote structural-buckling study failed: {message}"));
        let imported = import_exact_structural_study_result(
            pool.as_ref(),
            response,
            &buckling_expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let crate::ExactStructuralStudyAnalysisResult::StructuralBuckling(buckling_result) =
            &imported.analysis
        else {
            panic!("structural-buckling exact study returned another result kind")
        };
        assert_eq!(buckling_result.preload_workload.step_id.as_str(), "preload");
        assert_eq!(
            buckling_result.buckling_workload.step_id().as_str(),
            "buckling"
        );
        assert_eq!(
            buckling_result.prestress_state.source_result,
            buckling_result.preload_result_identity
        );
        buckling_result
            .preload_numerical_receipt
            .validate_against(
                buckling_result.preload_result_identity,
                &buckling_result.preload_result,
            )
            .unwrap();
        buckling_result
            .buckling_numerical_receipt
            .validate_against(
                buckling_result.buckling_numerical_receipt.source_result(),
                &buckling_result.buckling_result,
            )
            .unwrap();
        let local = runmat_analysis_fea::execute_structural_buckling_workload(
            &buckling_result.buckling_workload,
            &imported.mesh.artifact,
            &buckling_result.prestress_state,
        )
        .unwrap();
        assert_eq!(local, *buckling_result.buckling_result);

        let fixed_face = &box_selectors().fixed_face;
        let loaded_face = &box_selectors().loaded_face;
        let fixed_centroid = boundary_centroid(&imported.mesh.artifact, fixed_face);
        let loaded_centroid = boundary_centroid(&imported.mesh.artifact, loaded_face);
        assert!((fixed_centroid[0] - 0.0).abs() <= 1.0e-12);
        assert!((loaded_centroid[0] - 1.0).abs() <= 1.0e-12);
        assert!((fixed_centroid[1] - loaded_centroid[1]).abs() <= 1.0e-12);
        assert!((fixed_centroid[2] - loaded_centroid[2]).abs() <= 1.0e-12);
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
        invocation: runmat_analysis_core::GeneratorInvocation {
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
            effects: GeneratorEffectPolicy::DeclaredImmutableArtifacts,
        },
        domain: GeneratorDomain::TimeSeconds,
        sampling: SignalSamplingPolicy::Uniform {
            start: 0.0,
            end: 1.0,
            sample_count: 5,
        },
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
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
        .await
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
        .function_analysis(
            executable
                .manifest
                .identity
                .entrypoint_function
                .expect("pulse entrypoint function"),
        )
        .cloned()
        .expect("pulse static facts");
    let admitted = admit_named_signal_generator(
        &authored,
        revision,
        BundleCallable {
            owner_identity,
            qualified_name: "loads.pulse".into(),
            source_digest: Digest::sha256(source_text),
        },
        runmat_analysis_execution::named_callable::CompiledCallableBinding::from_selected_identity(
            &executable.manifest.identity,
            "loads.pulse",
        )
        .unwrap(),
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
        None,
    )
    .unwrap();
    (workload, prepared)
}

/// Canonical persistent identities of the box fixture's source solid and of its first two
/// source faces (x = 0 and x = 1).
pub(crate) struct BoxSelectors {
    pub(crate) solid: runmat_geometry_core::PersistentEntityId,
    pub(crate) fixed_face: runmat_geometry_core::PersistentEntityId,
    pub(crate) loaded_face: runmat_geometry_core::PersistentEntityId,
}

impl BoxSelectors {
    pub(crate) fn solid_selector(&self) -> String {
        self.solid.analysis_selection_id().unwrap()
    }

    pub(crate) fn fixed_face_selector(&self) -> String {
        self.fixed_face.analysis_selection_id().unwrap()
    }

    pub(crate) fn loaded_face_selector(&self) -> String {
        self.loaded_face.analysis_selection_id().unwrap()
    }
}

pub(crate) fn box_selectors() -> &'static BoxSelectors {
    static SELECTORS: std::sync::OnceLock<BoxSelectors> = std::sync::OnceLock::new();
    SELECTORS.get_or_init(|| {
        let imported = import_exact_cad(
            "box.brep",
            BOX,
            GeometryFormat::Brep,
            &ExactCadImportOptions {
                source_units: Some(runmat_geometry_core::LengthUnit::Meter),
                ..ExactCadImportOptions::default()
            },
            &GeometryImportContext::new(),
        )
        .unwrap();
        BoxSelectors {
            solid: imported.topology.regions[0].id.clone(),
            fixed_face: imported.source_face_ids[&0].clone(),
            loaded_face: imported.source_face_ids[&1].clone(),
        }
    })
}

fn box_structural_workload() -> StructuralLinearWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_id = "box".into();
    model.geometry_revision = 1;
    model.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::solid(box_selectors().solid_selector());
    model.constraints[0].visit_targets_mut(|target| {
        *target =
            runmat_analysis_core::RegionTarget::boundary(box_selectors().fixed_face_selector());
    });
    if let runmat_analysis_core::ExcitationTarget::Region(target) =
        &mut model.excitations[0].excitation_mut().target
    {
        *target =
            runmat_analysis_core::RegionTarget::boundary(box_selectors().loaded_face_selector());
    } else {
        panic!("cantilever fixture must use a region excitation");
    }
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
        require_solver_mesh_for_solid: true,
        state_input: None,
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
        schema_version: 2,
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        solver: runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
        residual_target: 1.0e-8,
    }
}

fn box_structural_frequency_workload() -> StructuralFrequencyWorkload {
    use runmat_analysis_core::*;
    let mut model = box_structural_workload().model;
    let mut step = runmat_analysis_core::ProcedureStep::structural_harmonic("frequency");
    step.independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    step.observations = vec![ObservationRequest::History(ObservationSpec {
        observation_id: ObservationId("frequency_peak".into()),
        quantity: ObservationQuantity::Displacement,
        value_shape: ObservationValueShape::Vector3,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Node,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::Magnitude,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Magnitude,
        sampling: SamplingPolicy::FrequencyHertz(vec![20.0, 40.0]),
        reduction: Some(ReductionPolicy {
            spatial: SpatialReduction::Maximum,
            series: SeriesReduction::Maximum,
            extrema_ties: ExtremaTiePolicy::FirstCanonical,
        }),
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Real64,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::Automatic,
    })];
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "box-structural-frequency-sequence",
        vec![step.clone()],
    );
    StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: step.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct {
            policy: runmat_analysis_fea::solve::dynamics::DirectHarmonicPolicy::default(),
        },
        prestress_transition: None,
    })
    .unwrap()
}

fn box_structural_modal_workload() -> StructuralModalWorkload {
    let mut model = box_structural_workload().model;
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "box-structural-modal-sequence",
        vec![step.clone()],
    );
    StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model,
        step_id: step.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap()
}

fn box_modal_transient_workloads() -> runmat_analysis_execution::StructuralStudyWorkloads {
    let mut model = box_structural_workload().model;
    let mut modal_step = ProcedureStep::structural_modal("modes");
    modal_step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    let mut transient_step =
        ProcedureStep::structural_modal_transient("response", "modes", "modal_basis");
    transient_step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
        interval_count: 10,
    };
    model.sequence.steps = vec![modal_step, transient_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("response".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-transient".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("response".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
    }];
    let modal = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: ProcedureStepId("modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let transient = StructuralModalTransientTemplate {
        schema_version: 2,
        initial_fields: Vec::new(),
        model,
        step_id: ProcedureStepId("response".into()),
        backend: ComputeBackend::Cpu,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
        residual_target: 1.0e-8,
    };
    runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: std::collections::BTreeMap::from([
            (
                ProcedureStepId("modes".into()),
                StructuralStepWorkload::ModalExtraction {
                    workload: Box::new(modal),
                },
            ),
            (
                ProcedureStepId("response".into()),
                StructuralStepWorkload::ModalTransient {
                    template: Box::new(transient),
                    block_plan: StructuralTransientBlockPlan::new(1).unwrap(),
                },
            ),
        ]),
    }
}

pub(crate) fn box_modal_response_spectrum_workloads(
) -> runmat_analysis_execution::StructuralStudyWorkloads {
    // The refined box's first twelve modes capture only 0.767492 of the
    // excitation mass. Keep the spectrum's 0.9 admission requirement and
    // provide a larger basis for this positive end-to-end fixture.
    const REQUESTED_MODES: usize = 48;
    let mut model = box_structural_workload().model;
    let mut modal_step = ProcedureStep::structural_modal("modes");
    modal_step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: REQUESTED_MODES,
    };
    let spectrum = ResponseSpectrum {
        spectrum_id: ResponseSpectrumId("box-global-x".into()),
        abscissa: SpectrumAbscissa::FrequencyHertz(runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Frequency,
            _,
        >::new(vec![1.0e-3, 1.0e8])),
        ordinates: runmat_analysis_core::NumericalDataSource::authored(
            vec![1.0e-3, 1.0e-3],
            "sha256:box-global-x",
        ),
        damping: SpectrumDamping::Ratio(0.05),
        response: SpectrumResponse::RelativeDisplacement,
        envelope: SpectrumEnvelope::Maximax,
        interpolation: SpectrumInterpolation::LinearFrequency,
        extrapolation: SpectrumExtrapolation::HoldEndpoint,
        direction: SpectrumDirection {
            frame: runmat_analysis_core::CoordinateFrame::Global,
            vector: [1.0, 0.0, 0.0],
        },
        modal_combination: ModalCombination::AbsoluteSum,
        directional_combination: DirectionalCombination::Srss,
        truncation: SpectrumTruncationPolicy {
            minimum_cumulative_mass_ratio: 0.9,
            correction: ModalTruncationCorrection::None,
        },
        rigid_response: RigidResponsePolicy::Disabled,
        construction: SpectrumConstruction::AuthoredDesign,
        evidence: SpectrumEvidence {
            sampling_adequacy_ratio: 1.0,
            max_interpolation_error: None,
        },
    };
    let response_step = ProcedureStep::structural_modal_response_spectrum(
        "spectrum",
        vec![spectrum.spectrum_id.clone()],
        "modes",
        "modal_basis",
    );
    let mut sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "box-modal-response-spectrum-sequence",
        vec![modal_step.clone(), response_step.clone()],
    );
    sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    sequence.transitions.push(StateTransition {
        transition_id: TransitionId("modes-to-spectrum".into()),
        producer_step: modal_step.step_id.clone(),
        consumer_step: response_step.step_id.clone(),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    model.sequence = sequence;
    model.response_spectra = vec![spectrum];
    let modal = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: modal_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(REQUESTED_MODES),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let response = StructuralModalResponseSpectrumTemplate {
        schema_version: 1,
        model,
        step_id: response_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
    };
    runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: std::collections::BTreeMap::from([
            (
                modal_step.step_id,
                StructuralStepWorkload::ModalExtraction {
                    workload: Box::new(modal),
                },
            ),
            (
                response_step.step_id,
                StructuralStepWorkload::ModalResponseSpectrum {
                    template: Box::new(response),
                },
            ),
        ]),
    }
}

pub(crate) fn box_random_response_workloads() -> runmat_analysis_execution::StructuralStudyWorkloads
{
    let mut model = box_structural_workload().model;
    let mut modal_step = ProcedureStep::structural_modal("modes");
    modal_step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    let excitation = runmat_analysis_core::RandomExcitationSpectrum {
        excitation_id: runmat_analysis_core::RandomExcitationId("box-broadband-x".into()),
        frequencies_hz: runmat_types::physical::Quantity::new(vec![
            1.0e-3, 100.0, 1_000.0, 10_000.0,
        ]),
        channels: vec![runmat_analysis_core::RandomExcitationChannel {
            channel_id: runmat_analysis_core::RandomExcitationChannelId("base-x".into()),
            quantity: runmat_analysis_core::RandomExcitationQuantity::BaseAcceleration,
            direction: SpectrumDirection {
                frame: runmat_analysis_core::CoordinateFrame::Global,
                vector: [1.0, 0.0, 0.0],
            },
        }],
        spectral_density: runmat_analysis_core::NumericalDataSource::authored(
            vec![
                runmat_analysis_core::CrossPowerSpectralDensity::diagonal(
                    &runmat_types::physical::Quantity::new([2.5e-4])
                );
                4
            ],
            "sha256:box-broadband-x",
        ),
        convention: runmat_analysis_core::SpectralDensityConvention::OneSidedPerHertz,
        interpolation: runmat_analysis_core::RandomSpectrumInterpolation::LinearFrequency,
        extrapolation: runmat_analysis_core::RandomSpectrumExtrapolation::Reject,
        evidence: runmat_analysis_core::RandomSpectrumEvidence {
            max_absolute_density_error: Some(
                runmat_types::physical::AccelerationPowerSpectralDensity::new(0.0),
            ),
            equivalent_degrees_of_freedom: None,
        },
    };
    let response_step = ProcedureStep::structural_random_response(
        "random",
        excitation.excitation_id.clone(),
        runmat_analysis_core::StructuralRandomResponseSettings {
            modal_damping_ratio: 0.05,
            minimum_cumulative_mass_ratio: 0.9,
        },
        "modes",
        "modal_basis",
    );
    let mut sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "box-random-response-sequence",
        vec![modal_step.clone(), response_step.clone()],
    );
    sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    sequence.transitions.push(StateTransition {
        transition_id: TransitionId("modes-to-random".into()),
        producer_step: modal_step.step_id.clone(),
        consumer_step: response_step.step_id.clone(),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    model.sequence = sequence;
    model.random_excitations = vec![excitation];
    let modal = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: modal_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let response = runmat_analysis_fea::execution::StructuralRandomResponseTemplate {
        schema_version: 1,
        model,
        step_id: response_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        integration_policy: runmat_analysis_fea::solve::dynamics::RandomResponseIntegrationPolicy {
            modal_damping_ratio: 0.04,
            minimum_cumulative_mass_ratio: 0.0,
            relative_variance_tolerance: 1.0e-5,
            maximum_refinement_levels: 14,
        },
        maximum_intervals_per_task: 2,
    };
    runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: std::collections::BTreeMap::from([
            (
                modal_step.step_id,
                StructuralStepWorkload::ModalExtraction {
                    workload: Box::new(modal),
                },
            ),
            (
                response_step.step_id,
                StructuralStepWorkload::RandomResponse {
                    template: Box::new(response),
                },
            ),
        ]),
    }
}

fn box_buckling_workloads() -> runmat_analysis_execution::StructuralStudyWorkloads {
    let mut preload_workload = box_structural_workload();
    let runmat_analysis_core::ExcitationTarget::Region(target) =
        &mut preload_workload.model.excitations[0]
            .excitation_mut()
            .target
    else {
        panic!("box preload fixture must use a region excitation")
    };
    *target = runmat_analysis_core::RegionTarget::boundary(box_selectors().loaded_face_selector());
    let runmat_analysis_core::PhysicalSignal::Force(signal) =
        &mut preload_workload.model.signals[0]
    else {
        panic!("box preload fixture must carry a force signal")
    };
    signal.representation = runmat_analysis_core::SignalRepresentation::Constant(
        runmat_analysis_core::SignalValue::Vector3([
            runmat_analysis_core::Force::new(-1_000.0),
            runmat_analysis_core::Force::new(0.0),
            runmat_analysis_core::Force::new(0.0),
        ]),
    );
    let preload = ProcedureStep::structural_preload("preload");
    let mut buckling =
        ProcedureStep::structural_buckling("buckling", "preload", "geometric_stiffness");
    buckling.independent_domain = IndependentDomain::Eigenvalue { requested_count: 1 };
    preload_workload.model.sequence.steps = vec![preload, buckling];
    preload_workload.model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("preload".into()),
        successor: ProcedureStepId("buckling".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    preload_workload.model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("preload-to-buckling".into()),
        producer_step: ProcedureStepId("preload".into()),
        consumer_step: ProcedureStepId("buckling".into()),
        state_id: StateId("geometric_stiffness".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
    }];
    preload_workload.step_id = ProcedureStepId("preload".into());
    let template = StructuralBucklingTemplate {
        schema_version: 1,
        model: preload_workload.model.clone(),
        step_id: ProcedureStepId("buckling".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        solver: StructuralBucklingSolverPolicy::new(1, 1.0e-8, 256).unwrap(),
    };
    runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: std::collections::BTreeMap::from([
            (
                ProcedureStepId("preload".into()),
                StructuralStepWorkload::Linear {
                    workload: Box::new(preload_workload),
                },
            ),
            (
                ProcedureStepId("buckling".into()),
                StructuralStepWorkload::Buckling {
                    template: Box::new(template),
                },
            ),
        ]),
    }
}

fn boundary_centroid(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    source_face: &runmat_geometry_core::PersistentEntityId,
) -> [f64; 3] {
    let node_ids = mesh
        .topology
        .boundary_faces
        .iter()
        .filter(|face| face.provenance.iter().any(|source| source == source_face))
        .flat_map(|face| face.node_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    assert!(!node_ids.is_empty(), "source face must own solver nodes");
    let mut centroid = [0.0; 3];
    for node_id in &node_ids {
        let coordinates = mesh
            .topology
            .nodes
            .iter()
            .find(|node| node.node_id == *node_id)
            .expect("boundary node must exist")
            .coordinates_m
            .canonical_value();
        for axis in 0..3 {
            centroid[axis] += coordinates[axis];
        }
    }
    for coordinate in &mut centroid {
        *coordinate /= node_ids.len() as f64;
    }
    centroid
}

fn coarse_box_request() -> MeshingRequest {
    let mut request = crate::resolve_native_meshing_request(
        &ExactCadImportOptions {
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            ..ExactCadImportOptions::default()
        },
        runmat_meshing_core::MeshingRequestSettings::default(),
    )
    .unwrap();
    request.metric.global_metric = MetricTensor3::isotropic_length_m(10.0).unwrap();
    request.quality.curve.maximum_chordal_deviation_m = runmat_types::physical::Length::new(0.1);
    request.quality.curve.maximum_tangent_change_degrees = 180.0;
    request.quality.curve.minimum_metric_edge_length = 0.01;
    request.quality.curve.maximum_metric_edge_length = 10.0;
    request.quality.surface.minimum_metric_angle_degrees = 0.1;
    request.quality.surface.maximum_physical_aspect_ratio = 1_000.0;
    request.quality.surface.maximum_chordal_deviation_m = runmat_types::physical::Length::new(0.1);
    request.quality.surface.maximum_normal_deviation_degrees = 180.0;
    request.quality.volume.maximum_metric_edge_length = 2.0;
    request.quality.volume.maximum_radius_edge_ratio = 10.0;
    request.quality.volume.minimum_scaled_jacobian = 0.01;
    request.limits.maximum_nodes = 10_000;
    request.limits.maximum_elements = 10_000;
    request
}
