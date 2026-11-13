use super::*;

#[tokio::test(flavor = "current_thread")]
async fn prestressed_frequency_reaches_an_admitted_result_over_encrypted_quic() {
    Box::pin(run_prestressed_frequency_conformance(false)).await;
}

#[tokio::test(flavor = "current_thread")]
async fn prestressed_modal_frequency_reaches_an_admitted_result_over_encrypted_quic() {
    Box::pin(run_prestressed_frequency_conformance(true)).await;
}

async fn run_prestressed_frequency_conformance(modal_reduced: bool) {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"prestressed-frequency-conformance\"\n[sources]\nroots = [\"src\"]\n",
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
        authorization_scope: "prestressed-frequency-conformance-run".into(),
        encryption_context: Digest::sha256(b"prestressed-frequency-conformance-encryption"),
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
    let prepared = Box::new(
        prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
            program_revision: revision.clone(),
            artifact_access: access.clone(),
            authored_study_digest: Digest::sha256(b"box structural prestressed frequency study"),
            frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
            geometry_source_digest: Digest::sha256(BOX),
            workloads: if modal_reduced {
                box_prestressed_modal_frequency_workloads()
            } else {
                box_prestressed_frequency_workloads()
            },
            analysis_accelerator_requests: Vec::new(),
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            input_materializations: Vec::new(),
            exact_meshing,
        })
        .unwrap(),
    );
    let expectation = prepared.result_expectation();
    let mut builder = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_project_runtime_policy_object(crate::remote::runtime_settings::fixture_policy(
            &revision,
        ))
        .unwrap()
        .with_compiled_package_closure();
    for program in prepared.required_programs().unwrap() {
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

    let imported_geometry = import_exact_cad(
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
    let document = imported_geometry.geometry_document().unwrap();
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
    // Keep nested transport/execution state on the heap rather than carrying every
    // async frame through the default test-thread stack during synchronous import.
    let imported = Box::pin(run_exact_study(
        prepared,
        expectation,
        bundle,
        inventory,
        resources,
        access,
    ))
    .await;
    match imported.analysis {
        crate::ExactStructuralStudyAnalysisResult::StructuralPrestressedFrequency(result)
            if !modal_reduced =>
        {
            assert_eq!(result.preload_workload.step_id.as_str(), "preload");
            assert_eq!(
                result.frequency.resolved_workload.step_id().as_str(),
                "prestressed_frequency"
            );
            assert_eq!(
                result.prestress_state.source_result,
                result.preload_result_identity
            );
            assert_eq!(result.frequency.result.samples().len(), 2);
            assert_eq!(result.frequency.entries.len(), 2);
            assert_eq!(
                result.frequency.object_catalog.roots.len(),
                3 + result.frequency.entries.len()
            );
        }
        crate::ExactStructuralStudyAnalysisResult::StructuralPrestressedModalFrequency(result)
            if modal_reduced =>
        {
            assert_eq!(result.preload_workload.step_id.as_str(), "preload");
            assert_eq!(
                result.modal_workload.step_id().as_str(),
                "prestressed_modes"
            );
            assert_eq!(
                result.frequency.resolved_workload.step_id().as_str(),
                "prestressed_frequency"
            );
            assert_eq!(
                result.prestress_state.source_result,
                result.preload_result_identity
            );
            assert_eq!(
                result.basis_state.source_result,
                result.modal_numerical_receipt.source_result()
            );
            assert_eq!(result.frequency.result.samples().len(), 2);
            assert_eq!(result.frequency.entries.len(), 2);
        }
        _ => panic!("prestressed-frequency exact study returned another result kind"),
    }
}

async fn run_exact_study(
    prepared: Box<crate::PreparedExactStructuralStudyRoot>,
    expectation: crate::ExactStructuralStudyResultExpectation,
    bundle: Vec<u8>,
    inventory: ResourceInventory,
    resources: ResourceRequest,
    access: ArtifactAccess,
) -> crate::ExactStructuralStudyResult {
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"prestressed-frequency-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"prestressed-frequency-pool"]);
    let worker = WorkerSpec {
        host: crate::test_support::host_inventory(),
        id: runmat_execution::identity::WorkerId::derive(&[b"prestressed-frequency-worker"]),
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
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([93; 32]).unwrap();
    let server = run_remote_meshing_worker_quic(RemoteMeshingWorkerQuicRequest {
        worker: RemoteWorkerQuicRequest {
            listener,
            run_identity: access.authorization_scope.clone(),
            worker: worker.clone(),
            driver_fence: 91,
            session_id: [97; 16],
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
                node_identity: "prestressed-frequency-node".into(),
                worker: worker.clone(),
                driver_fence: 91,
                session_id: [97; 16],
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
            91,
            bundle,
            access.authorization_scope.clone(),
        )
        .unwrap();
        pool.add_worker(channel).await.unwrap();
        for object in prepared.transport_objects().unwrap() {
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
            Box::pin(crate::remote::exact_structural_study::execute_for_test(
                Arc::clone(&pool),
                prepared.request(),
                &access.authorization_scope,
                &resources,
                &mut recovery,
            )),
        )
        .await
        .expect("remote prestressed-frequency exact study timed out")
        .unwrap()
        .unwrap_or_else(|message| panic!("remote prestressed-frequency study failed: {message}"));
        let imported = import_exact_structural_study_result(
            pool.as_ref(),
            response,
            &expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        pool.remove_worker(worker.id, false).await.unwrap();
        imported
    };
    let (server_result, imported) = tokio::join!(server, client);
    server_result.unwrap();
    imported
}

fn box_prestressed_frequency_workloads() -> runmat_analysis_execution::StructuralStudyWorkloads {
    let runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        mut workloads,
    } = box_buckling_workloads()
    else {
        unreachable!()
    };
    let StructuralStepWorkload::Linear {
        workload: mut preload,
    } = workloads
        .remove(&ProcedureStepId("preload".into()))
        .unwrap()
    else {
        unreachable!()
    };
    let preload_step = ProcedureStep::structural_preload("preload");
    let mut frequency = ProcedureStep::structural_prestressed_harmonic(
        "prestressed_frequency",
        "preload",
        "geometric_stiffness",
    );
    frequency.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    preload.model.sequence.steps = vec![preload_step, frequency];
    preload.model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("preload".into()),
        successor: ProcedureStepId("prestressed_frequency".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    preload.model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("preload-to-prestressed-frequency".into()),
        producer_step: ProcedureStepId("preload".into()),
        consumer_step: ProcedureStepId("prestressed_frequency".into()),
        state_id: StateId("geometric_stiffness".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
    }];
    preload.step_id = ProcedureStepId("preload".into());
    let template = StructuralPrestressedFrequencyTemplate {
        schema_version: 1,
        model: preload.model.clone(),
        step_id: ProcedureStepId("prestressed_frequency".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralPrestressedFrequencySolverTemplate::Direct {
            policy: runmat_analysis_fea::solve::dynamics::DirectHarmonicPolicy::default(),
        },
    };
    runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: std::collections::BTreeMap::from([
            (
                ProcedureStepId("preload".into()),
                StructuralStepWorkload::Linear { workload: preload },
            ),
            (
                ProcedureStepId("prestressed_frequency".into()),
                StructuralStepWorkload::PrestressedFrequency {
                    template: Box::new(template),
                },
            ),
        ]),
    }
}

fn box_prestressed_modal_frequency_workloads() -> runmat_analysis_execution::StructuralStudyWorkloads
{
    let runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        mut workloads,
    } = box_prestressed_frequency_workloads()
    else {
        unreachable!()
    };
    let StructuralStepWorkload::Linear {
        workload: mut preload,
    } = workloads
        .remove(&ProcedureStepId("preload".into()))
        .unwrap()
    else {
        unreachable!()
    };
    let preload_step = ProcedureStep::structural_preload("preload");
    let mut modal = ProcedureStep::structural_prestressed_modal(
        "prestressed_modes",
        "preload",
        "geometric_stiffness",
    );
    modal.independent_domain = IndependentDomain::Eigenvalue { requested_count: 2 };
    let mut frequency = ProcedureStep::structural_prestressed_harmonic(
        "prestressed_frequency",
        "preload",
        "geometric_stiffness",
    );
    frequency.solution_space = runmat_analysis_core::SolutionSpace::ModalReduced;
    frequency.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    frequency
        .state_inputs
        .push(runmat_analysis_core::StateInput {
            input_id: "modal_basis".into(),
            required_kind: runmat_analysis_core::StateKind::ModalBasis,
            source: runmat_analysis_core::StateSource::Upstream {
                producer_step: ProcedureStepId("prestressed_modes".into()),
                state_id: StateId("modal_basis".into()),
            },
            use_as: runmat_analysis_core::StateUse::ModalBasis,
            conflict: runmat_analysis_core::InitialStateConflictPolicy::Reject,
        });
    preload.model.sequence.steps = vec![preload_step, modal, frequency];
    preload.model.sequence.dependencies = vec![
        SequenceDependency {
            predecessor: ProcedureStepId("preload".into()),
            successor: ProcedureStepId("prestressed_modes".into()),
            reason: SequenceDependencyReason::StateTransfer,
        },
        SequenceDependency {
            predecessor: ProcedureStepId("preload".into()),
            successor: ProcedureStepId("prestressed_frequency".into()),
            reason: SequenceDependencyReason::StateTransfer,
        },
        SequenceDependency {
            predecessor: ProcedureStepId("prestressed_modes".into()),
            successor: ProcedureStepId("prestressed_frequency".into()),
            reason: SequenceDependencyReason::StateTransfer,
        },
    ];
    preload.model.sequence.transitions = vec![
        StateTransition {
            transition_id: TransitionId("preload-to-prestressed-modal".into()),
            producer_step: ProcedureStepId("preload".into()),
            consumer_step: ProcedureStepId("prestressed_modes".into()),
            state_id: StateId("geometric_stiffness".into()),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        },
        StateTransition {
            transition_id: TransitionId("preload-to-prestressed-frequency".into()),
            producer_step: ProcedureStepId("preload".into()),
            consumer_step: ProcedureStepId("prestressed_frequency".into()),
            state_id: StateId("geometric_stiffness".into()),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        },
        StateTransition {
            transition_id: TransitionId("prestressed-modal-to-frequency".into()),
            producer_step: ProcedureStepId("prestressed_modes".into()),
            consumer_step: ProcedureStepId("prestressed_frequency".into()),
            state_id: StateId("modal_basis".into()),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        },
    ];
    preload.step_id = ProcedureStepId("preload".into());
    let modal_template = StructuralPrestressedModalTemplate {
        schema_version: 1,
        model: preload.model.clone(),
        step_id: ProcedureStepId("prestressed_modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(2),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
    };
    let frequency_template = StructuralPrestressedFrequencyTemplate {
        schema_version: 1,
        model: preload.model.clone(),
        step_id: ProcedureStepId("prestressed_frequency".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralPrestressedFrequencySolverTemplate::ModalReduced,
    };
    runmat_analysis_execution::StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: std::collections::BTreeMap::from([
            (
                ProcedureStepId("preload".into()),
                StructuralStepWorkload::Linear { workload: preload },
            ),
            (
                ProcedureStepId("prestressed_modes".into()),
                StructuralStepWorkload::PrestressedModal {
                    template: Box::new(modal_template),
                },
            ),
            (
                ProcedureStepId("prestressed_frequency".into()),
                StructuralStepWorkload::PrestressedFrequency {
                    template: Box::new(frequency_template),
                },
            ),
        ]),
    }
}
