use super::*;

#[tokio::test(flavor = "current_thread")]
async fn prestressed_modal_reaches_an_admitted_result_over_encrypted_quic() {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"prestressed-modal-conformance\"\n[sources]\nroots = [\"src\"]\n",
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
        authorization_scope: "prestressed-modal-conformance-run".into(),
        encryption_context: Digest::sha256(b"prestressed-modal-conformance-encryption"),
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
            authored_study_digest: Digest::sha256(b"box structural prestressed modal study"),
            frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
            geometry_source_digest: Digest::sha256(BOX),
            workloads: box_prestressed_modal_workloads(),
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
    Box::pin(run_prestressed_exact_study_conformance(
        prepared,
        expectation,
        bundle,
        inventory,
        resources,
        access,
    ))
    .await;
}

async fn verify_prestressed_modal_exact_study(
    pool: &Arc<RemotePoolDriver>,
    prepared: &crate::PreparedExactStructuralStudyRoot,
    expectation: &crate::ExactStructuralStudyResultExpectation,
    access: &ArtifactAccess,
    resources: &ResourceRequest,
) {
    let mut recovery = RecordingRecoveryPublisher::default();
    let response = tokio::time::timeout(
        Duration::from_secs(180),
        crate::remote::exact_structural_study::execute_for_test(
            Arc::clone(pool),
            prepared.request(),
            &access.authorization_scope,
            resources,
            &mut recovery,
        ),
    )
    .await
    .expect("remote prestressed-modal exact study timed out")
    .unwrap()
    .unwrap_or_else(|message| panic!("remote prestressed-modal study failed: {message}"));
    let imported = import_exact_structural_study_result(
        pool.as_ref(),
        response,
        expectation,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let crate::ExactStructuralStudyAnalysisResult::StructuralPrestressedModal(modal_result) =
        &imported.analysis
    else {
        panic!("prestressed-modal exact study returned another result kind")
    };
    assert_eq!(modal_result.preload_workload.step_id.as_str(), "preload");
    assert_eq!(
        modal_result.modal_workload.step_id().as_str(),
        "prestressed_modes"
    );
    assert_eq!(
        modal_result.prestress_state.source_result,
        modal_result.preload_result_identity
    );
    assert!(!modal_result.modal_result.basis().modes().is_empty());
    modal_result
        .modal_numerical_receipt
        .validate_against(
            modal_result.modal_numerical_receipt.source_result(),
            &modal_result.modal_result,
        )
        .unwrap();
    runmat_analysis_fea::recover_completed_structural_prestressed_modal_run(
        &modal_result.modal_workload,
        &imported.mesh.artifact,
        &modal_result.prestress_state,
        &modal_result.modal_result,
        &modal_result.modal_numerical_receipt,
    )
    .unwrap();
}

async fn run_prestressed_exact_study_conformance(
    prepared: Box<crate::PreparedExactStructuralStudyRoot>,
    expectation: crate::ExactStructuralStudyResultExpectation,
    bundle: Vec<u8>,
    inventory: ResourceInventory,
    resources: ResourceRequest,
    access: ArtifactAccess,
) {
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"prestressed-modal-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"prestressed-modal-pool"]);
    let worker = WorkerSpec {
        host: crate::test_support::host_inventory(),
        id: runmat_execution::identity::WorkerId::derive(&[b"prestressed-modal-worker"]),
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
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([83; 32]).unwrap();
    let server = run_remote_meshing_worker_quic(RemoteMeshingWorkerQuicRequest {
        worker: RemoteWorkerQuicRequest {
            listener,
            run_identity: access.authorization_scope.clone(),
            worker: worker.clone(),
            driver_fence: 79,
            session_id: [89; 16],
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
                node_identity: "prestressed-modal-node".into(),
                worker: worker.clone(),
                driver_fence: 79,
                session_id: [89; 16],
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
            79,
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
        Box::pin(verify_prestressed_modal_exact_study(
            &pool,
            &prepared,
            &expectation,
            &access,
            &resources,
        ))
        .await;
        pool.remove_worker(worker.id, false).await.unwrap();
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.unwrap();
}

fn box_prestressed_modal_workloads() -> runmat_analysis_execution::StructuralStudyWorkloads {
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
    let mut modal = ProcedureStep::structural_prestressed_modal(
        "prestressed_modes",
        "preload",
        "geometric_stiffness",
    );
    modal.independent_domain = IndependentDomain::Eigenvalue { requested_count: 1 };
    preload.model.sequence.steps = vec![preload_step, modal];
    preload.model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("preload".into()),
        successor: ProcedureStepId("prestressed_modes".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    preload.model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("preload-to-prestressed-modal".into()),
        producer_step: ProcedureStepId("preload".into()),
        consumer_step: ProcedureStepId("prestressed_modes".into()),
        state_id: StateId("geometric_stiffness".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
    }];
    let template = StructuralPrestressedModalTemplate {
        schema_version: 1,
        model: preload.model.clone(),
        step_id: ProcedureStepId("prestressed_modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(1),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
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
                    template: Box::new(template),
                },
            ),
        ]),
    }
}
