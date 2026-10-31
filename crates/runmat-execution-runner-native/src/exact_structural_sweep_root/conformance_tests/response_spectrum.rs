use super::*;

#[tokio::test(flavor = "current_thread")]
async fn response_spectrum_fields_reach_an_admitted_result_over_encrypted_quic() {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"response-spectrum-conformance\"\n[sources]\nroots = [\"src\"]\n",
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
        authorization_scope: "response-spectrum-conformance-run".into(),
        encryption_context: Digest::sha256(b"response-spectrum-conformance-encryption"),
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
            authored_study_digest: Digest::sha256(b"box modal response-spectrum study"),
            frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
            geometry_source_digest: Digest::sha256(BOX),
            workloads: box_modal_response_spectrum_workloads(),
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
    Box::pin(run_response_spectrum_conformance(
        prepared,
        expectation,
        bundle,
        inventory,
        resources,
        access,
        ModalConsumerExpectation::ResponseSpectrum,
    ))
    .await;
}

#[derive(Clone, Copy)]
pub(super) enum ModalConsumerExpectation {
    ResponseSpectrum,
    RandomResponse,
}

pub(super) async fn run_response_spectrum_conformance(
    prepared: Box<crate::PreparedExactStructuralStudyRoot>,
    expectation: crate::ExactStructuralStudyResultExpectation,
    bundle: Vec<u8>,
    inventory: ResourceInventory,
    resources: ResourceRequest,
    access: ArtifactAccess,
    expected: ModalConsumerExpectation,
) {
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"response-spectrum-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"response-spectrum-pool"]);
    let worker = WorkerSpec {
        host: crate::test_support::host_inventory(),
        id: runmat_execution::identity::WorkerId::derive(&[b"response-spectrum-worker"]),
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
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([97; 32]).unwrap();
    let server = run_remote_meshing_worker_quic(RemoteMeshingWorkerQuicRequest {
        worker: RemoteWorkerQuicRequest {
            listener,
            run_identity: access.authorization_scope.clone(),
            worker: worker.clone(),
            driver_fence: 101,
            session_id: [103; 16],
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
                node_identity: "response-spectrum-node".into(),
                worker: worker.clone(),
                driver_fence: 101,
                session_id: [103; 16],
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
            101,
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
        Box::pin(verify_response_spectrum_result(
            &pool,
            &prepared,
            &expectation,
            &access,
            &resources,
            expected,
        ))
        .await;
        pool.remove_worker(worker.id, false).await.unwrap();
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.unwrap();
}

async fn verify_response_spectrum_result(
    pool: &Arc<RemotePoolDriver>,
    prepared: &crate::PreparedExactStructuralStudyRoot,
    expectation: &crate::ExactStructuralStudyResultExpectation,
    access: &ArtifactAccess,
    resources: &ResourceRequest,
    expected: ModalConsumerExpectation,
) {
    let response = tokio::time::timeout(
        Duration::from_secs(180),
        crate::remote::exact_structural_study::execute_for_test(
            Arc::clone(pool),
            prepared.request(),
            &access.authorization_scope,
            resources,
            &mut RecordingRecoveryPublisher::default(),
        ),
    )
    .await
    .expect("remote modal response-spectrum exact study timed out")
    .unwrap()
    .unwrap_or_else(|message| {
        panic!("remote modal response-spectrum exact study failed: {message}")
    });
    let imported = import_exact_structural_study_result(
        pool.as_ref(),
        response,
        expectation,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    match (expected, imported.analysis) {
        (
            ModalConsumerExpectation::ResponseSpectrum,
            crate::ExactStructuralStudyAnalysisResult::StructuralModalResponseSpectrum(result),
        ) => {
            assert_eq!(result.modal_workload.step_id().as_str(), "modes");
            assert_eq!(result.response_workload.step_id().as_str(), "spectrum");
            assert_terminal_publication_usage(
                &result.response_publication,
                &result.response_publication_baseline_usage,
                &result.response_resource_usage,
            );
            result
                .response_numerical_receipt
                .validate_against(
                    result.response_numerical_receipt.source_result(),
                    &result.response_result,
                )
                .unwrap();
            assert!(result
                .field_recovery
                .field(runmat_analysis_fea::solve::dynamics::StructuralLinearFieldQuantity::Stress)
                .is_some());
        }
        (
            ModalConsumerExpectation::RandomResponse,
            crate::ExactStructuralStudyAnalysisResult::StructuralRandomResponse(result),
        ) => {
            assert_eq!(result.modal_workload.step_id().as_str(), "modes");
            assert_eq!(result.response_workload.step_id().as_str(), "random");
            assert_terminal_publication_usage(
                &result.response_publication,
                &result.response_publication_baseline_usage,
                &result.response_resource_usage,
            );
            result
                .response_numerical_receipt
                .validate_against(
                    result.response_numerical_receipt.source_result(),
                    &result.response_result,
                )
                .unwrap();
            assert_eq!(
                result
                    .statistics_state
                    .payload
                    .structural_random_statistics()
                    .unwrap(),
                &runmat_analysis_fea::execution::StructuralRandomStatistics::from_result(
                    &result.response_result,
                )
                .unwrap()
            );
        }
        _ => panic!("encrypted modal-consumer study returned another result kind"),
    }
}

fn assert_terminal_publication_usage(
    publication: &runmat_analysis_execution::ImportedPublishedResult,
    baseline: &runmat_analysis_execution::PreparedResourceUsage,
    terminal: &runmat_analysis_execution::PreparedResourceUsage,
) {
    assert_ne!(baseline.evidence.identity(), terminal.evidence.identity());
    assert_eq!(
        publication.result.source_resource_usage,
        baseline.evidence.identity()
    );
    runmat_analysis_execution::validate_retained_result_publication_resource_usage(
        &publication.result,
        &baseline.evidence,
        &terminal.evidence,
    )
    .unwrap();

    let added = terminal
        .evidence
        .attempts
        .iter()
        .filter(|attempt| !baseline.evidence.attempts.contains(attempt))
        .collect::<Vec<_>>();
    assert_eq!(added.len(), 1);
    let publication_plan = terminal
        .evidence
        .plans
        .iter()
        .find(|plan| plan.identity == added[0].plan)
        .unwrap();
    assert_eq!(
        publication_plan.input.phase,
        runmat_analysis_fea::planning::resource::FeaResourcePhase::ResultPublication(
            publication.result.request,
        )
    );
}
