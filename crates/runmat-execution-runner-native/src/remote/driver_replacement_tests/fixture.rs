use super::*;

pub(super) struct ExactStudyFixture {
    pub(super) control: DurableControl,
    pub(super) run_key: RunKeyMaterial,
    pub(super) inventory: ResourceInventory,
    pub(super) expectation: crate::ExactStructuralStudyResultExpectation,
}

pub(super) fn exact_study_fixture() -> ExactStudyFixture {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"driver-replacement-study\"\n[sources]\nroots = [\"src\"]\n",
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
        authorization_scope: RUN_ID.into(),
        encryption_context: Digest::sha256(b"driver replacement exact study encryption"),
    };
    let meshing_request = coarse_box_request();
    let exact_meshing = prepare_exact_meshing_root(ExactMeshingRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access.clone(),
        source_name: "box.brep",
        source_bytes: BOX,
        import_options: ExactCadImportOptions::default(),
        contact_definitions: Vec::new(),
        meshing_request,
        preferred_edges_per_partition: 64,
        preferred_faces_per_partition: 64,
        maximum_chunk_bytes: 1024,
    })
    .unwrap();
    let prepared = prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
        program_revision: revision.clone(),
        artifact_access: access,
        authored_study_digest: Digest::sha256(b"driver replacement direct transient"),
        frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
        geometry_source_digest: Digest::sha256(BOX),
        workloads: StructuralStudyWorkloads::DirectTransient {
            workload: Box::new(box_direct_transient_workload()),
            block_plan: StructuralTransientBlockPlan::new(1).unwrap(),
        },
        analysis_accelerator_requests: Vec::new(),
        signal_materializations: Vec::new(),
        exact_meshing,
    })
    .unwrap();
    let expectation = prepared.result_expectation();
    let mut builder = ExecutionBundleBuilder::native(&project, revision.clone())
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
    let request = prepared.request().clone();
    let descriptor = serde_json::to_vec(&ProgramExecutionDescriptor {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe: request.recipe.clone(),
        artifact: request.artifact.clone(),
        function: request.function,
        requested_outputs: request.requested_outputs,
    })
    .unwrap();
    let run_key = RunKeyMaterial::from_entropy([41; 32]).unwrap();
    let mut artifacts = BTreeMap::new();
    let mut locations = Vec::new();
    for (index, object) in prepared
        .transport_objects()
        .unwrap()
        .into_iter()
        .enumerate()
    {
        let artifact_id = format!("input-object-{index}");
        let ciphertext =
            seal_object(&run_key, RUN_ID, EncryptionPurpose::Input, &object.bytes).unwrap();
        artifacts.insert(
            artifact_id.clone(),
            StoredArtifact {
                kind: "input".into(),
                ciphertext,
            },
        );
        locations.push(ProgramExecutionObjectLocation {
            reference: object.reference,
            artifact_id,
        });
    }
    let inputs = serde_json::to_vec(&ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        arguments: request.arguments,
        objects: locations,
    })
    .unwrap();
    let initial = [
        ("bundle", EncryptionPurpose::Bundle, bundle),
        ("program", EncryptionPurpose::Program, descriptor),
        ("input", EncryptionPurpose::Input, inputs),
    ];
    let mut initial_artifacts = Vec::new();
    for (index, (kind, purpose, plaintext)) in initial.into_iter().enumerate() {
        let artifact_id = format!("initial-{index}");
        let ciphertext = seal_object(&run_key, RUN_ID, purpose, &plaintext).unwrap();
        let stored = StoredArtifact {
            kind: kind.into(),
            ciphertext,
        };
        initial_artifacts.push(artifact_download(&artifact_id, &stored));
        artifacts.insert(artifact_id, stored);
    }
    let resources = resources();
    let inventory = super::super::pool_resources::inventory(&resources).unwrap();
    let workers = [
        (1, worker_allocation("worker-a", 11)),
        (2, worker_allocation("worker-b", 12)),
    ]
    .into_iter()
    .collect();
    let driver_material = driver_identity_material();
    let driver_recipient = runmat_execution_artifact::encryption::ExecutionRecipientKey {
        suite:
            runmat_execution_artifact::encryption::ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1,
        fingerprint: driver_material.evidence.recipient.fingerprint.clone(),
        public_key: driver_material.evidence.recipient.public_key.clone(),
        valid_after_unix_millis: driver_material.evidence.recipient.valid_after_unix_millis,
        valid_before_unix_millis: driver_material.evidence.recipient.valid_before_unix_millis,
        custodian_uri: None,
    };
    let envelope = PortableExecutionEncryption
        .seal_run_key_with_entropy([43; 32], &driver_recipient, &run_key, RUN_ID, 1)
        .unwrap();
    ExactStudyFixture {
        control: DurableControl {
            project_revision: project_revision_identity(&revision),
            endpoint_fingerprint: driver_recipient.fingerprint.clone(),
            run_key_envelope: encode_run_key_envelope(&envelope).unwrap(),
            run_key: run_key.clone(),
            resources,
            workers,
            state: Mutex::new(DurableState {
                artifacts,
                initial_artifacts,
                latest_checkpoint: None,
                next_artifact: 0,
                driver_interrupted_at_commit: false,
                transitions: Vec::new(),
                worker_authorizations: Vec::new(),
                bootstrap_checkpoints: Vec::new(),
            }),
            recovery_committed: tokio::sync::Notify::new(),
        },
        run_key,
        inventory,
        expectation,
    }
}

fn box_direct_transient_workload() -> StructuralTransientWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_id = "box".into();
    model.geometry_revision = 1;
    model.material_assignments.clear();
    model.constraints[0].visit_targets_mut(|target| {
        *target = runmat_analysis_core::RegionTarget::boundary("face_000001");
    });
    let runmat_analysis_core::ExcitationTarget::Region(target) =
        &mut model.excitations[0].excitation_mut().target
    else {
        panic!("cantilever fixture must use a regional excitation")
    };
    *target = runmat_analysis_core::RegionTarget::boundary("face_000002");
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

fn resources() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1 << 30,
        scratch_bytes: 1 << 30,
        accelerator_count: 0,
        accelerator_class: None,
        accelerator_memory_bytes: 0,
        maximum_wall_millis: 180_000,
    }
}

fn worker_allocation(allocation_id: &str, entropy: u8) -> DriverWorkerAllocation {
    let now = now_millis();
    let (recipient, _) = PortableExecutionEncryption
        .recipient_from_entropy_with_derived_fingerprint(
            [entropy; 32],
            now.saturating_sub(30_000),
            now + 3_600_000,
        )
        .unwrap();
    let signer = EndpointIdentitySigner::from_secret([entropy + 20; 32]).unwrap();
    let mut evidence = EndpointIdentityEvidence {
        schema_version: 1,
        org_id: "org-1".into(),
        cluster_id: "cluster-1".into(),
        node_id: format!("node-{allocation_id}"),
        allocation_lease_id: allocation_id.into(),
        fencing_token: 1,
        run_identity: RUN_ID.into(),
        identity_public_key: signer.public_key().to_vec(),
        identity_fingerprint: signer.fingerprint(),
        recipient: EndpointRecipientKey {
            suite: "x25519-hkdf-sha256-aes128gcm-v1".into(),
            public_key: recipient.public_key,
            fingerprint: recipient.fingerprint,
            valid_after_unix_millis: recipient.valid_after_unix_millis,
            valid_before_unix_millis: recipient.valid_before_unix_millis,
        },
        direct_quic_endpoints: Vec::new(),
        trust_tier: ExecutionTrustTier::CustomerTrusted,
        attestation_class: None,
        attestation_evidence: None,
        issued_at_unix_millis: now,
        expires_at_unix_millis: now + 3_600_000,
        signature: vec![0; 64],
    };
    signer.sign(&mut evidence).unwrap();
    DriverWorkerAllocation {
        allocation_lease_id: allocation_id.into(),
        fencing_token: 1,
        state: "active".into(),
        resources: resources(),
        endpoint_identity: Some(evidence),
        run_key_envelope_authorized: false,
    }
}

pub(super) fn driver_identity_material() -> EndpointIdentityMaterial {
    let now = now_millis();
    let entropy = [37; 32];
    let (recipient, _) = PortableExecutionEncryption
        .recipient_from_entropy_with_derived_fingerprint(
            entropy,
            now.saturating_sub(30_000),
            now + 3_600_000,
        )
        .unwrap();
    EndpointIdentityMaterial::new(
        entropy,
        EndpointIdentityEvidence {
            schema_version: 1,
            org_id: "org-1".into(),
            cluster_id: "cluster-1".into(),
            node_id: "driver-node".into(),
            allocation_lease_id: DRIVER_ALLOCATION.into(),
            fencing_token: 1,
            run_identity: RUN_ID.into(),
            identity_public_key: vec![1; 32],
            identity_fingerprint: "unused-by-driver".into(),
            recipient: EndpointRecipientKey {
                suite: "x25519-hkdf-sha256-aes128gcm-v1".into(),
                public_key: recipient.public_key,
                fingerprint: recipient.fingerprint,
                valid_after_unix_millis: recipient.valid_after_unix_millis,
                valid_before_unix_millis: recipient.valid_before_unix_millis,
            },
            direct_quic_endpoints: Vec::new(),
            trust_tier: ExecutionTrustTier::CustomerTrusted,
            attestation_class: None,
            attestation_evidence: None,
            issued_at_unix_millis: now,
            expires_at_unix_millis: now + 3_600_000,
            signature: vec![0; 64],
        },
    )
}

pub(super) fn driver_authority(fence: u64, server_url: String) -> DriverAuthority {
    DriverAuthority {
        server_url,
        run_id: RUN_ID.into(),
        org_id: "org-1".into(),
        project_id: "project-1".into(),
        allocation_lease_id: DRIVER_ALLOCATION.into(),
        driver_lease_id: format!("driver-{fence}"),
        fencing_token: fence,
        credential: format!("credential-{fence}"),
    }
}

pub(super) async fn run_worker(
    relay_url: String,
    allocation: &DriverWorkerAllocation,
    inventory: &ResourceInventory,
    run_key: RunKeyMaterial,
    driver_fence: u64,
) -> crate::NativeExecutionResult<()> {
    let allocation_id = allocation.allocation_lease_id.clone();
    let worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[
            RUN_ID.as_bytes(),
            allocation_id.as_bytes(),
        ]),
        pool_id: runmat_execution::identity::PoolId::derive(&[RUN_ID.as_bytes(), b"remote-pool"]),
        resources: inventory.clone(),
    };
    run_remote_meshing_worker_relay_cached(
        RemoteWorkerRelayRequest {
            url: &relay_url,
            headers: &[],
            run_identity: RUN_ID.into(),
            worker,
            driver_fence,
            session_id: session_id(RUN_ID, &allocation_id),
            run_key,
            limits: FrameLimits::default(),
        },
        None,
        Arc::new(native_meshing_kernel_dispatcher()),
        NativeMeshingHostLimits::default(),
    )
    .await
}
