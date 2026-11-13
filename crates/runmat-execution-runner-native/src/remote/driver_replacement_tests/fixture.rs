use super::*;

pub(super) struct ExactStudyFixture {
    pub(super) control: DurableControl,
    pub(super) run_key: RunKeyMaterial,
    pub(super) inventory: ResourceInventory,
    pub(super) expectation: StudyExpectation,
}

pub(super) enum StudyExpectation {
    Exact(Box<crate::ExactStructuralStudyResultExpectation>),
    Typed(Box<crate::StructuralStudyResultExpectation>),
    Meshing(ArtifactAccess),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ExactStudyFamily {
    MeshingRoot,
    StructuralTransient,
    ModalResponseSpectrum,
    ModalRandomResponse,
    ThermalPrestress(thermal::Case),
}

impl ExactStudyFamily {
    pub(super) const fn held_attempt_target(
        self,
    ) -> super::super::worker_execution::HeldAttemptTarget {
        use super::super::worker_execution::HeldAttemptTarget;
        use runmat_analysis_fea::execution::AnalysisWorkKind;
        match self {
            Self::MeshingRoot => HeldAttemptTarget::ExactMeshingAdmission,
            Self::StructuralTransient => {
                HeldAttemptTarget::Analysis(AnalysisWorkKind::SolveStructuralTransientBlock)
            }
            Self::ModalResponseSpectrum => HeldAttemptTarget::Analysis(
                AnalysisWorkKind::SolveStructuralModalResponseSpectrumDirection,
            ),
            Self::ModalRandomResponse => HeldAttemptTarget::Analysis(
                AnalysisWorkKind::SolveStructuralRandomResponseIntervals,
            ),
            Self::ThermalPrestress(_) => {
                HeldAttemptTarget::Analysis(AnalysisWorkKind::SolveThermal)
            }
        }
    }

    pub(super) const fn authored_digest_label(self) -> &'static [u8] {
        match self {
            Self::MeshingRoot => b"driver replacement exact meshing root",
            Self::StructuralTransient => b"driver replacement direct transient",
            Self::ModalResponseSpectrum => b"driver replacement modal response spectrum",
            Self::ModalRandomResponse => b"driver replacement modal random response",
            Self::ThermalPrestress(_) => b"driver replacement authored thermal prestress",
        }
    }

    pub(super) const fn interruption(self) -> ReplacementInterruption {
        match self {
            Self::MeshingRoot => ReplacementInterruption::PoolTaskCommitted,
            Self::StructuralTransient => ReplacementInterruption::RecoveryClosure,
            Self::ModalResponseSpectrum | Self::ModalRandomResponse | Self::ThermalPrestress(_) => {
                ReplacementInterruption::ReplayableExecutionStart
            }
        }
    }

    fn workloads(self) -> StructuralStudyWorkloads {
        match self {
            Self::MeshingRoot => unreachable!("meshing root does not carry structural workloads"),
            Self::StructuralTransient => StructuralStudyWorkloads::StructuralTransient {
                workload: Box::new(box_direct_transient_workload()),
                block_plan: StructuralTransientBlockPlan::new(1).unwrap(),
            },
            Self::ModalResponseSpectrum => crate::exact_structural_sweep_root::conformance_tests::box_modal_response_spectrum_workloads(),
            Self::ModalRandomResponse => crate::exact_structural_sweep_root::conformance_tests::box_random_response_workloads(),
            Self::ThermalPrestress(_) => unreachable!("typed root compiles authored workloads"),
        }
    }
}

pub(super) fn exact_study_fixture(family: ExactStudyFamily) -> ExactStudyFixture {
    exact_study_fixture_with_wall_time(family, resources().maximum_wall_millis)
}

pub(super) fn exact_study_fixture_with_wall_time(
    family: ExactStudyFamily,
    maximum_wall_millis: u64,
) -> ExactStudyFixture {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    if let ExactStudyFamily::ThermalPrestress(case) = family {
        thermal::write_authored_project(project_root.path(), case);
    }
    let project_document =
        "[package]\nname = \"driver-replacement-study\"\n[sources]\nroots = [\"src\"]\n[runtime.language]\ncompat = \"matlab\"\n";
    std::fs::write(project_root.path().join("runmat.toml"), project_document).unwrap();
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
    let (request, objects, programs, expectation) =
        if let ExactStudyFamily::ThermalPrestress(case) = family {
            let prepared = thermal::prepare_authored_root(
                project_root.path(),
                revision.clone(),
                access.clone(),
                case,
            );
            (
                prepared.request().clone(),
                prepared.transport_objects().unwrap(),
                prepared
                    .required_programs()
                    .iter()
                    .map(|program| (program.recipe.clone(), program.artifact.clone()))
                    .collect::<Vec<_>>(),
                StudyExpectation::Typed(Box::new(prepared.result_expectation())),
            )
        } else if family == ExactStudyFamily::MeshingRoot {
            let prepared = prepare_exact_meshing_root(ExactMeshingRootPreparation {
                program_revision: revision.clone(),
                artifact_access: access.clone(),
                source_name: "box.brep",
                source_bytes: BOX,
                import_options: ExactCadImportOptions {
                    source_units: Some(runmat_geometry_core::LengthUnit::Meter),
                    ..ExactCadImportOptions::default()
                },
                contact_definitions: Vec::new(),
                meshing_request: coarse_box_request(),
                preferred_edges_per_partition: 64,
                preferred_faces_per_partition: 64,
                maximum_chunk_bytes: 1024,
            })
            .unwrap();
            let objects = prepared
                .request()
                .arguments
                .iter()
                .map(|argument| {
                    let runmat_execution::value::ValuePayload::Object(reference) = argument else {
                        panic!("meshing root inputs must be portable objects")
                    };
                    let source = prepared
                        .objects()
                        .iter()
                        .find(|object| object.descriptor.digest == reference.logical_digest)
                        .expect("meshing source object");
                    ProgramExecutionObject {
                        reference: reference.as_ref().clone(),
                        bytes: source.bytes.clone(),
                    }
                })
                .collect();
            (
                prepared.request().clone(),
                objects,
                prepared
                    .required_programs()
                    .unwrap()
                    .into_iter()
                    .map(|program| (program.recipe, program.artifact))
                    .collect::<Vec<_>>(),
                StudyExpectation::Meshing(access.clone()),
            )
        } else {
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
                meshing_request,
                preferred_edges_per_partition: 64,
                preferred_faces_per_partition: 64,
                maximum_chunk_bytes: 1024,
            })
            .unwrap();
            let prepared =
                prepare_exact_structural_study_root(ExactStructuralStudyRootPreparation {
                    program_revision: revision.clone(),
                    artifact_access: access,
                    authored_study_digest: Digest::sha256(family.authored_digest_label()),
                    frozen_project_digest: Some(Digest::from_bytes(
                        *project_identity.graph_digest.bytes(),
                    )),
                    geometry_source_digest: Digest::sha256(BOX),
                    workloads: family.workloads(),
                    analysis_accelerator_requests: Vec::new(),
                    local_execution:
                        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
                    input_materializations: Vec::new(),
                    exact_meshing,
                })
                .unwrap();
            (
                prepared.request().clone(),
                prepared.transport_objects().unwrap(),
                prepared
                    .required_programs()
                    .unwrap()
                    .into_iter()
                    .map(|program| (program.recipe, program.artifact))
                    .collect::<Vec<_>>(),
                StudyExpectation::Exact(Box::new(prepared.result_expectation())),
            )
        };
    let mut builder = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_project_runtime_policy_object(
            runmat_core::cluster_settings::capture_project_policy(
                &runmat_config::runtime::ClusterRuntimePolicy::from_project_document(
                    project_document,
                    runmat_config::document::RunmatConfigFormat::Toml,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
        .with_compiled_package_closure();
    for (recipe, artifact) in programs {
        builder =
            builder.with_materialized_program(recipe, artifact.form, artifact.executable_bytes);
    }
    let mut bundle = Vec::new();
    write_bundle(
        &builder.build().unwrap(),
        &mut bundle,
        ArchiveLimits::default(),
    )
    .unwrap();
    let descriptor = serde_json::to_vec(&ProgramExecutionDescriptor {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: request.recipe.clone(),
        artifact: request.artifact.clone(),
        callable: request.callable.clone(),
        requested_outputs: request.requested_outputs,
    })
    .unwrap();
    let run_key = RunKeyMaterial::from_entropy([41; 32]).unwrap();
    let mut artifacts = BTreeMap::new();
    let mut locations = Vec::new();
    for (index, object) in objects.into_iter().enumerate() {
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
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        context: runmat_execution::ProgramInvocationContext::Direct,
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
    let mut resources = resources();
    resources.maximum_wall_millis = maximum_wall_millis;
    let inventory = super::super::pool_resources::inventory(&resources, &[]).unwrap();
    let workers = [
        (1, worker_allocation("worker-a", 11, &resources)),
        (2, worker_allocation("worker-b", 12, &resources)),
        (3, worker_allocation("worker-c", 13, &resources)),
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
            worker_identity_secrets: BTreeMap::from([(1, [31; 32]), (2, [32; 32]), (3, [33; 32])]),
            interruption: family.interruption(),
            interrupted_target: family.held_attempt_target(),
            state: Mutex::new(DurableState {
                artifacts,
                initial_artifacts,
                latest_checkpoint: None,
                next_artifact: 0,
                driver_interruption_triggered: false,
                second_driver_interruption_triggered: false,
                interrupted_committed_task: None,
                cancellation_heartbeat_observed: false,
                transitions: Vec::new(),
                worker_authorizations: Vec::new(),
                bootstrap_checkpoints: Vec::new(),
                retirements: BTreeMap::new(),
            }),
            recovery_committed: tokio::sync::Notify::new(),
            clock_control: Mutex::new(None),
        },
        run_key,
        inventory,
        expectation,
    }
}

fn box_direct_transient_workload() -> StructuralTransientWorkload {
    let selectors = crate::exact_structural_sweep_root::conformance_tests::box_selectors();
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.geometry_id = "box".into();
    model.geometry_revision = 1;
    model.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::solid(selectors.solid_selector());
    model.constraints[0].visit_targets_mut(|target| {
        *target = runmat_analysis_core::RegionTarget::boundary(selectors.fixed_face_selector());
    });
    let runmat_analysis_core::ExcitationTarget::Region(target) =
        &mut model.excitations[0].excitation_mut().target
    else {
        panic!("cantilever fixture must use a regional excitation")
    };
    *target = runmat_analysis_core::RegionTarget::boundary(selectors.loaded_face_selector());
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

pub(super) fn coarse_box_request() -> MeshingRequest {
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

fn resources() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        // The simulated remote worker runs on this test host. Advertise only
        // the existing OS/container-capped machine budget; a fixed 1 GiB
        // inventory underreported the authored thermal/random task envelopes.
        memory_bytes: crate::machine_capacity::memory_bytes()
            .expect("driver replacement fixture requires discoverable host memory"),
        scratch_bytes: 1 << 30,
        accelerators: Vec::new(),
        maximum_wall_millis: 180_000,
    }
}

fn worker_allocation(
    allocation_id: &str,
    entropy: u8,
    resources: &ResourceRequest,
) -> DriverWorkerAllocation {
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
        org_id: crate::remote::test_ids::TEST_ORG_ID.into(),
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
        resources: resources.clone(),
        accelerator_devices: Vec::new(),
        host: crate::test_support::host_inventory(),
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
            org_id: crate::remote::test_ids::TEST_ORG_ID.into(),
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
        org_id: crate::remote::test_ids::test_org_id(),
        project_id: "project-1".into(),
        allocation_lease_id: DRIVER_ALLOCATION.into(),
        driver_lease_id: crate::remote::test_ids::test_driver_lease_id(fence),
        fencing_token: fence,
        credential: format!("credential-{fence}"),
    }
}

pub(super) async fn run_worker(
    relay_url: String,
    allocation_id: &str,
    inventory: &ResourceInventory,
    run_key: RunKeyMaterial,
    driver_fence: u64,
    held_attempt: Option<super::super::worker_execution::HeldWorkloadAttempt>,
    journal: super::super::worker_attempt_journal::WorkerAttemptJournalOwner,
) -> crate::NativeExecutionResult<()> {
    let allocation_id = allocation_id.to_string();
    let worker = WorkerSpec {
        host: crate::test_support::host_inventory(),
        id: runmat_execution::identity::WorkerId::derive(&[
            RUN_ID.as_bytes(),
            allocation_id.as_bytes(),
        ]),
        pool_id: runmat_execution::identity::PoolId::derive(&[RUN_ID.as_bytes(), b"remote-pool"]),
        resources: inventory.clone(),
    };
    let request = RemoteWorkerRelayRequest {
        url: &relay_url,
        headers: &[],
        run_identity: RUN_ID.into(),
        worker,
        driver_fence,
        session_id: session_id(RUN_ID, &allocation_id),
        run_key,
        limits: FrameLimits::default(),
        ready_v1: false,
    };
    if let Some(attempt) = held_attempt {
        super::super::worker_entry::run_remote_meshing_worker_relay_with_held_attempt(
            request,
            Arc::new(native_meshing_kernel_dispatcher()),
            NativeMeshingHostLimits::default(),
            attempt,
            journal,
        )
        .await
    } else {
        super::super::worker_entry::run_remote_meshing_worker_relay_cached_with_journal(
            request,
            None,
            Arc::new(native_meshing_kernel_dispatcher()),
            NativeMeshingHostLimits::default(),
            journal,
        )
        .await
    }
}
