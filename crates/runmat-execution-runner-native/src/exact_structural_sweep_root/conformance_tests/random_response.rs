use super::*;

#[tokio::test(flavor = "current_thread")]
async fn random_response_reaches_an_admitted_result_over_encrypted_quic() {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"random-response-conformance\"\n[sources]\nroots = [\"src\"]\n",
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
        authorization_scope: "random-response-conformance-run".into(),
        encryption_context: Digest::sha256(b"random-response-conformance-encryption"),
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
            authored_study_digest: Digest::sha256(b"box random-response study"),
            frozen_project_digest: Some(Digest::from_bytes(*project_identity.graph_digest.bytes())),
            geometry_source_digest: Digest::sha256(BOX),
            workloads: box_random_response_workloads(),
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
    // This positive conformance run provisions a real local worker. Its
    // canonical random-response plan can exceed the old synthetic 1 GiB pool;
    // use the same finite host-capacity snapshot as ordinary native workers.
    let memory_bytes = crate::machine_capacity::memory_bytes().unwrap();
    let resources = ResourceRequest {
        cpu_millicores: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial()
            .checked_cpu_millicores()
            .unwrap()
            .into(),
        memory_bytes,
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
    super::response_spectrum::run_response_spectrum_conformance(
        prepared,
        expectation,
        bundle,
        inventory,
        resources,
        access,
        super::response_spectrum::ModalConsumerExpectation::RandomResponse,
    )
    .await;
}
