use super::*;
use runmat_analysis_core::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn fixture_with_memory(
    maximum_resident_bytes: u64,
) -> (
    AcousticFrequencyWorkload,
    SolverMeshArtifact,
    Vec<PreparedAnalysisObjects>,
) {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(3.0));
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    let mut step = ProcedureStep::acoustic_harmonic("acoustic");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![0.1, 0.2]);
    step.accuracy.resources.maximum_resident_bytes = Some(maximum_resident_bytes);
    model.sequence.steps = vec![step];
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "pressure".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
                value: PrescribedValue::Constant(Pressure::new(1.0)),
            },
        }));
    let workload =
        AcousticFrequencyWorkload::new(runmat_analysis_fea::execution::AcousticHarmonicWorkload {
            model,
            step_id: ProcedureStepId("acoustic".into()),
            frequency: Frequency::new(0.1),
            solver: Default::default(),
            power_balance_relative_tolerance: 1e-8,
        })
        .unwrap();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let entries = workload
        .plan()
        .entries()
        .unwrap()
        .into_iter()
        .map(|entry| {
            let result = workload.entry(entry).unwrap().execute(&mesh).unwrap();
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::SolverResult,
                result.source_result_identity().unwrap().digest(),
                &result.canonical_bytes().unwrap(),
                1024,
                ObjectNamespace::ResultValue,
                ObjectInventoryLimits::default(),
            )
            .unwrap()
        })
        .collect();
    (workload, mesh, entries)
}

#[test]
fn source_bound_original_objects_reject_missing_duplicate_foreign_and_metadata_tampering() {
    let (workload, mesh, entries) = fixture_with_memory(8 * 1024 * 1024 * 1024);
    let mut numerical = decode_acoustic_frequency_assembly_entries(&entries).unwrap();
    validate_acoustic_frequency_assembly_entries(&workload, &mesh, &numerical).unwrap();
    numerical.reverse();
    validate_acoustic_frequency_assembly_entries(&workload, &mesh, &numerical).unwrap();
    assert!(
        validate_acoustic_frequency_assembly_entries(&workload, &mesh, &numerical[..1]).is_err()
    );
    numerical[1] = numerical[0].clone();
    assert!(validate_acoustic_frequency_assembly_entries(&workload, &mesh, &numerical).is_err());
    let mut tampered = entries;
    tampered[0].manifest.logical_identity = Digest::sha256("foreign");
    assert!(decode_acoustic_frequency_assembly_entries(&tampered).is_err());
}

#[test]
fn aggregate_admission_cannot_enlarge_original_authored_memory_ceiling() {
    let (workload, mesh, entries) = fixture_with_memory(1024 * 1024);
    let numerical = decode_acoustic_frequency_assembly_entries(&entries).unwrap();
    validate_acoustic_frequency_assembly_entries(&workload, &mesh, &numerical).unwrap();
    let mut inputs = vec![
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::AcousticFrequencyWorkload,
            workload.identity().unwrap().digest(),
            &workload.canonical_bytes().unwrap(),
            1024,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::Mesh,
            Digest::from_bytes(*mesh.canonical_digest.bytes()),
            &mesh.canonical_encode().unwrap(),
            1024,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
    ];
    inputs.extend(entries);
    let error =
        rederive_acoustic_frequency_assembly_resource_plan(&workload, &mesh, &numerical, &inputs)
            .unwrap_err();
    assert!(matches!(
        error,
        AnalysisBridgeError::Resource(
            runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling { .. }
        )
    ));
}

#[test]
fn aggregate_accounts_mesh_as_input_not_as_emitted_output() {
    let (workload, mesh, entries) = fixture_with_memory(8 * 1024 * 1024 * 1024);
    let request = runmat_analysis_fea::execution::AcousticFrequencyAssemblyRequest::new(vec![
        None;
        entries.len()
    ])
    .unwrap();
    let prepared = prepare_acoustic_frequency_assembly_task(AcousticFrequencyAssemblyPreparation {
        authored_study_digest: Digest::sha256("authored"),
        frozen_project_digest: None,
        workload,
        mesh,
        entries,
        request: request.clone(),
        artifact_access: ArtifactAccess {
            authorization_scope: "test".into(),
            encryption_context: Digest::sha256("scope"),
        },
        program_revision: ProgramRevision::new(
            Digest::sha256("graph"),
            Digest::sha256("source"),
            runmat_execution::ProgramEnvironment::new(
                1,
                1,
                Digest::sha256("runtime"),
                Digest::sha256("catalog"),
                "matlab",
            )
            .unwrap(),
        )
        .unwrap(),
        output_policy: AnalysisOutputPolicy::new(1024, ObjectInventoryLimits::default()).unwrap(),
    })
    .unwrap();
    let mesh_bytes = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::Mesh)
        .unwrap()
        .manifest
        .encoded_length;
    let emitted_bytes: u64 = prepared
        .inputs
        .iter()
        .filter(|input| {
            matches!(
                input.manifest.kind,
                AnalysisNumericalObjectKind::AcousticFrequencyWorkload
                    | AnalysisNumericalObjectKind::AcousticFrequencyAssemblyRequest
                    | AnalysisNumericalObjectKind::SolverResult
            )
        })
        .map(|input| input.manifest.encoded_length)
        .sum();
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::AcousticFrequencyAssembly(
        demand,
    ) = &prepared.resource_plan.input.demand
    else {
        panic!("expected assembly demand");
    };
    assert_eq!(
        demand.inputs.encoded_payload_bytes.0,
        emitted_bytes + mesh_bytes
    );
    assert_eq!(
        prepared.resource_plan.breakdown.input_encoded_payload_bytes,
        demand.inputs.encoded_payload_bytes
    );
    let profile = AcousticFrequencyResult::assembly_encoding_profile(
        &prepared.workload,
        &prepared.mesh,
        &prepared.entries,
    )
    .unwrap();
    assert_eq!(
        demand.result_output.encoded_payload_bytes_upper_bound.0,
        profile.encoded_bytes_upper_bound
    );
    assert!(demand.inputs.decoded_resident_bytes_admission_budget.0 > mesh_bytes);
    let assembled = runmat_analysis_fea::execution::assemble_acoustic_frequency_result(
        prepared.workload.clone(),
        &prepared.mesh,
        prepared.entries.clone(),
        &request,
    )
    .unwrap();
    let actual_result_profile = runmat_canonical_codec::profile_serialized(
        &assembled.result,
        runmat_canonical_codec::CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
    )
    .unwrap();
    assert_eq!(profile, actual_result_profile);
    assert!(
        assembled.result.canonical_bytes().unwrap().len() as u64
            <= demand.result_output.encoded_payload_bytes_upper_bound.0
    );
    assert!(
        assembled.publication.canonical_bytes().unwrap().len() as u64
            <= demand
                .publication_output
                .encoded_payload_bytes_upper_bound
                .0
    );
    // Fields are recovered and published by each frequency entry's own task; the fan-in
    // emits only the assembled result and publication manifest.
}
