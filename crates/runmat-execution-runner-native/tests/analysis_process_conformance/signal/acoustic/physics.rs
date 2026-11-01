//! Generated/explicit equivalence on admitted canonical frequency tasks; no distributed solve claim.
use super::{Drive, SignalFixture};
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    execute_serial_analysis, import_acoustic_harmonic_result, prepare_acoustic_harmonic_task,
    AcousticHarmonicTaskPreparation, AnalysisHostResponse, AnalysisOutputPolicy,
};
use runmat_analysis_fea::{
    execution::{
        AcousticFrequencyResult, AcousticFrequencyWorkload, AcousticHarmonicSolverResult,
        AcousticHarmonicWorkload,
    },
    physics::acoustic::AcousticHarmonicPolicy,
};
use runmat_execution_artifact::{
    cache::FilesystemObjectStore, object::ObjectInventoryLimits, ArtifactAccess,
};
use runmat_execution_runner_native::NativeAnalysisKernel;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, SolverMeshArtifact,
};

const FREQUENCIES: [f64; 5] = [0.125, 0.1875, 0.25, 0.375, 0.5];

pub(super) fn compare(drive: Drive, fixture: &SignalFixture, generated: &[(&str, PhysicalSignal)]) {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut explicit_identity = None;
        let mut generated_identities = vec![None; generated.len()];
        for chunk in [1024, 8192] {
            let explicit = execute(drive, fixture, fixture.explicit.clone(), &mesh, chunk);
            let identity = explicit.source_result_identity().unwrap();
            if let Some(previous) = explicit_identity {
                assert_eq!(identity, previous);
            }
            explicit_identity = Some(identity);
            for (index, (placement, table)) in generated.iter().enumerate() {
                let actual = execute(drive, fixture, table.clone(), &mesh, chunk);
                let identity = actual.source_result_identity().unwrap();
                // Equal physical ordinates do not erase actual generator provenance.
                assert_ne!(identity, explicit_identity.unwrap());
                if let Some(previous) = generated_identities[index] {
                    assert_eq!(identity, previous);
                }
                generated_identities[index] = Some(identity);
                for (a, b) in actual.results().iter().zip(explicit.results()) {
                    assert_eq!(a.frequency(), b.frequency());
                    assert_eq!(
                        a.pressure_pa(),
                        b.pressure_pa(),
                        "{drive:?}/{order:?}/{placement}"
                    );
                    assert_eq!(a.velocity_m_per_s(), b.velocity_m_per_s());
                    assert_eq!(a.intensity_w_per_m2(), b.intensity_w_per_m2());
                    assert_eq!(a.energy_density_j_per_m3(), b.energy_density_j_per_m3());
                    assert_eq!(
                        a.mean_input_power_w().to_bits(),
                        b.mean_input_power_w().to_bits()
                    );
                    assert_eq!(
                        a.mean_impedance_loss_w().to_bits(),
                        b.mean_impedance_loss_w().to_bits()
                    );
                    assert!(a.mean_impedance_loss_w() > 0.0);
                    assert!(a.mean_input_power_w() > 0.0);
                    assert!(a.power_relative_error() < 1e-8);
                    assert_eq!(
                        a.relative_residual().to_bits(),
                        b.relative_residual().to_bits()
                    );
                    assert_eq!(a.source_evidence(), b.source_evidence());
                    if !matches!(drive, Drive::Pressure) {
                        assert_eq!(a.source_evidence().len(), 1);
                        assert!(a.source_evidence()[0].accepted);
                    }
                }
            }
        }
        assert!(generated_identities
            .windows(2)
            .all(|pair| pair[0] == pair[1]));
    }
}

fn execute(
    drive: Drive,
    fixture: &SignalFixture,
    signal: PhysicalSignal,
    mesh: &SolverMeshArtifact,
    chunk: usize,
) -> AcousticFrequencyResult {
    let sweep = AcousticFrequencyWorkload::new(workload(drive, signal)).unwrap();
    let digest = sweep.identity().unwrap().digest();
    let limits = ObjectInventoryLimits::default();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("acoustic-objects");
    let mut results = Vec::new();
    let mut tasks = std::collections::BTreeSet::new();
    let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
    // Independent tasks complete backwards; canonical fan-in restores declared order.
    for entry in sweep.plan().entries().unwrap().into_iter().rev() {
        let workload = sweep.entry(entry).unwrap();
        let task = prepare_acoustic_harmonic_task(AcousticHarmonicTaskPreparation {
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            authored_study_digest: digest,
            frozen_project_digest: None,
            program_revision: fixture.revision.clone(),
            artifact_access: ArtifactAccess {
                authorization_scope: "study:acoustic-spectrum".into(),
                encryption_context: digest,
            },
            workload: workload.clone(),
            mesh: mesh.clone(),
            output_policy: AnalysisOutputPolicy::new(chunk, limits).unwrap(),
            required_capabilities: Default::default(),
        })
        .unwrap();
        assert!(tasks.insert(task.host.workload.execution_plan));
        task.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.program.request,
            &mut store,
            &NativeAnalysisKernel,
            chunk,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&task.host, &completed)
            .unwrap()
            .program_response();
        drop(store);
        store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let imported = import_acoustic_harmonic_result(&store, response, &task, limits).unwrap();
        imported.result.validate_for(&workload, mesh).unwrap();
        let reopened = AcousticHarmonicSolverResult::from_canonical_bytes(
            &imported.result.canonical_bytes().unwrap(),
        )
        .unwrap();
        reopened.validate_for(&workload, mesh).unwrap();
        assert_eq!(reopened, imported.result);
        results.push(reopened);
    }
    assert_eq!(tasks.len(), FREQUENCIES.len());
    let assembled =
        AcousticFrequencyResult::assemble(sweep.clone(), mesh, results.clone()).unwrap();
    results.reverse();
    assert_eq!(
        assembled,
        AcousticFrequencyResult::assemble(sweep.clone(), mesh, results).unwrap()
    );
    let cold = AcousticFrequencyResult::from_canonical_bytes(&assembled.canonical_bytes().unwrap())
        .unwrap();
    cold.validate_for(&sweep, mesh).unwrap();
    assert_eq!(cold, assembled);
    assert_eq!(
        cold.results()
            .iter()
            .map(|r| r.frequency().canonical_value())
            .collect::<Vec<_>>(),
        FREQUENCIES
    );
    cold
}

fn workload(drive: Drive, signal: PhysicalSignal) -> AcousticHarmonicWorkload {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    let id = signal.signal_id().clone();
    model.signals = vec![signal];
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries = vec![AnalysisBoundary {
        boundary_id: "matched-outlet".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:3")),
        operator: BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(6.0),
        },
    }];
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(3.0));
    model.materials[0].acoustic = Some(MaterialAcousticModel {
        speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(2.0),
        damping_ratio: runmat_types::physical::DampingRatio::new(0.0),
    });
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("solid"));
    let mut step = ProcedureStep::acoustic_harmonic("acoustic-spectrum");
    step.independent_domain = IndependentDomain::FrequencyHertz(FREQUENCIES.to_vec());
    model.sequence.steps = vec![step];
    let action = AnalysisExcitation {
        excitation_id: ExcitationId("drive".into()),
        target: ExcitationTarget::Region(RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("face:0"),
        )),
        frame: CoordinateFrame::BoundaryNormalTangent,
        direction: None,
        distribution: ExcitationDistribution::UniformKinematic,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: id,
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: false,
            relative_tolerance: 1e-10,
        },
    };
    model.excitations.push(match drive {
        Drive::Pressure => ExcitationAction::AcousticPressure(action),
        Drive::NormalVelocity => ExcitationAction::NormalVelocity(action),
        Drive::IncidentPressure => ExcitationAction::AcousticIncidentPressure(action),
    });
    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("drive".into()),
        action_ids: vec![ExcitationId("drive".into())],
        activation: ExcitationCaseActivation::Always,
    });
    AcousticHarmonicWorkload {
        model,
        step_id: ProcedureStepId("acoustic-spectrum".into()),
        frequency: Frequency::new(FREQUENCIES[0]),
        solver: AcousticHarmonicPolicy::default(),
        power_balance_relative_tolerance: 1e-8,
    }
}
