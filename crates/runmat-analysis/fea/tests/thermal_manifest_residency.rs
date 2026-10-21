//! Isolated metadata construction/identity/header peak, excluding source histories and transport.
#[path = "../support/peak_allocator.rs"]
mod peak_allocator;

use runmat_analysis_core::*;
use runmat_analysis_fea::{
    admission::admit_nonstructural,
    execution::{ThermalSolverResultManifest, ThermalWorkload},
    fixtures::{fixture_model, FixtureId},
    planning::resource::ThermalPreassemblyResourceFacts,
    ComputeBackend, ThermalProcedureControl,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload(samples: usize) -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut action = model.excitations[0].excitation().clone();
    action.target = ExcitationTarget::WholeModel;
    action.direction = None;
    action.distribution = ExcitationDistribution::IntensityPerVolume;
    action.active_step_ids.clear();
    action.projection.preserve_moment = false;
    let PhysicalSignal::Force(old) = &model.signals[0] else {
        panic!("fixture signal")
    };
    model.signals = vec![PhysicalSignal::HeatGenerationDensity(Signal {
        signal_id: action.signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(
            HeatGenerationDensity::new(0.0),
        )),
        precision: old.precision,
        evidence: old.evidence.clone(),
        provenance: old.provenance.clone(),
    })];
    model.excitations = vec![ExcitationAction::HeatGeneration(action)];
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps(
        "thermal_manifest",
        vec![ProcedureStep::thermal_transient("thermal_α".repeat(128))],
    );
    let mut alternate = model.materials[0].clone();
    alternate.material_id = "expected_material_β".repeat(128);
    model.material_assignments[0].expected_material_id = alternate.material_id.clone();
    model.material_assignments[0].confidence = EvidenceConfidence::Probable;
    model.materials.push(alternate);
    ThermalWorkload {
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler {
            snapshot_count: samples,
        },
        residual_target: 1e-4,
    }
}

#[test]
fn manifest_construction_and_header_peak_fit_composed_reservation() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for samples in [8, 128] {
            let workload = workload(samples);
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let identity = admit_nonstructural(
                PhysicsFamily::Thermal,
                workload.procedure.procedure(),
                Some(&mesh),
                SourceImplementation::CanonicalVolumeProjection,
                BoundaryTransferImplementation::CanonicalBoundaryProjection,
            )
            .unwrap();
            let facts = ThermalPreassemblyResourceFacts::derive(
                &[],
                &workload.model,
                &workload.model.sequence.steps[0],
                &mesh,
                &workload.solve_schedule().unwrap(),
                &identity,
            )
            .unwrap();
            let result = workload.execute(&mesh).unwrap();
            assert_eq!(result.run.spatial_projection_evidence.len(), samples);
            let bound = facts.manifest_retained.reservation_bytes().unwrap().0
                + facts.manifest_catalog.reservation_bytes().unwrap().0
                + facts.manifest_metadata.reservation_bytes().0
                + facts.field_construction.reservation_bytes().0
                + facts.history.validation_scratch_bytes.0;
            let baseline = peak_allocator::begin_measurement();
            let manifest = ThermalSolverResultManifest::from_result(&result).unwrap();
            let bytes = manifest.canonical_bytes().unwrap();
            std::hint::black_box((&manifest, &bytes));
            let peak = peak_allocator::end_measurement().saturating_sub(baseline);
            assert!(
                peak <= bound,
                "{order:?} samples={samples} peak={peak} admitted={bound}"
            );
            println!(
                "{order:?} samples={samples} manifest_peak={peak} admitted={bound} encoded={}",
                bytes.len()
            );
            assert_eq!(
                manifest.identity(),
                result.source_result_identity().unwrap()
            );
        }
    }
    println!(
        "process_peak_rss_bytes={:?}",
        peak_allocator::process_peak_rss_bytes()
    );
}
