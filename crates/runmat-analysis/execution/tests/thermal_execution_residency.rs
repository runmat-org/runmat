//! Isolated requested Rust heap across thermal preparation, solve, and output construction.
#[path = "../../fea/support/peak_allocator/counting.rs"]
mod counting;

use runmat_analysis_core::{AnalysisSequence, ProcedureStep, Temperature, ThermalDomain};
use runmat_analysis_execution::{
    execute_thermal_solver_result_objects, projected_transport_output_resource_facts,
};
use runmat_analysis_fea::{
    execution::{ThermalSolverResultManifest, ThermalWorkload},
    fixtures::{fixture_model, FixtureId},
    ComputeBackend, ThermalProcedureControl,
};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload(samples: usize) -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps(
        "thermal",
        vec![ProcedureStep::thermal_transient("insulated")],
    );
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
fn complete_thermal_execution_fits_composed_requested_heap_reservation() {
    const CHUNK_BYTES: usize = 64 * 1024;
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for samples in [8, 128] {
            let workload = workload(samples);
            let mesh = canonical_tetrahedron_solver_mesh(order);
            let preparation = workload.prepare(&mesh).unwrap();
            let transport = preparation.transport_profile(CHUNK_BYTES).unwrap();
            let output = projected_transport_output_resource_facts(&transport).unwrap();
            let bound = preparation.numerical_reservation_bytes().unwrap().0
                + output.construction_resident_bytes_admission_budget.0;
            drop(preparation);

            // Only the caller's model/mesh remain outside this measured end-to-end growth.
            let baseline = counting::begin_measurement();
            let prepared = execute_thermal_solver_result_objects(
                &workload,
                &mesh,
                CHUNK_BYTES,
                ObjectNamespace::ResultValue,
                ObjectInventoryLimits::default(),
            );
            let growth = counting::end_measurement().saturating_sub(baseline);
            let prepared = prepared.unwrap();
            assert!(
                growth <= bound,
                "{order:?} samples={samples} peak={growth} reservation={bound}"
            );
            prepared.manifest.validate().unwrap();
            assert_eq!(prepared.root, prepared.objects[0].descriptor);
            assert_eq!(prepared.objects.len(), prepared.manifest.chunks.len() + 1);
            assert_eq!(
                prepared.objects[0].bytes,
                prepared.manifest.canonical_bytes().unwrap()
            );
            for object in &prepared.objects {
                object.validate().unwrap();
            }
            let closure_bytes = prepared
                .objects
                .iter()
                .map(|object| object.bytes.len() as u64)
                .sum::<u64>();
            assert!(closure_bytes <= transport.total_bytes_upper_bound);
            let header = prepared
                .partition_payload(prepared.manifest.logical_identity)
                .unwrap();
            let manifest = ThermalSolverResultManifest::from_canonical_bytes(&header).unwrap();
            assert_eq!(manifest.fields.len(), samples);
            println!(
                "{order:?} samples={samples} peak_growth={growth} reserved={bound} closure_bytes={closure_bytes}"
            );
        }
    }
}
