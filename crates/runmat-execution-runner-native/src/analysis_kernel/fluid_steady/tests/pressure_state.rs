//! Full native output, receipt and portable cold-state authority at a large gauge.
use super::*;

// witness: FEA-FLUID-EQUATIONS-001/pressure_state @
//   incompressible_laminar_steady/canonical_tetrahedra/native_process;
//   boundary: Native result/receipt preserves original governing fields under gauge shift; not
//   distributed algebra or producer-death transient evidence.
#[test]
fn native_pressure_projection_preserves_original_governing_state() {
    let mut baseline = None;
    for reference in [0.0, 101325.0] {
        let (workload, mesh, task) = task_with_pressure(true, 1024, reference);
        let outputs = invoke(&task).unwrap();
        let mut outputs = outputs.into_iter();
        let original = outputs
            .next()
            .unwrap()
            .into_prepared(1024, task.output_policy.inventory_limits())
            .unwrap();
        let bytes = original.payload();
        let result = FluidSteadySolverResult::from_canonical_bytes_for(
            &bytes,
            Digest::sha256(&bytes),
            &workload,
            &mesh,
        )
        .unwrap();
        result.recover_original(&workload, &mesh).unwrap();
        let receipt = outputs
            .next()
            .unwrap()
            .into_prepared(1024, task.output_policy.inventory_limits())
            .unwrap();
        let receipt =
            FluidNumericalReceiptArtifact::from_canonical_bytes(&receipt.payload()).unwrap();
        receipt.validate_against(&workload, &result, &mesh).unwrap();
        let original = result.original();
        assert_eq!(
            original.pressure_pa.canonical_values().len(),
            original.pressure_fluctuation_pa.canonical_values().len()
        );
        for (absolute, fluctuation) in original
            .pressure_pa
            .canonical_values()
            .iter()
            .zip(original.pressure_fluctuation_pa.canonical_values())
        {
            assert_eq!(absolute.to_bits(), (reference + fluctuation).to_bits());
        }
        let fields = (
            original.velocity_m_per_s.clone(),
            original.pressure_fluctuation_pa.clone(),
            original.face_volume_flux_m3_per_s.clone(),
        );
        if let Some(baseline) = &baseline {
            assert_eq!(&fields, baseline);
        } else {
            baseline = Some(fields);
        }
    }
}
