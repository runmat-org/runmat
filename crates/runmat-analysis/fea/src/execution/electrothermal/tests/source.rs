use super::*;
use runmat_meshing_core::ElementOrder;

// witness: FEA-COUPLING-ELECTROTHERMAL-001/cold @ electrothermal/tet4|tet10/public_study;
//   boundary: Both-order retained source/target binding and mutated density rejection in one
//   process; no native termination/restart or remote transfer.
#[test]
fn retained_source_cold_binding_authenticates_original_result_and_target() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order, 2);
        let electrical = electrical_workload(2.0, 0.0);
        let result = electrical.execute(&mesh).unwrap();
        let expected = result.source_result_identity().unwrap();
        let source =
            RetainedJouleHeatSource::new(electrical, result, expected, thermal(), &mesh).unwrap();
        let bound = source.bind(&mesh).unwrap();
        let identity = source.identity().unwrap();
        let bytes = source.canonical_bytes().unwrap();
        assert!(source.validation_workspace_bytes(&mesh).unwrap() > bytes.len() as u64);
        drop(source);
        let cold = RetainedJouleHeatSource::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(cold.identity().unwrap(), identity);
        cold.validate_for(&bound, &mesh).unwrap();
        assert!(cold.validate_for(&thermal(), &mesh).is_err());
        assert!((cold.transfer().power_w() - 12.0).abs() < 1e-8);
        let mut wire = serde_json::to_value(&cold).unwrap();
        wire["transfer"]["density_w_per_m3"][0] = serde_json::json!(13.0);
        let changed: RetainedJouleHeatSource = serde_json::from_value(wire).unwrap();
        assert_ne!(changed.identity().unwrap(), identity);
        assert!(changed.validate_for(&bound, &mesh).is_err());
    }
}
