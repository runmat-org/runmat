use super::*;
use crate::model::validation_storage::test_control::Budget;
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The finite healing report prepares borrowed revision membership once per validation;
//   repeated calls retain no scratch and exact/short budgets and deadline interruption preserve
//   cleanup. This is not independent geometric healing certification or a whole-model performance
//   measurement.
#[test]
fn healing_prepares_one_borrowed_revision_index_and_releases_it() {
    let report = report();
    let control = Budget::new(1 << 20);
    report.validate_with_control(&control).unwrap();
    let peak = control.peak();
    let probes = control.probes();
    assert!(peak > 0);
    assert_eq!(control.live(), 0);
    for _ in 0..32 {
        report.validate_with_control(&control).unwrap();
        assert_eq!(control.live(), 0);
        assert_eq!(control.peak(), peak);
    }
    report.validate_with_control(&Budget::new(peak)).unwrap();
    let short = Budget::new(peak - 1);
    assert!(report.validate_with_control(&short).is_err());
    assert_eq!(short.live(), 0);
    let stopped = Budget::with_stop(1 << 20, probes / 2);
    assert!(report.validate_with_control(&stopped).is_err());
    assert_eq!(stopped.live(), 0);
}
