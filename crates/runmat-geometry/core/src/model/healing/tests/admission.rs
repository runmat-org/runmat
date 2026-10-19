use super::*;
use crate::model::validation_storage::test_control::Budget;
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
