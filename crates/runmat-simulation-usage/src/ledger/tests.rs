use super::*;

fn ops(value: u64) -> OperationCount {
    OperationCount::from_operations(value)
}

#[test]
fn reservations_enforce_grant_and_cap_before_work_and_charge_completed_prefixes() {
    let ledger = UsageLedger::new(ops(10), ops(4)).unwrap();
    let mut work = ledger.reserve(UsageCategory::Meshing, ops(3)).unwrap();
    assert!(
        matches!(ledger.reserve(UsageCategory::Solve, ops(2)), Err(LedgerError::MoreGrantRequired { required, granted }) if required == ops(5) && granted == ops(4))
    );
    work.complete(ops(1)).unwrap();
    assert_eq!(ledger.snapshot().unwrap().completed.meshing, ops(1));
    assert_eq!(ledger.snapshot().unwrap().reserved, ops(2));
    assert_eq!(work.complete(ops(3)), Err(LedgerError::InvalidReservation));
    drop(work);
    assert_eq!(ledger.snapshot().unwrap().reserved, ops(0));
    ledger.update_grant(ops(10)).unwrap();
    ledger.update_grant(ops(10)).unwrap();
    assert_eq!(ledger.update_grant(ops(11)), Err(LedgerError::InvalidGrant));
    assert_eq!(ledger.update_grant(ops(9)), Err(LedgerError::InvalidGrant));
    assert!(matches!(
        ledger.reserve(UsageCategory::Solve, ops(10)),
        Err(LedgerError::BudgetExceeded { .. })
    ));
    assert_eq!(ledger.finish().unwrap().total().unwrap(), ops(1));
}

#[test]
fn finalization_waits_for_all_owners_and_closes_admission() {
    let ledger = UsageLedger::new(ops(1), ops(1)).unwrap();
    let work = ledger.reserve(UsageCategory::Other, ops(0)).unwrap();
    assert_eq!(ledger.finish(), Err(LedgerError::OutstandingWork));
    drop(work);
    let final_usage = ledger.finish().unwrap();
    assert_eq!(ledger.finish().unwrap(), final_usage);
    assert!(matches!(
        ledger.reserve(UsageCategory::Other, ops(0)),
        Err(LedgerError::Closed)
    ));
    assert_eq!(ledger.update_grant(ops(1)), Err(LedgerError::Closed));
}

#[test]
fn parallel_batches_share_one_exact_grant_without_per_operation_locks() {
    let ledger = UsageLedger::new(ops(16), ops(16)).unwrap();
    std::thread::scope(|scope| {
        let mut first = ledger.reserve(UsageCategory::Assembly, ops(8)).unwrap();
        let mut second = ledger
            .reserve(UsageCategory::Factorization, ops(8))
            .unwrap();
        assert!(matches!(
            ledger.reserve(UsageCategory::Solve, ops(1)),
            Err(LedgerError::BudgetExceeded { .. })
        ));
        scope.spawn(move || first.complete(ops(8)).unwrap());
        scope.spawn(move || second.complete(ops(8)).unwrap());
    });
    let usage = ledger.finish().unwrap();
    assert_eq!(usage.assembly, ops(8));
    assert_eq!(usage.factorization, ops(8));
    assert_eq!(usage.total().unwrap(), ops(16));
    let wire = serde_json::to_value(usage).unwrap();
    assert_eq!(wire["assembly"], "8");
    assert!(serde_json::from_str::<CompletedUsage>(r#"{"meshing":"0"}"#).is_err());
}

#[test]
fn maximum_count_never_wraps_and_poisoned_owners_cannot_publish() {
    let ledger = UsageLedger::new(ops(u64::MAX), ops(u64::MAX)).unwrap();
    let mut work = ledger.reserve(UsageCategory::Solve, ops(u64::MAX)).unwrap();
    work.complete(ops(u64::MAX)).unwrap();
    assert!(matches!(
        ledger.reserve(UsageCategory::Solve, ops(1)),
        Err(LedgerError::CountOverflow)
    ));
    drop(work);
    assert_eq!(ledger.finish().unwrap().total().unwrap(), ops(u64::MAX));

    let ledger = UsageLedger::new(ops(1), ops(1)).unwrap();
    let work = ledger.reserve(UsageCategory::Solve, ops(1)).unwrap();
    let _panic = std::panic::catch_unwind(|| {
        let _state = ledger.state.lock().unwrap();
        panic!("interrupted ledger owner");
    });
    drop(work);
    assert_eq!(ledger.snapshot(), Err(LedgerError::InvalidState));
    assert_eq!(ledger.finish(), Err(LedgerError::InvalidState));
}
