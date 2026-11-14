use super::*;

const GIB: u64 = 1 << 30;

#[test]
fn instance_budget_is_the_address_space_after_the_module_within_the_device() {
    let platform = BrowserMemoryPlatform {
        module_baseline_bytes: 3 << 20,
        device_memory_bytes: None,
    };
    assert_eq!(platform.instance_budget_bytes(), 4 * GIB - (3 << 20));
    let small_device = BrowserMemoryPlatform {
        device_memory_bytes: Some(2 * GIB),
        ..platform
    };
    assert_eq!(small_device.instance_budget_bytes(), 2 * GIB);
    let large_device = BrowserMemoryPlatform {
        device_memory_bytes: Some(8 * GIB),
        ..platform
    };
    assert_eq!(large_device.instance_budget_bytes(), 4 * GIB - (3 << 20));
}

#[test]
fn free_heap_counts_live_bytes_not_grown_memory() {
    let platform = BrowserMemoryPlatform {
        module_baseline_bytes: 0,
        device_memory_bytes: None,
    };
    // A session that once grew to 3 GiB but holds 200 MiB live has 3.8 GiB free.
    assert_eq!(platform.free_bytes(200 << 20), 4 * GIB - (200 << 20));
    assert_eq!(platform.free_bytes(u64::MAX), 0);
}

#[test]
fn solve_admission_names_memory_then_scratch_with_numbers() {
    assert_eq!(admit_solve_bytes(700, 10, 300, 1000), Ok(()));
    let memory = admit_solve_bytes(701, 10, 300, 1000).unwrap_err();
    assert_eq!(
        memory,
        BrowserResourceLimit {
            term: BrowserResourceTerm::SolveMemory,
            required_bytes: Some(1001),
            maximum_bytes: 1000,
        }
    );
    let scratch = admit_solve_bytes(10, 1001, 0, 1000).unwrap_err();
    assert_eq!(scratch.term, BrowserResourceTerm::SolveScratch);
    assert_eq!(scratch.required_bytes, Some(1001));
    assert_eq!(scratch.maximum_bytes, 1000);
}

#[test]
fn worker_request_must_fit_the_granted_allowance_and_the_worker() {
    assert_eq!(admit_worker_request(10, 100, 1000), Ok(()));
    let allowance = admit_worker_request(10, 1001, 1000).unwrap_err();
    assert_eq!(
        (
            allowance.term,
            allowance.required_bytes,
            allowance.maximum_bytes
        ),
        (BrowserResourceTerm::WorkerAllowance, Some(1001), 1000)
    );
    let request = admit_worker_request(101, 100, 1000).unwrap_err();
    assert_eq!(
        (request.required_bytes, request.maximum_bytes),
        (Some(101), 100)
    );
}

#[test]
fn planner_ceiling_refusal_reports_the_smallest_path_and_its_term() {
    let resident =
        BrowserResourceLimit::from_planning(&ResourcePlanningError::StaticSolveCeiling {
            factorization_resident_bytes: Some(9_000),
            multigrid_resident_bytes: 5_000,
            multigrid_scratch_bytes: Some(100),
            maximum: 4_000,
        })
        .unwrap();
    assert_eq!(
        resident,
        BrowserResourceLimit {
            term: BrowserResourceTerm::SolveMemory,
            required_bytes: Some(5_000),
            maximum_bytes: 4_000,
        }
    );
    let scratch = BrowserResourceLimit::from_planning(&ResourcePlanningError::StaticSolveCeiling {
        factorization_resident_bytes: None,
        multigrid_resident_bytes: 3_000,
        multigrid_scratch_bytes: Some(4_500),
        maximum: 4_000,
    })
    .unwrap();
    assert_eq!(
        (scratch.term, scratch.required_bytes),
        (BrowserResourceTerm::SolveScratch, Some(4_500))
    );
    let budget = BrowserResourceLimit::from_planning(&ResourcePlanningError::MemoryBudget {
        required_memory_bytes: 10,
        available_memory_bytes: 20,
        required_scratch_bytes: 30,
        available_scratch_bytes: 25,
    })
    .unwrap();
    assert_eq!(
        (budget.term, budget.required_bytes, budget.maximum_bytes),
        (BrowserResourceTerm::SolveScratch, Some(30), 25)
    );
    assert_eq!(
        BrowserResourceLimit::from_planning(&ResourcePlanningError::Overflow("x")),
        None
    );
}

#[test]
fn diagnostic_carries_code_term_numbers_and_next_action() {
    let limit = BrowserResourceLimit {
        term: BrowserResourceTerm::PreparedRequest,
        required_bytes: Some(5_368_709_120),
        maximum_bytes: 4_290_000_000,
    };
    let diagnostic = limit.diagnostic();
    assert_eq!(diagnostic["identifier"], BROWSER_RESOURCE_LIMIT_CODE);
    assert_eq!(diagnostic["kind"], "runtime");
    assert_eq!(diagnostic["resourceLimit"]["term"], "prepared_request");
    assert_eq!(
        diagnostic["resourceLimit"]["requiredBytes"],
        5_368_709_120_u64
    );
    assert_eq!(
        diagnostic["resourceLimit"]["maximumBytes"],
        4_290_000_000_u64
    );
    assert_eq!(
        diagnostic["resourceLimit"]["nextAction"],
        "Continue in the desktop app."
    );
    assert_eq!(
        diagnostic["message"],
        "The prepared simulation requires 5368709120 bytes of browser memory, maximum 4290000000 bytes. Continue in the desktop app."
    );
    for term in [
        BrowserResourceTerm::SolveMemory,
        BrowserResourceTerm::SolveScratch,
        BrowserResourceTerm::NumericalResult,
        BrowserResourceTerm::WorkerAllowance,
    ] {
        let value = BrowserResourceLimit {
            term,
            required_bytes: Some(2),
            maximum_bytes: 1,
        }
        .diagnostic();
        assert_eq!(value["identifier"], BROWSER_RESOURCE_LIMIT_CODE);
        assert!(value["message"]
            .as_str()
            .unwrap()
            .contains("requires 2 bytes"));
    }
    let unsized_limit = BrowserResourceLimit {
        term: BrowserResourceTerm::SolveScratch,
        required_bytes: None,
        maximum_bytes: 1,
    };
    assert!(unsized_limit.diagnostic()["resourceLimit"]["requiredBytes"].is_null());
}
