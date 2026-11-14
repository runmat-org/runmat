use super::*;
use crate::context::{RuntimeContext, RuntimeServicePorts};
use std::{cell::Cell, rc::Rc};

struct DenyingHost(Rc<Cell<u32>>);
impl AsyncFeaRunAdmission for DenyingHost {
    fn request_grant(
        &self,
        _: FeaRunRequest,
        _: FeaRunUsage,
        _: runmat_simulation_usage::OperationCount,
    ) -> RuntimeServiceFuture<Result<FeaRunBudget, FeaRunDenied>> {
        Box::pin(async { Err(FeaRunDenied::SignInRequired) })
    }
    fn admit(
        &self,
        request: FeaRunRequest,
    ) -> RuntimeServiceFuture<Result<Box<dyn AsyncFeaRunLease>, FeaRunDenied>> {
        assert_eq!(request.site, super::super::FeaRunSite::Local);
        assert_eq!(request.kind, FeaRunKind::Solve);
        assert_eq!(
            request.study_identity,
            runmat_execution::Digest::sha256(b"exact-domain-root").to_string()
        );
        self.0.set(self.0.get() + 1);
        Box::pin(async { Err(FeaRunDenied::SignInRequired) })
    }
}

#[test]
fn async_domain_admission_denies_before_polling_any_mesh_or_solver_effect() {
    let admitted = Rc::new(Cell::new(0));
    let effects = Cell::new(0);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_fea_run_admission(Rc::new(DenyingHost(admitted.clone()))),
        );
    let _entered = runtime.enter();
    let result = futures::executor::block_on(run(
        FeaRunKind::Solve,
        runmat_execution::Digest::sha256(b"exact-domain-root"),
        async {
            effects.set(effects.get() + 1);
            Ok::<_, FeaRunExecutionError>(42)
        },
        |failure| failure,
        |result, lease_error| result.and(Err(lease_error)),
    ));
    assert!(matches!(
        result,
        Err(FeaRunExecutionError::Denied(FeaRunDenied::SignInRequired))
    ));
    assert_eq!(admitted.get(), 1);
    assert_eq!(effects.get(), 0);
}

#[test]
fn no_async_host_keeps_the_existing_domain_gate_and_does_not_precharge_scripts() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let _entered = runtime.enter();
    let result = futures::executor::block_on(run(
        FeaRunKind::Solve,
        runmat_execution::Digest::sha256(b"domain"),
        async { Ok::<_, FeaRunExecutionError>(42) },
        |failure| failure,
        |result, lease_error| result.and(Err(lease_error)),
    ));
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn numerical_capacity_renews_without_charging_or_overlapping_primitive_work() {
    use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
    use std::cell::RefCell;

    fn ops(value: u64) -> OperationCount {
        OperationCount::from_operations(value)
    }
    struct CapacityHost {
        budget: FeaRunBudget,
        requests: Rc<RefCell<Vec<(OperationCount, OperationCount)>>>,
    }
    impl AsyncFeaRunAdmission for CapacityHost {
        fn request_grant(
            &self,
            _: FeaRunRequest,
            usage: FeaRunUsage,
            required: OperationCount,
        ) -> RuntimeServiceFuture<Result<FeaRunBudget, FeaRunDenied>> {
            self.requests
                .borrow_mut()
                .push((usage.completed_ops.total().unwrap(), required));
            let renewed = FeaRunBudget {
                granted_ops: required,
                ..self.budget
            };
            Box::pin(async move { Ok(renewed) })
        }
        fn admit(
            &self,
            _: FeaRunRequest,
        ) -> RuntimeServiceFuture<Result<Box<dyn AsyncFeaRunLease>, FeaRunDenied>> {
            Box::pin(async { Err(FeaRunDenied::LicenseInvalid) })
        }
    }

    let budget = FeaRunBudget {
        budget_ops: ops(20),
        granted_ops: ops(0),
        expires_at_unix_millis: runmat_time::unix_timestamp_ms() as u64 + 60_000,
        heartbeat_millis: 30_000,
    };
    let requests = Rc::new(RefCell::new(Vec::new()));
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_fea_run_admission(Rc::new(CapacityHost {
                budget,
                requests: Rc::clone(&requests),
            })),
        );
    let request = FeaRunRequest::local(
        FeaRunKind::Solve,
        1000,
        runmat_execution::Digest::sha256(b"numerical-capacity"),
    )
    .unwrap();
    let child = FeaRunScope::from_parent_supervised_delegation(request, budget, 0).unwrap();
    futures::executor::block_on(child.run_future(async {
        let usage = UsageContext::capture().unwrap();
        ensure_available_capacity(&usage, ops(7), Some(&runtime))
            .await
            .unwrap();
        let before = usage.ledger().snapshot().unwrap();
        assert_eq!(before.completed.total().unwrap(), ops(0));
        assert_eq!(before.reserved, ops(0));
        let mut work = usage
            .ledger()
            .reserve(UsageCategory::Solve, ops(5))
            .unwrap();
        assert_eq!(
            ensure_available_capacity(&usage, ops(1), Some(&runtime)).await,
            Err(FeaRunDenied::Accounting(LedgerError::OutstandingWork))
        );
        work.complete(ops(5)).unwrap();
        drop(work);
        ensure_available_capacity(&usage, ops(5), Some(&runtime))
            .await
            .unwrap();
        assert_eq!(usage.ledger().snapshot().unwrap().completed.solve, ops(5));
        assert!(matches!(
            ensure_available_capacity(&usage, ops(16), Some(&runtime)).await,
            Err(FeaRunDenied::Accounting(LedgerError::BudgetExceeded { .. }))
        ));
    }));
    assert_eq!(
        *requests.borrow(),
        vec![(ops(0), ops(7)), (ops(5), ops(10))]
    );
}

/// The browser numerical worker's capacity port: `ensure_available_capacity` on a
/// parent-supervised child, granting what each request requires.
mod awaited_assembly {
    use super::*;
    use runmat_simulation_usage::{
        AsyncCapacityOwner, LedgerError, OperationCount, UsageCategory, UsageContext,
    };

    struct GrantingHost(FeaRunBudget);
    impl AsyncFeaRunAdmission for GrantingHost {
        fn request_grant(
            &self,
            _: FeaRunRequest,
            _: FeaRunUsage,
            required: OperationCount,
        ) -> RuntimeServiceFuture<Result<FeaRunBudget, FeaRunDenied>> {
            let renewed = FeaRunBudget {
                granted_ops: required,
                ..self.0
            };
            Box::pin(async move { Ok(renewed) })
        }
        fn admit(
            &self,
            _: FeaRunRequest,
        ) -> RuntimeServiceFuture<Result<Box<dyn AsyncFeaRunLease>, FeaRunDenied>> {
            Box::pin(async { Err(FeaRunDenied::LicenseInvalid) })
        }
    }

    /// Mirrors runmat-wasm `BrowserNumericalCapacity`.
    struct WorkerCapacity<'a>(&'a RuntimeContext);
    impl AsyncCapacityOwner for WorkerCapacity<'_> {
        fn ensure_capacity(
            &self,
            additional: OperationCount,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), LedgerError>> + '_>>
        {
            Box::pin(async move {
                let usage = UsageContext::capture().ok_or(LedgerError::GrantUnavailable)?;
                ensure_available_capacity(&usage, additional, Some(self.0))
                    .await
                    .map_err(|denied| match denied {
                        FeaRunDenied::Accounting(cause) => cause,
                        _ => LedgerError::GrantUnavailable,
                    })
            })
        }
    }

    fn straight() -> [[f64; 3]; 10] {
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut nodes = [[0.0; 3]; 10];
        nodes[..4].copy_from_slice(&corners);
        for (edge, [left, right]) in runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS
            .into_iter()
            .enumerate()
        {
            nodes[4 + edge] =
                std::array::from_fn(|axis| 0.5 * (corners[left][axis] + corners[right][axis]));
        }
        nodes
    }

    type StepResult = Result<
        runmat_meshing_core::Tet10StiffnessMoments,
        runmat_meshing_core::Tet10CapacityError<LedgerError>,
    >;

    /// One awaited Tet10 element in a fresh browser-style child; returns the step's result and
    /// the usage the worker's final report would carry.
    fn run_element(nodes: [[f64; 3]; 10]) -> (StepResult, FeaRunUsage) {
        let budget = FeaRunBudget {
            budget_ops: OperationCount::from_operations(1 << 40),
            granted_ops: OperationCount::ZERO,
            expires_at_unix_millis: runmat_time::unix_timestamp_ms() as u64 + 60_000,
            heartbeat_millis: 30_000,
        };
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default()
                        .with_fea_run_admission(Rc::new(GrantingHost(budget))),
                );
        let request = FeaRunRequest::local(
            FeaRunKind::Solve,
            1000,
            runmat_execution::Digest::sha256(b"awaited-tet10-assembly"),
        )
        .unwrap();
        let child = FeaRunScope::from_parent_supervised_delegation(request, budget, 0).unwrap();
        let capacity = WorkerCapacity(&runtime);
        let result = futures::executor::block_on(child.run_future(
            runmat_meshing_core::tetrahedron10_stiffness_moments_with_capacity(
                &nodes,
                UsageCategory::Assembly,
                Some(&capacity),
            ),
        ));
        // A capacity refusal denies the whole child; a domain failure leaves it admitted.
        let admitted = child.check();
        let control = child.control();
        let outcome = if result.is_ok() {
            FeaRunOutcome::Completed
        } else {
            FeaRunOutcome::Failed
        };
        child.settle(outcome).unwrap();
        assert!(admitted.is_ok(), "{admitted:?}");
        (result, control.settled_usage().unwrap())
    }

    #[test]
    fn a_curved_quadratic_element_assembles_through_the_browser_capacity_port() {
        let mut curved = straight();
        curved[4][1] -= 0.05;
        let (result, usage) = run_element(curved);
        let moments = match result {
            Ok(moments) => moments,
            Err(runmat_meshing_core::Tet10CapacityError::Admission(cause)) => {
                panic!("capacity refused: {cause:?}")
            }
            Err(runmat_meshing_core::Tet10CapacityError::Invalidity(error)) => {
                panic!("{error:?}")
            }
        };
        assert_ne!(
            moments.rule,
            runmat_meshing_core::Tet10StiffnessRule::Affine
        );
        assert!(usage.completed_ops.assembly > OperationCount::ZERO);
    }

    #[test]
    fn a_step_that_fails_mid_assembly_reports_its_work_and_its_own_error() {
        let mut rejected = straight();
        rejected[4][0] += 0.24;
        let (result, usage) = run_element(rejected);
        // The surfaced error is the element's, not an accounting refusal.
        assert!(matches!(
            result,
            Err(runmat_meshing_core::Tet10CapacityError::Invalidity(
                runmat_meshing_core::Tet10ElementInvalidity::InsufficientJacobianMargin
                    | runmat_meshing_core::Tet10ElementInvalidity::InvertedOrDegenerate
            ))
        ));
        // The final usage is certifiable after the failure and carries the completed prefix.
        assert!(usage.completed_ops.assembly > OperationCount::ZERO);
    }
}
