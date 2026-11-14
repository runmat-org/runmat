//! The browser's numerical worker awaits capacity through an owner that quotes only while no
//! batch is outstanding (`UsageLedger::exclusive_work_requirement`). Every awaited Tet10 path must
//! release its batches before the next request, or the first quadratic element of a browser
//! assembly fails with `OutstandingWork`.
use super::*;
use runmat_simulation_usage::{AsyncCapacityOwner, UsageLedger};
use std::{future::Future, pin::Pin, sync::Arc};

struct ExclusiveOwner<'a>(&'a UsageLedger);

impl AsyncCapacityOwner for ExclusiveOwner<'_> {
    fn ensure_capacity(
        &self,
        additional: OperationCount,
    ) -> Pin<Box<dyn Future<Output = Result<(), LedgerError>> + '_>> {
        let quoted = self.0.exclusive_work_requirement(additional).map(|_| ());
        Box::pin(async move { quoted })
    }
}

fn ready<R>(future: impl Future<Output = R>) -> R {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(result) => result,
        std::task::Poll::Pending => unreachable!("an immediate owner never suspends"),
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
    for (edge, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
        nodes[4 + edge] =
            std::array::from_fn(|axis| 0.5 * (corners[left][axis] + corners[right][axis]));
    }
    nodes
}

type Outcome = (
    Result<Tet10StiffnessMoments, Tet10ElementInvalidity>,
    runmat_simulation_usage::LedgerSnapshot,
);

fn with_owner(nodes: &[[f64; 3]; 10]) -> Outcome {
    let limit = OperationCount::from_operations(1 << 40);
    let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    let owner = ExclusiveOwner(&ledger);
    let result = context.run_sync(|| {
        ready(tetrahedron10_stiffness_moments_with_capacity(
            nodes,
            UsageCategory::Assembly,
            Some(&owner),
        ))
    });
    let result = result.map_err(|error| match error {
        Tet10CapacityError::Invalidity(error) => error,
        Tet10CapacityError::Admission(cause) => Tet10ElementInvalidity::Accounting(cause),
    });
    (result, ledger.snapshot().unwrap())
}

fn without_owner(nodes: &[[f64; 3]; 10]) -> Outcome {
    let limit = OperationCount::from_operations(1 << 40);
    let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    let result = context
        .run_sync(|| tetrahedron10_stiffness_moments_for_category(nodes, UsageCategory::Assembly));
    (result, ledger.snapshot().unwrap())
}

#[test]
fn awaited_tet10_moments_release_each_batch_before_the_next_capacity_request() {
    let mut mild = straight();
    mild[4][1] -= 0.05;
    let mut strong = straight();
    strong[4][0] += 0.24;
    let mut inverted = straight();
    inverted[3][2] = -1.0;
    // Affine, curved (margin bisection and tensor rules), rejected margin, and inverted.
    for nodes in [straight(), mild, strong, inverted] {
        let (owned, owned_usage) = with_owner(&nodes);
        let (reference, reference_usage) = without_owner(&nodes);
        assert!(
            !matches!(owned, Err(Tet10ElementInvalidity::Accounting(_))),
            "{owned:?}"
        );
        assert_eq!(owned, reference);
        // A failed element still completes and reports the same work as the synchronous rule.
        assert_eq!(owned_usage.completed, reference_usage.completed);
        assert!(owned_usage.completed.assembly > OperationCount::ZERO);
        assert_eq!(owned_usage.reserved, OperationCount::ZERO);
    }
}
