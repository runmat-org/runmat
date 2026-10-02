use super::*;
use futures::task::noop_waker;
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

struct PendingCharge(Rc<Cell<bool>>);

impl Future for PendingCharge {
    type Output = BuiltinResult<()>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingCharge {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

struct ArrayPolynomial {
    dropped: Rc<Cell<bool>>,
    charged: Rc<Cell<u64>>,
}

impl QuadratureEvaluator for ArrayPolynomial {
    type Sample = Arc<Vec<f64>>;

    async fn checkpoint(&mut self) -> BuiltinResult<()> {
        Ok(())
    }

    async fn charge(&mut self, units: u64) -> BuiltinResult<()> {
        self.charged.set(self.charged.get() + units);
        PendingCharge(self.dropped.clone()).await
    }

    async fn evaluate(&mut self, x: f64) -> BuiltinResult<Self::Sample> {
        Ok(Arc::new(vec![x * x; 5000]))
    }

    fn midpoint(&self, a: f64, b: f64) -> f64 {
        super::super::midpoint(a, b)
    }

    fn tolerances(&self) -> (f64, f64) {
        (1.0e-6, 0.0)
    }

    fn max_evaluations(&self) -> usize {
        5
    }

    fn max_depth(&self) -> usize {
        30
    }

    fn evaluation_limit_error(&self) -> RuntimeError {
        crate::build_runtime_error("evaluation limit").build()
    }

    fn convergence_error(&self) -> RuntimeError {
        crate::build_runtime_error("convergence failure").build()
    }

    fn nonrepresentable_error(&self) -> RuntimeError {
        crate::build_runtime_error("nonrepresentable result").build()
    }

    fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError {
        crate::build_runtime_error(format!("array arithmetic: {error:?}")).build()
    }
}

#[test]
fn large_array_algebra_yields_mid_component_and_abandonment_drops_turn() {
    let dropped = Rc::new(Cell::new(false));
    let charged = Rc::new(Cell::new(0));
    let mut evaluator = ArrayPolynomial {
        dropped: dropped.clone(),
        charged: charged.clone(),
    };
    let mut invocation = Box::pin(integrate_array(&mut evaluator, 0.0, 1.0));
    let waker = noop_waker();
    assert!(invocation
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(charged.get(), 4096);
    assert!(!dropped.get());
    drop(invocation);
    assert!(dropped.get());
}
