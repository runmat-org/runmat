//! A real host input service supplies the pending boundary for generated entries.

use super::*;
use futures::channel::oneshot;
use futures::task::noop_waker_ref;
use runmat_value::Value;
use std::cell::Cell;
use std::collections::VecDeque;
use std::future::Future;
use std::sync::Mutex;
use std::task::{Context, Poll};

type InputTurn = (
    Option<oneshot::Receiver<()>>,
    Result<crate::execution::InputResponse, String>,
);

thread_local! {
    static MEMBER_TARGET: Cell<Option<runmat_gc::GcHandle>> = const { Cell::new(None) };
    static MEMBER_READS: Cell<Option<f64>> = const { Cell::new(None) };
    static RHS_CALLS: Cell<Option<f64>> = const { Cell::new(None) };
}

pub(super) fn observed_member_target() -> runmat_gc::GcHandle {
    MEMBER_TARGET.with(|target| target.get().expect("member receiver witness"))
}

pub(super) fn observed_member_reads() -> Option<f64> {
    MEMBER_READS.with(Cell::get)
}

pub(super) fn observed_rhs_calls() -> Option<f64> {
    RHS_CALLS.with(Cell::get)
}

pub(super) fn install_input(
    session: &mut RunMatSession,
    calls: Arc<AtomicUsize>,
    turns: Vec<InputTurn>,
) {
    MEMBER_TARGET.with(|target| target.set(None));
    MEMBER_READS.with(|reads| reads.set(None));
    RHS_CALLS.with(|count| count.set(None));
    let turns = Arc::new(Mutex::new(VecDeque::from(turns)));
    session.install_async_input_handler(move |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        if let Some(Value::HandleObject(handle)) =
            runmat_runtime::workspace::session::global_value("pending_member_witness")
        {
            MEMBER_TARGET.with(|target| target.set(Some(handle.target)));
            runmat_runtime::workspace::session::store_global_named(
                "pending_member_witness",
                Value::Num(0.0),
            )
            .expect("replace the observed pending-member witness");
        }
        if let Some(Value::Num(reads)) =
            runmat_runtime::workspace::session::global_value("member_reads")
        {
            MEMBER_READS.with(|observed| observed.set(Some(reads)));
        }
        if let Some(Value::Num(count)) =
            runmat_runtime::workspace::session::global_value("rhs_calls")
        {
            RHS_CALLS.with(|observed| observed.set(Some(count)));
        }
        let (gate, result) = turns
            .lock()
            .expect("input turns")
            .pop_front()
            .expect("generated native site replayed an input");
        async move {
            if let Some(gate) = gate {
                gate.await.map_err(|_| "input gate abandoned".to_string())?;
            }
            result
        }
    });
}

pub(super) fn pending_once(
    mut call: std::pin::Pin<
        &mut impl Future<
            Output = Result<runmat_runtime::execution::RootedValueSequence, crate::RunError>,
        >,
    >,
    calls: &AtomicUsize,
) {
    for _ in 0..8 {
        let mut task = Context::from_waker(noop_waker_ref());
        if let Poll::Ready(outcome) = call.as_mut().poll(&mut task) {
            panic!("native entry completed before its service gate: {outcome:?}");
        }
        if calls.load(Ordering::SeqCst) == 1 {
            for _ in 0..2 {
                assert!(call.as_mut().poll(&mut task).is_pending());
                assert_eq!(
                    calls.load(Ordering::SeqCst),
                    1,
                    "pending native site replayed"
                );
            }
            return;
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1, "service was not reached");
}
