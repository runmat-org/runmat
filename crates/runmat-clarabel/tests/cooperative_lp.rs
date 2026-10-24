//! RunMat extension controls; upstream numerical tests remain unchanged.
use clarabel::{algebra::*, solver::*};
use std::future::Future;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll};

fn problem() -> DefaultSolver<f64> {
    let p = CscMatrix::zeros((2, 2));
    let a = CscMatrix::from(&[[1., 0.], [0., 2.], [3., 2.], [-1., 0.], [0., -1.]]);
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .build()
        .unwrap();
    DefaultSolver::new(
        &p,
        &[-3., -5.],
        &a,
        &[4., 12., 18., 0., 0.],
        &[NonnegativeConeT(5)],
        settings,
    )
    .unwrap()
}

async fn parity() {
    let mut synchronous = problem();
    synchronous.solve();
    let mut turns = 0usize;
    let cooperative = problem()
        .solve_with_progress(&mut turns, async |turns, _info| {
            *turns += 1;
            let mut first = true;
            futures::future::poll_fn(|context| {
                if first {
                    first = false;
                    context.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            })
            .await;
            Ok::<_, ()>(())
        })
        .await
        .unwrap();
    assert_eq!(synchronous.solution.status, SolverStatus::Solved);
    assert_eq!(cooperative.solution.status, synchronous.solution.status);
    assert_eq!(
        cooperative.solution.iterations,
        synchronous.solution.iterations
    );
    assert_eq!(cooperative.solution.x, synchronous.solution.x);
    assert_eq!(cooperative.solution.z, synchronous.solution.z);
    assert_eq!(cooperative.solution.s, synchronous.solution.s);
    assert_eq!(cooperative.solution.obj_val, synchronous.solution.obj_val);
    assert!(turns > 0);
}

#[test]
fn synchronous_and_actually_pending_cooperative_numerics_match() {
    futures::executor::block_on(parity());
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_cooperative_numerical_parity() {
    parity().await;
}

struct DropProbe(Arc<AtomicUsize>);
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn typed_time_limit_uses_normal_finalization_and_retains_owned_drop() {
    let drops = Arc::new(AtomicUsize::new(0));
    let probe = DropProbe(drops.clone());
    let mut solver = problem();
    solver.set_termination_callback(move |_| {
        std::hint::black_box(&probe);
        false
    });
    let mut turns = 0usize;
    let solver =
        futures::executor::block_on(solver.solve_with_progress(&mut turns, async |turns, _| {
            *turns += 1;
            Ok::<_, u32>(ProgressControl::Stop(ProgressStop::TimeLimit))
        }))
        .unwrap();
    assert_eq!(turns, 1);
    assert_eq!(solver.info.status, SolverStatus::MaxTime);
    assert_eq!(solver.solution.status, SolverStatus::MaxTime);
    assert_eq!(solver.solution.iterations, 0);
    assert_eq!(solver.solution.x.len(), 2);
    assert_eq!(solver.solution.z.len(), 5);
    assert!(
        solver.timers.is_some(),
        "normal finalization restores solver timers"
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(solver);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn cancelling_owned_progress_drops_solver_once_and_progress_errors_are_unchanged() {
    let drops = Arc::new(AtomicUsize::new(0));
    let probe = DropProbe(drops.clone());
    let mut solver = problem();
    solver.set_termination_callback(move |_| {
        std::hint::black_box(&probe);
        false
    });
    let mut context = ();
    let mut pending = Box::pin(solver.solve_with_progress(&mut context, async |_, _| {
        futures::future::pending::<Result<(), u32>>().await
    }));
    let waker = futures::task::noop_waker();
    assert!(pending
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(pending);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let error = futures::executor::block_on(
        problem().solve_with_progress(&mut context, async |_, _| Err::<(), _>(917u32)),
    );
    match error {
        Err(code) => assert_eq!(code, 917),
        Ok(_) => panic!("progress error was lost"),
    }
}
