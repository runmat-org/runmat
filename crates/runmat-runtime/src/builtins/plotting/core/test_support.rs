use super::state;

#[doc(hidden)]
pub use state::PlotTestLockGuard;

#[doc(hidden)]
pub fn lock_plot_test_context() -> PlotTestLockGuard {
    state::lock_plot_test_registry()
}

#[cfg(test)]
pub(crate) fn ensure_plot_test_env() {
    state::disable_rendering_for_tests();
}

#[cfg(test)]
pub(crate) fn lock_plot_registry() -> state::PlotTestLockGuard {
    state::lock_plot_test_registry()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::lock_plot_test_context;
    use crate::builtins::common::test_support::global_state_test_guard;
    use std::sync::{mpsc, Arc, Barrier};
    use std::time::Duration;

    #[test]
    fn plot_and_global_fixtures_admit_both_nesting_orders() {
        let start = Arc::new(Barrier::new(2));
        let (done, completion) = mpsc::channel();
        let workers: Vec<_> = [false, true]
            .into_iter()
            .map(|global_first| {
                let start = Arc::clone(&start);
                let done = done.clone();
                std::thread::spawn(move || {
                    start.wait();
                    for _ in 0..32 {
                        if global_first {
                            let _global = global_state_test_guard();
                            let _plot = lock_plot_test_context();
                        } else {
                            let _plot = lock_plot_test_context();
                            let _global = global_state_test_guard();
                        }
                    }
                    done.send(()).unwrap();
                })
            })
            .collect();
        for _ in 0..2 {
            completion
                .recv_timeout(Duration::from_secs(10))
                .expect("plot/global fixture nesting must not deadlock");
        }
        for worker in workers {
            worker.join().unwrap();
        }
    }
}
