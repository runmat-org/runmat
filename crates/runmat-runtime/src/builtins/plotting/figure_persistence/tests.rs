use super::*;
use crate::builtins::plotting::figure::figure_builtin;
use crate::builtins::plotting::plot::plot_builtin;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use crate::builtins::plotting::{
    clear_figure, close_figure, figure_handles, reset_hold_state_for_run, reset_plot_state,
};
use futures::executor::block_on;

fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
    let guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    guard
}

fn tensor(data: &[f64]) -> Tensor {
    Tensor::new(data.to_vec(), vec![data.len()]).expect("persistence test vector")
}

fn temp_path(stem: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_figure_persistence_{stem}_{}.fig",
        std::process::id()
    ));
    path
}

mod contract;
mod archive_validation;
mod filesystem;
mod publication;
mod roundtrip;
mod save_validation;
mod selected_binding;
mod staging_fixture;
mod validation;
mod load_validation;
