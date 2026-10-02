use super::*;
use crate::builtins::plotting::line::line_builtin;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use futures::executor::block_on;

fn host_line() -> f64 {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("x"));
    let y = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![1, 2]).expect("y"));
    let Value::Num(handle) = block_on(line_builtin(vec![x, y])).expect("line") else {
        panic!("expected graphics handle");
    };
    handle
}

fn x_data(handle: f64) -> Value {
    let plot = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
    block_on(get_properties_async(plot, Some("XData"), "get")).expect("XData")
}

#[path = "host.rs"]
mod host;
#[cfg(not(target_arch = "wasm32"))]
#[path = "native.rs"]
mod native;
#[cfg(target_arch = "wasm32")]
#[path = "browser.rs"]
mod browser;
