use super::*;
use crate::builtins::plotting::line::line_builtin;
use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
use futures::executor::block_on;

fn host_line() -> f64 {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("x"));
    let y = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![1, 2]).expect("y"));
    let Value::Num(handle) = block_on(line_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![x, y]),
    ))
    .expect("line") else {
        panic!("expected graphics handle");
    };
    handle
}

fn x_data(handle: f64) -> Value {
    let plot = resolve_plot_handle(&Value::Num(handle), "get").expect("handle");
    block_on(get_properties_async(plot, Some("XData"), "get")).expect("XData")
}

fn host_scatter() -> f64 {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("x"));
    let y = Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![1, 2]).expect("y"));
    block_on(crate::builtins::plotting::scatter::scatter_builtin(None, {
        let mut coordinates = vec![x, y];
        coordinates.extend(Vec::new());
        crate::builtins::common::test_support::builtin_arguments(coordinates)
    }))
    .expect("scatter")
}

fn host_line3() -> f64 {
    let inputs = [vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
        .into_iter()
        .map(|values| Value::Tensor(Tensor::new(values, vec![1, 2]).expect("coordinate")))
        .collect();
    block_on(crate::builtins::plotting::plot3::plot3_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(inputs),
    ))
    .expect("plot3")
}

#[path = "host.rs"]
mod host;
#[cfg(not(target_arch = "wasm32"))]
#[path = "native.rs"]
mod native;
#[cfg(not(target_arch = "wasm32"))]
#[path = "native_sources.rs"]
mod native_sources;
#[cfg(target_arch = "wasm32")]
#[path = "browser.rs"]
mod browser;
#[cfg(target_arch = "wasm32")]
#[path = "browser_sources.rs"]
mod browser_sources;
