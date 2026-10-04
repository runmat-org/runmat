use super::*;
use crate::builtins::common::test_support;
use crate::builtins::io::filetext::registry;
use crate::builtins::io::filetext::{fclose, fopen};
use crate::RuntimeError;
#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_accelerate_api::HostTensorView;
use runmat_filesystem::File;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::system_time_now;
use runmat_value::{IntegerStorage, Tensor};
use std::io::Read;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

fn unwrap_error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn run_evaluate(
    fid_value: &Value,
    data_value: &Value,
    rest: &[Value],
) -> BuiltinResult<FwriteEval> {
    futures::executor::block_on(evaluate(fid_value, data_value, rest))
}

fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
    futures::executor::block_on(fopen::evaluate(args))
}

fn run_fclose(args: &[Value]) -> BuiltinResult<fclose::FcloseEval> {
    futures::executor::block_on(fclose::evaluate(args))
}

fn registry_guard() -> registry::RegistryTestGuard {
    registry::test_guard()
}

mod controls;
mod admission;
mod writing;

#[cfg(not(target_arch = "wasm32"))]
fn unique_path(prefix: &str) -> PathBuf {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards");
    let filename = format!(
        "runmat_{prefix}_{}_{}.tmp",
        now.as_secs(),
        now.subsec_nanos()
    );
    std::env::temp_dir().join(filename)
}

#[cfg(target_arch = "wasm32")]
fn unique_path(prefix: &str) -> PathBuf {
    test_support::filesystem::unique_path(prefix)
}
