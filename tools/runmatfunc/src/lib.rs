pub mod app;
pub mod builtin;
pub mod cli;
pub mod codex;
pub mod context;
pub mod errors;
pub mod jobs;
pub mod logging;
pub mod tui;
pub mod workspace;

pub fn run() -> Result<(), anyhow::Error> {
    touch_runtime();
    let args = cli::parse();
    logging::init(&args)?;
    let mut ctx = app::AppContext::new(&args)?;
    cli::handle_command(&mut ctx, &args)
}

fn touch_runtime() {
    let value = runmat_builtins::Value::Bool(false);
    let _ = runmat_runtime::is_gpu_value(&value);
}

#[cfg(all(test, feature = "embedded-codex"))]
#[ctor::ctor]
fn handle_apply_patch_invocation_for_tests() {
    if let Some(code) = crate::codex::apply_patch::maybe_run_apply_patch_from_env() {
        std::process::exit(code);
    }
}
