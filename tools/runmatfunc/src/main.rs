fn main() {
    if let Some(code) = runmatfunc::codex::apply_patch::maybe_run_apply_patch_from_env() {
        std::process::exit(code);
    }

    if let Err(err) = runmatfunc::run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
