fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.get(1).map(|s| s.as_str()) == Some(codex_apply_patch_flag()) {
        #[cfg(feature = "embedded-codex")]
        {
            std::process::exit(run_apply_patch(args.get(2)));
        }
        #[cfg(not(feature = "embedded-codex"))]
        {
            eprintln!("error: apply_patch mode requires the embedded-codex feature");
            std::process::exit(1);
        }
    }

    if let Err(err) = runmatfunc::run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

#[cfg(feature = "embedded-codex")]
fn run_apply_patch(patch_arg: Option<&String>) -> i32 {
    use std::io::{Read, Write};

    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();

    let patch = match patch_arg {
        Some(patch) => patch.clone(),
        None => {
            let mut buf = String::new();
            match std::io::stdin().read_to_string(&mut buf) {
                Ok(_) if !buf.is_empty() => buf,
                _ => {
                    eprintln!(
                        "Usage: runmatfunc --codex-run-as-apply-patch '<PATCH>'\n       echo '<PATCH>' | runmatfunc --codex-run-as-apply-patch"
                    );
                    return 2;
                }
            }
        }
    };

    match codex_apply_patch::apply_patch(&patch, &mut stdout, &mut stderr) {
        Ok(()) => {
            let _ = stdout.flush();
            0
        }
        Err(_) => 1,
    }
}

#[cfg(feature = "embedded-codex")]
fn codex_apply_patch_flag() -> &'static str {
    codex_core::CODEX_APPLY_PATCH_ARG1
}

#[cfg(not(feature = "embedded-codex"))]
fn codex_apply_patch_flag() -> &'static str {
    "--codex-run-as-apply-patch"
}
