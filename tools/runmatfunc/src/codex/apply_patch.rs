#[cfg(feature = "embedded-codex")]
pub fn maybe_run_apply_patch_from_env() -> Option<i32> {
    use std::io::{Read, Write};

    let mut args = std::env::args();
    let _program = args.next();
    let Some(flag) = args.next() else {
        return None;
    };

    if flag != codex_core::CODEX_APPLY_PATCH_ARG1 {
        return None;
    }

    let patch_arg = args.next();
    let patch = match patch_arg {
        Some(patch) => patch,
        None => {
            let mut buf = String::new();
            match std::io::stdin().read_to_string(&mut buf) {
                Ok(_) if !buf.is_empty() => buf,
                _ => {
                    eprintln!(
                        "Usage: runmatfunc --codex-run-as-apply-patch '<PATCH>'\n       echo '<PATCH>' | runmatfunc --codex-run-as-apply-patch"
                    );
                    return Some(2);
                }
            }
        }
    };

    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let exit_code = match codex_apply_patch::apply_patch(&patch, &mut stdout, &mut stderr) {
        Ok(()) => {
            let _ = stdout.flush();
            0
        }
        Err(_) => 1,
    };

    Some(exit_code)
}

#[cfg(not(feature = "embedded-codex"))]
pub fn maybe_run_apply_patch_from_env() -> Option<i32> {
    let mut args = std::env::args();
    let _program = args.next();
    let Some(flag) = args.next() else {
        return None;
    };

    if flag != "--codex-run-as-apply-patch" {
        return None;
    }

    eprintln!("error: apply_patch mode requires the embedded-codex feature");
    Some(1)
}
