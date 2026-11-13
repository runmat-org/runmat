use super::*;
use clap::Parser;

// The native GUI keeps execution on an ordinary session thread. Its dispatcher
// must not inline every possible command's asynchronous state into that thread's
// reactor/select frame. This checks construction only: no stdin, GUI or runtime
// execution is started by the regression.
#[test]
fn command_dispatch_futures_keep_bounded_inline_state() {
    let config = RunMatRuntimeConfig::default();
    let graphics = runmat_core::SessionGraphicsHost::Unavailable;
    const MAX_INLINE_STATE: usize = 32 * 1024;
    for args in [
        vec!["runmat", "qualification.m"],
        vec!["runmat", "run", "qualification.m"],
        vec!["runmat", "repl"],
        vec!["runmat"],
    ] {
        let cli = Cli::try_parse_from(args).expect("valid CLI fixture");
        let dispatch = dispatch_with_graphics(&cli, &config, &graphics);
        let size = std::mem::size_of_val(&dispatch);
        assert!(size <= MAX_INLINE_STATE, "dispatcher inline state: {size}");
        if let Some(command) = cli.command.clone() {
            let command = execute_command(command, &cli, &config, &graphics);
            let size = std::mem::size_of_val(&command);
            assert!(size <= MAX_INLINE_STATE, "command inline state: {size}");
        }
    }
}
