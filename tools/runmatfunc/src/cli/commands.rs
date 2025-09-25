use crate::app::AppContext;
use crate::cli::args::Command;
use crate::cli::CliArgs;
use anyhow::Result;

/// Dispatch parsed CLI commands into application actions.
pub fn handle_command(ctx: &mut AppContext, args: &CliArgs) -> Result<()> {
    match &args.command {
        None | Some(Command::Browse) => ctx.launch_tui(),
        Some(Command::Manifest) => ctx.print_manifest(),
        Some(Command::Docs { out_dir }) => ctx.emit_docs(out_dir),
        Some(Command::Builtin { name, model, codex }) => {
            ctx.run_builtin(name, model.clone(), *codex)
        }
    }
}
