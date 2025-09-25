use crate::app::AppContext;
use crate::cli::args::{Command, QueueAction};
use crate::cli::CliArgs;
use anyhow::Result;

/// Dispatch parsed CLI commands into application actions.
pub fn handle_command(ctx: &mut AppContext, args: &CliArgs) -> Result<()> {
    match &args.command {
        None | Some(Command::Browse) => ctx.launch_tui(),
        Some(Command::Manifest) => ctx.print_manifest(),
        Some(Command::Docs { out_dir }) => ctx.emit_docs(out_dir.as_deref()),
        Some(Command::Builtin {
            name,
            model,
            codex,
            diff,
        }) => ctx.run_builtin(name, model.clone(), *codex, *diff),
        Some(Command::Queue { action }) => match action {
            QueueAction::Add {
                builtin,
                model,
                codex,
            } => ctx.queue_add(builtin, model.clone(), *codex),
            QueueAction::List => ctx.queue_list(),
            QueueAction::Run { max } => ctx.queue_run(*max),
            QueueAction::Clear => ctx.queue_clear(),
        },
    }
}
