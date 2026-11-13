//! Ordinary CLI native graphics owner. No session-local value crosses lanes.

use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_core::SessionGraphicsHost;
use runmat_plot::gui::host_loop::{NativeGraphicsHost, WindowOpenPort};

use crate::cli::{Cli, Commands};

pub(super) fn applies(cli: &Cli) -> bool {
    let local = match &cli.command {
        None => cli.execution.cluster.is_none(),
        Some(Commands::Run { execution, .. }) => execution.cluster.is_none(),
        Some(Commands::Repl { .. }) => true,
        _ => false,
    };
    local && cli.emit_bytecode.is_none()
}

struct LaneExit(WindowOpenPort);
impl Drop for LaneExit {
    fn drop(&mut self) {
        self.0.shutdown();
    }
}

pub(super) fn run(cli: Cli, config: RunMatRuntimeConfig) -> Result<()> {
    runmat_plot::register_main_thread();
    let gui = config.plotting.gui.clone().unwrap_or_default();
    let host = NativeGraphicsHost::new(gui.maximum_pending_opens, gui.maximum_windows)
        .context("failed to create native graphics host on the OS main thread")?;
    let opener = host.open_port();
    let graphics = SessionGraphicsHost::Native {
        opener: opener.clone(),
        gui,
    };
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let execution = std::thread::Builder::new().name("runmat-session".into()).spawn(move || {
        let _exit = LaneExit(opener);
        let reactor = tokio::runtime::Builder::new_current_thread().enable_all().build()
            .context("failed to create native session reactor")?;
        // Construction, all polls, cancellation/drop, and GC stay on this lane.
        reactor.block_on(async move {
            super::bootstrap::initialize_execution_lane(&config, false)?;
            tokio::select! {
                result = super::dispatch::dispatch_with_graphics(&cli, &config, &graphics) => result,
                _ = stopped => anyhow::bail!("native graphics host stopped before session completion"),
            }
        })
    }).context("failed to start native session lane")?;
    let host_result = host.run();
    let _ = stop.send(());
    let execution_result = execution
        .join()
        .map_err(|_| anyhow::anyhow!("native session lane panicked"))?;
    host_result.context("native graphics host failed")?;
    execution_result
}
