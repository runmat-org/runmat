use anyhow::Result;
use runmat_config::RunMatConfig;

use crate::cli::{Cli, Commands};
use crate::commands::{accel, benchmark, config, gc, kernel, pkg, repl, script, snapshot, version};
use crate::remote;

pub async fn dispatch(cli: &Cli, config: &RunMatConfig) -> Result<()> {
    let command = cli.command.clone();
    let mut script_path = cli.script.clone();
    let mut emit_bytecode = cli.emit_bytecode.clone();
    if command.is_none() && script_path.is_none() {
        if let Some(path) = emit_bytecode.clone() {
            let is_matlab = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("m"))
                .unwrap_or(false);
            if is_matlab || path.exists() {
                script_path = Some(path);
                emit_bytecode = Some(std::path::PathBuf::from("-"));
            }
        }
    }

    match (command, script_path) {
        (Some(command), None) => execute_command(command, cli, config).await,
        (None, Some(script_path)) => {
            script::execute_script(script_path, emit_bytecode, cli, config).await
        }
        (None, None) => repl::execute_repl(config).await,
        (Some(_), Some(_)) => {
            log::error!("Cannot specify both command and script file");
            std::process::exit(1);
        }
    }
}

async fn execute_command(command: Commands, cli: &Cli, config: &RunMatConfig) -> Result<()> {
    match command {
        Commands::Repl { verbose } => {
            let mut repl_config = config.clone();
            repl_config.runtime.verbose = verbose || config.runtime.verbose;
            repl::execute_repl(&repl_config).await
        }
        Commands::Kernel {
            ip,
            key,
            transport,
            signature_scheme,
            shell_port,
            iopub_port,
            stdin_port,
            control_port,
            hb_port,
            connection_file,
        } => {
            kernel::execute_kernel(
                ip,
                key,
                transport,
                signature_scheme,
                shell_port,
                iopub_port,
                stdin_port,
                control_port,
                hb_port,
                connection_file,
                cli.timeout,
            )
            .await
        }
        Commands::KernelConnection { connection_file } => {
            kernel::execute_kernel_with_connection(connection_file, cli.timeout).await
        }
        Commands::Run { file, args } => {
            script::execute_script_with_args(file, args, cli.emit_bytecode.clone(), cli, config)
                .await
        }
        Commands::Version { detailed } => {
            version::show_version(detailed);
            Ok(())
        }
        Commands::Info => version::show_system_info(cli).await,
        Commands::AccelInfo { json, reset } => accel::show_accel_info(json, reset).await,
        #[cfg(feature = "wgpu")]
        Commands::AccelCalibrate {
            input,
            dry_run,
            json,
        } => accel::execute_accel_calibrate(input, dry_run, json).await,
        Commands::Gc { gc_command } => gc::execute_gc_command(gc_command).await,
        Commands::Benchmark {
            file,
            iterations,
            jit,
        } => benchmark::execute_benchmark(file, iterations, jit, cli, config).await,
        Commands::Snapshot { snapshot_command } => {
            snapshot::execute_snapshot_command(snapshot_command).await
        }
        Commands::Config { config_command } => {
            config::execute_config_command(config_command, config).await
        }
        Commands::Login {
            server,
            api_key,
            email,
            credential_store,
            org,
            project,
        } => {
            remote::execute_login_command(server, api_key, email, credential_store, org, project)
                .await
        }
        Commands::Org { org_command } => remote::execute_org_command(org_command).await,
        Commands::Project { project_command } => {
            remote::execute_project_command(project_command).await
        }
        Commands::Fs { fs_command } => remote::execute_fs_command(fs_command).await,
        Commands::Remote { remote_command } => remote::execute_remote_command(remote_command).await,
        Commands::Pkg { pkg_command } => pkg::execute_pkg_command(pkg_command).await,
    }
}
