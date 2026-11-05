use anyhow::Result;
use runmat_config::runtime::RunMatRuntimeConfig;

use crate::cli::{Cli, Commands};
use crate::commands::session::future::boxed_command;
use crate::commands::{
    accel, batch, benchmark, check, compile, config, fea, gc, job, mesh, mex, native_interface,
    package, repl, script, test, version,
};
use crate::remote;

pub async fn dispatch(cli: &Cli, config: &RunMatRuntimeConfig) -> Result<()> {
    boxed_command(|| {
        dispatch_with_graphics(cli, config, &runmat_core::SessionGraphicsHost::Unavailable)
    })
    .await
}

pub(crate) async fn dispatch_with_graphics(
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    graphics: &runmat_core::SessionGraphicsHost,
) -> Result<()> {
    let command = cli.command.clone();
    let mut script_path = cli.script.clone();
    let mut emit_bytecode = cli.emit_bytecode.clone();
    if command.is_some()
        && (cli.execution != crate::cli::ExecutionTargetArgs::default()
            || cli.script_json
            || cli.script_function.is_some()
            || !cli.script_args.is_empty())
    {
        anyhow::bail!(
            "place execution options after the subcommand, or use the filename shorthand"
        );
    }
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

    if command.is_none()
        && script_path.is_none()
        && !matches!(cli.execution.resolve()?, crate::cli::ExecutionTarget::Local)
    {
        anyhow::bail!("remote execution requires a project-owned source file");
    }

    // A command's asynchronous state belongs to that command, not every caller's
    // session-stack frame. Keep polling/drop on this lane; no task is spawned.
    match (command, script_path) {
        (Some(command), None) => {
            boxed_command(|| execute_command(command, cli, config, graphics)).await
        }
        (None, Some(script_path)) => {
            boxed_command(|| {
                script::execute_script_with_graphics(
                    script::ScriptInvocation {
                        script: script_path,
                        args: cli.script_args.clone(),
                        function: cli.script_function.clone(),
                        emit_bytecode_path: emit_bytecode,
                        json: cli.script_json,
                        fea_root: crate::cli::FeaRootArgs::default(),
                        execution: cli.execution.clone(),
                    },
                    cli,
                    config,
                    graphics,
                )
            })
            .await
        }
        (None, None) if !atty::is(atty::Stream::Stdin) => {
            boxed_command(|| {
                script::execute_stdin_script_with_graphics(emit_bytecode, cli, config, graphics)
            })
            .await
        }
        (None, None) => {
            boxed_command(|| repl::execute_repl_with_graphics(config, cli, graphics)).await
        }
        (Some(_), Some(_)) => {
            log::error!("Cannot specify both command and script file");
            std::process::exit(1);
        }
    }
}

async fn execute_command(
    command: Commands,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    graphics: &runmat_core::SessionGraphicsHost,
) -> Result<()> {
    // Bound this dispatcher independently of the largest unrelated subcommand.
    match command {
        Commands::Repl { verbose } => {
            let mut repl_config = config.clone();
            repl_config.runtime.verbose = verbose || config.runtime.verbose;
            boxed_command(|| repl::execute_repl_with_graphics(&repl_config, cli, graphics)).await
        }
        Commands::Run {
            file,
            root,
            execution,
            function,
            json,
            args,
        } => {
            boxed_command(|| {
                script::execute_script_with_graphics(
                    script::ScriptInvocation {
                        script: file,
                        args,
                        function,
                        emit_bytecode_path: cli.emit_bytecode.clone(),
                        json,
                        fea_root: root,
                        execution,
                    },
                    cli,
                    config,
                    graphics,
                )
            })
            .await
        }
        Commands::Mesh {
            source,
            execution,
            source_units,
            output,
            evidence,
            target_size,
            deviation,
            element_order,
            max_elements,
            seed,
            shared_interfaces,
            force,
            json,
        } => {
            boxed_command(|| {
                mesh::execute_with_target(
                    mesh::MeshCommand {
                        source,
                        source_units,
                        output,
                        evidence,
                        target_size_m: target_size,
                        maximum_deviation_m: deviation,
                        element_order,
                        maximum_elements: max_elements,
                        deterministic_seed: seed,
                        shared_interfaces,
                        force,
                        json,
                    },
                    execution,
                    cli,
                    config,
                )
            })
            .await
        }
        Commands::Compile {
            file,
            output,
            optimization,
            policy,
            explain_link,
            link_plan_json,
            linker,
            selected_pack_manifest,
            selected_components,
            selected_host_admission,
            keep_temps,
            force,
        } => {
            boxed_command(|| {
                compile::execute(
                    compile::CompileRequest {
                        file,
                        output,
                        optimization,
                        policy,
                        explain_link,
                        link_plan_json,
                        linker,
                        selected_pack_manifest,
                        selected_components,
                        selected_host_admission,
                        keep_temps,
                        force,
                    },
                    cli,
                    config,
                )
            })
            .await
        }
        Commands::Mex(args) => mex::execute(args),
        Commands::Mexcuda(args) => mex::execute_cuda(args),
        Commands::NativeInterface { command } => native_interface::execute(command),
        Commands::Check {
            file,
            root,
            json,
            verbose,
            deny,
            search_paths,
        } => {
            boxed_command(|| {
                check::execute_check(
                    file,
                    cli,
                    config,
                    check::CheckRequest {
                        root,
                        json,
                        verbose,
                        deny,
                        search_paths,
                    },
                )
            })
            .await
        }
        Commands::Plan { file, root, json } => {
            boxed_command(|| fea::plan(file, root, json, cli, config)).await
        }
        Commands::Snapshot {
            file,
            study,
            step,
            case,
            views,
            out,
            presentation,
        } => {
            boxed_command(|| {
                fea::snapshot(
                    file,
                    study,
                    step,
                    case,
                    views,
                    out,
                    presentation,
                    cli,
                    config,
                )
            })
            .await
        }
        Commands::RenderSnapshot {
            snapshot,
            out,
            views,
            presentation,
        } => boxed_command(|| fea::render_snapshot(snapshot, out, views, presentation)).await,
        Commands::VerifySnapshot { snapshot, out } => fea::verify_snapshot(snapshot, out),
        Commands::Reassess {
            file,
            study,
            original_definition,
            result_manifest,
            json,
        } => {
            boxed_command(|| {
                fea::reassess(
                    file,
                    study,
                    original_definition,
                    result_manifest,
                    json,
                    cli,
                    config,
                )
            })
            .await
        }
        Commands::Version { detailed } => {
            version::show_version(detailed);
            Ok(())
        }
        Commands::Info { json } => boxed_command(|| version::show_system_info(config, json)).await,
        Commands::AccelInfo { json, reset } => {
            boxed_command(|| accel::show_accel_info(json, reset)).await
        }
        #[cfg(feature = "wgpu")]
        Commands::AccelCalibrate {
            input,
            refresh,
            dry_run,
            json,
        } => boxed_command(|| accel::execute_accel_calibrate(input, refresh, dry_run, json)).await,
        Commands::Gc { gc_command } => boxed_command(|| gc::execute_gc_command(gc_command)).await,
        Commands::Benchmark {
            file,
            iterations,
            jit,
            warmup,
            report,
        } => {
            boxed_command(|| {
                benchmark::execute_benchmark(file, iterations, jit, warmup, report, cli, config)
            })
            .await
        }
        Commands::Config { config_command } => {
            boxed_command(|| config::execute_config_command(config_command, config)).await
        }
        Commands::Login {
            server,
            api_key,
            email,
            credential_store,
            org,
            project,
        } => {
            boxed_command(|| {
                remote::execute_login_command(
                    server,
                    api_key,
                    email,
                    credential_store,
                    org,
                    project,
                )
            })
            .await
        }
        Commands::Org { org_command } => {
            boxed_command(|| remote::execute_org_command(org_command)).await
        }
        Commands::Project { project_command } => {
            boxed_command(|| remote::execute_project_command(project_command)).await
        }
        Commands::Fs { fs_command } => {
            boxed_command(|| remote::execute_fs_command(fs_command)).await
        }
        Commands::Package { package_command } => {
            boxed_command(|| package::execute(package_command, cli)).await
        }
        Commands::Test(args) => boxed_command(|| test::execute(*args, cli, config)).await,
        Commands::Batch { batch_command } => boxed_command(|| batch::execute(batch_command)).await,
        Commands::Cluster { cluster_command } => {
            boxed_command(|| remote::execute_cluster_command(cluster_command)).await
        }
        Commands::Node(args) => boxed_command(|| remote::execute_node_command(args)).await,
        Commands::Job { job_command } => boxed_command(|| job::execute(job_command)).await,
        Commands::RecoveryKey { command } => boxed_command(|| job::execute_recovery(command)).await,
        Commands::Remote { remote_command } => {
            boxed_command(|| remote::execute_remote_command(remote_command, cli, config)).await
        }
    }
}

#[cfg(test)]
#[path = "dispatch/future_size_tests.rs"]
mod future_size_tests;
