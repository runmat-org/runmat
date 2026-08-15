use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;

use crate::{
    cli::{AotOptLevel, AotPolicy, Cli},
    commands::{package::install_project_for_source, script::resolve_script_input},
    presentation,
};

pub struct CompileRequest {
    pub file: PathBuf,
    pub output: Option<PathBuf>,
    pub optimization: AotOptLevel,
    pub policy: AotPolicy,
    pub explain_link: bool,
    pub link_plan_json: Option<PathBuf>,
    pub linker: Option<PathBuf>,
    pub keep_temps: bool,
    pub force: bool,
}

pub async fn execute(
    request: CompileRequest,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let CompileRequest {
        file,
        output,
        optimization,
        policy,
        explain_link,
        link_plan_json,
        linker,
        keep_temps,
        force,
    } = request;
    let file = resolve_script_input(file)?;
    if !file
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("m"))
    {
        anyhow::bail!("runmat compile currently accepts a .m entrypoint");
    }
    let runtime = runmat_aot::archive::embedded_runtime_archive()?.ok_or_else(|| {
        anyhow::anyhow!(
            "this RunMat build does not contain a native compile runtime; rebuild it with scripts/build-runmat-with-aot-runtime.sh (or the Windows PowerShell equivalent)"
        )
    })?;
    let policy = match policy {
        AotPolicy::NativeSpecialized => runmat_aot::compile::CompilationPolicy::NativeSpecialized,
        AotPolicy::ClosedWorld => runmat_aot::compile::CompilationPolicy::ClosedWorld,
        AotPolicy::DynamicRuntime => runmat_aot::compile::CompilationPolicy::DynamicRuntime,
        AotPolicy::Portable => runmat_aot::compile::CompilationPolicy::Portable,
    };
    policy.validate(&runtime.manifest.capabilities)?;
    let source_text = std::fs::read_to_string(&file)
        .with_context(|| format!("failed to read compile entrypoint `{}`", file.display()))?;
    let mut session =
        super::session::create_session(false, false, config, "failed to create compile session")?;
    let _project_lease = install_project_for_source(&mut session, &file, cli).await?;
    let source = runmat_core::ExecutableSource::new("root", file.to_string_lossy(), source_text);
    let unit = session
        .compile_executable_unit(source, None)
        .await
        .map_err(|error| anyhow::anyhow!(error))?;
    let program_link_plan = runmat_aot::compile::build_program_link_plan(&unit, &runtime, policy)?;
    if explain_link {
        print_link_explanation(&program_link_plan);
    }
    if let Some(path) = link_plan_json.as_deref() {
        write_link_plan(path, &program_link_plan, force)?;
    }
    let object = runmat_aot::emit_native_object(
        &unit,
        runmat_aot::NativeObjectOptions {
            optimization: match optimization {
                AotOptLevel::None => runmat_native_codegen::aot::NativeOptimization::None,
                AotOptLevel::Size => runmat_native_codegen::aot::NativeOptimization::SpeedAndSize,
                AotOptLevel::Speed => runmat_native_codegen::aot::NativeOptimization::Speed,
            },
            retained_functions: Some(program_link_plan.retained_functions()),
            runtime_binding_mode: match policy {
                runmat_aot::compile::CompilationPolicy::ClosedWorld => {
                    runmat_native_codegen::aot::AotRuntimeBindingMode::ClosedWorld
                }
                _ => runmat_native_codegen::aot::AotRuntimeBindingMode::Dynamic,
            },
            retained_builtin_bindings: program_link_plan.retained_builtin_bindings.clone(),
        },
    )?;
    let output = output.unwrap_or_else(|| default_output(&file));
    let linked = runmat_aot::link::link_standalone(
        &object,
        &runtime,
        &runmat_aot::link::LinkRequest {
            output,
            linker,
            keep_temps,
            overwrite: force,
        },
    )?;
    println!(
        "{} {}",
        presentation::stdout().success("Compiled"),
        linked.output.display()
    );
    if let Some(temporary) = linked.retained_temps {
        println!("Temporary link inputs: {}", temporary.display());
    }
    Ok(())
}

fn print_link_explanation(plan: &runmat_aot::compile::ProgramLinkPlan) {
    println!("Link policy: {:?}", plan.policy);
    println!("Target: {}", plan.target_triple);
    println!("Reachable program and runtime symbols:");
    for node in &plan.reachability.nodes {
        println!(
            "  [{:?}] {} :: {} ({})",
            node.certainty, node.module, node.symbol, node.id
        );
        for edge in plan
            .reachability
            .edges
            .iter()
            .filter(|edge| edge.to == node.id)
        {
            let source = edge.from.as_deref().unwrap_or("link root");
            if let Some(detail) = edge.detail.as_deref() {
                println!("    <- {source}: {:?} ({detail})", edge.reason);
            } else {
                println!("    <- {source}: {:?}", edge.reason);
            }
        }
    }
    println!("Retained runtime families:");
    for family in &plan.retained_runtime_families {
        println!("  {}: {}", family.module, family.reason);
    }
    if !plan.retained_builtin_bindings.is_empty() {
        println!("Retained runtime builtin bindings:");
        for binding in &plan.retained_builtin_bindings {
            println!(
                "  {}#{}: {}",
                binding.name, binding.variant, binding.native_symbol
            );
        }
    }
    if !plan.omitted_runtime_families.is_empty() {
        println!(
            "Omitted runtime families: {}",
            plan.omitted_runtime_families.join(", ")
        );
    }
}

fn write_link_plan(
    path: &Path,
    plan: &runmat_aot::compile::ProgramLinkPlan,
    overwrite: bool,
) -> Result<()> {
    if path.exists() && !overwrite {
        anyhow::bail!(
            "link-plan output `{}` already exists; pass --force to replace it",
            path.display()
        );
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create link-plan directory `{}`",
                parent.display()
            )
        })?;
    }
    let directory = parent.unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(directory).with_context(|| {
        format!(
            "failed to create temporary link plan in `{}`",
            directory.display()
        )
    })?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), plan)
        .context("failed to encode link plan")?;
    use std::io::Write as _;
    temporary
        .as_file_mut()
        .write_all(b"\n")
        .context("failed to finish link plan")?;
    temporary
        .as_file_mut()
        .sync_all()
        .context("failed to sync link plan")?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to publish link plan `{}`", path.display()))?;
    Ok(())
}

fn default_output(file: &Path) -> PathBuf {
    let stem = file
        .file_stem()
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| std::ffi::OsStr::new("program"));
    let mut output = PathBuf::from(stem);
    if cfg!(target_os = "windows") {
        output.set_extension("exe");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::default_output;

    #[test]
    fn default_output_uses_entrypoint_stem() {
        let output = default_output(std::path::Path::new("src/main.m"));
        if cfg!(target_os = "windows") {
            assert_eq!(output, std::path::Path::new("main.exe"));
        } else {
            assert_eq!(output, std::path::Path::new("main"));
        }
    }
}
