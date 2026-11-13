use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;

use crate::{
    cli::{AotOptLevel, AotPolicy, Cli},
    commands::{package::install_project_for_source, script::resolve_script_input},
    presentation,
};

mod selected;

pub struct CompileRequest {
    pub file: PathBuf,
    pub output: Option<PathBuf>,
    pub optimization: AotOptLevel,
    pub policy: AotPolicy,
    pub explain_link: bool,
    pub link_plan_json: Option<PathBuf>,
    pub linker: Option<PathBuf>,
    pub selected_pack_manifest: Option<PathBuf>,
    pub selected_components: Option<PathBuf>,
    pub selected_host_admission: Option<PathBuf>,
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
        selected_pack_manifest,
        selected_components,
        selected_host_admission,
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
    let selected_paths = selected::SelectedPaths::from_options(
        selected_pack_manifest,
        selected_components,
        selected_host_admission,
    )?;
    let runtime = if selected_paths.is_some() {
        None
    } else {
        Some(runmat_aot::archive::embedded_runtime_archive()?.ok_or_else(|| {
            anyhow::anyhow!(
                "this RunMat build does not contain a native compile runtime; rebuild it with scripts/build-runmat-with-aot-runtime.sh (or the Windows PowerShell equivalent)"
            )
        })?)
    };
    let policy = match policy {
        AotPolicy::NativeSpecialized => runmat_aot::compile::CompilationPolicy::NativeSpecialized,
        AotPolicy::ClosedWorld => runmat_aot::compile::CompilationPolicy::ClosedWorld,
        AotPolicy::DynamicRuntime => runmat_aot::compile::CompilationPolicy::DynamicRuntime,
        AotPolicy::Portable => runmat_aot::compile::CompilationPolicy::Portable,
    };
    if let Some(runtime) = runtime.as_ref() {
        policy.validate(&runtime.manifest.capabilities)?;
    }
    let mut session =
        super::session::create_session(false, false, config, "failed to create compile session")?;
    let project = install_project_for_source(&mut session, &file, cli).await?;
    let source = super::package::executable_source(
        project.as_ref().map(|project| &project.resolved.frozen),
        &file,
    )
    .await?;
    let unit = session
        .compile_executable_unit(&source, None)
        .await
        .map_err(|error| {
            anyhow::anyhow!(
                "{}",
                crate::diagnostics::format_frontend_error(
                    &error,
                    &source.relative_path,
                    Some(&source.text),
                )
                .unwrap_or_else(|| error.to_string())
            )
        })?;
    let report = unit.reachability_report();
    let inferred = unit.inferred_interop_manifest();
    let mut selected = inferred
        .adapters
        .iter()
        .map(|item| item.adapter.clone())
        .collect::<BTreeSet<_>>();
    let mut selected_mex_names = BTreeSet::new();
    if let Some(project) = project.as_ref() {
        for node in &report.nodes {
            if node.kind == runmat_mir::analysis::ReachabilityNodeKind::ExternalCallable {
                for declaration in &project.resolved.frozen.mex_artifacts {
                    if declaration.name == node.symbol {
                        selected_mex_names.insert(declaration.name.clone());
                    }
                }
            }
        }
        if !selected_mex_names.is_empty() {
            selected.insert(runmat_types::ForeignAdapterId::new(
                runmat_mex::MEX_ADAPTER_ID,
            )?);
        }
    }
    if report.has_unknown_edges {
        // An unbounded dynamic target may resolve any installed foreign call.
        // Grant the four supported runtime families and this frozen project's
        // declared artifacts, explicitly in the authenticated interop
        // manifest, never by receiver discovery.
        for adapter in [
            runmat_native_ffi::NATIVE_FFI_ADAPTER_ID,
            runmat_mex::MEX_ADAPTER_ID,
            runmat_java::JAVA_ADAPTER_ID,
            runmat_python::PYTHON_ADAPTER_ID,
        ] {
            selected.insert(runmat_types::ForeignAdapterId::new(adapter)?);
        }
    }
    let foreign_artifacts = if let Some(project) = project.as_ref() {
        super::package::prepare_foreign_artifacts_for_selection(
            &project.resolved.frozen,
            config,
            &selected,
            (!report.has_unknown_edges).then_some(&selected_mex_names),
        )?
    } else if selected.contains(&runmat_types::ForeignAdapterId::new(
        runmat_python::PYTHON_ADAPTER_ID,
    )?) {
        super::package::prepare_python_runtime(config)?
    } else {
        super::package::PreparedForeignArtifacts::empty()
    };
    let closed_grant = runmat_types::InteropManifest {
        schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
        foreign_types: Vec::new(),
        adapters: selected
            .iter()
            .cloned()
            .map(|adapter| runmat_types::ForeignAdapterRequirement {
                adapter,
                minimum_version: 1,
                capabilities: runmat_types::CapabilitySet(BTreeSet::from([
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                ])),
                execution_stack: runmat_types::ExecutionStackRequirement::Process,
                artifact_identities: Vec::new(),
            })
            .collect(),
        adapter_contracts: Vec::new(),
    };
    let interop = runmat_types::InteropManifest::merge([
        inferred,
        closed_grant,
        foreign_artifacts.interop.clone(),
    ])
    .map_err(|error| anyhow::anyhow!("{}: {}", error.path, error.message))?;
    let output = output.unwrap_or_else(|| default_output(&file));
    if let Some(paths) = selected_paths {
        return selected::execute(
            &unit,
            interop,
            foreign_artifacts,
            config,
            selected::SelectedCompileRequest {
                paths,
                output,
                optimization,
                policy,
                explain_link,
                link_plan_json,
                keep_temps,
                force,
            },
        );
    }
    let runtime = runtime.expect("pure compile route has an embedded runtime");
    let program_link_plan = runmat_aot::compile::build_program_link_plan_with_interop(
        &unit, &runtime, policy, interop,
    )?;
    if explain_link {
        print_link_explanation(&program_link_plan);
    }
    if let Some(path) = link_plan_json.as_deref() {
        write_link_plan(path, &program_link_plan, force)?;
    }
    let object = runmat_aot::emit_native_object(
        &unit,
        runmat_aot::NativeObjectOptions {
            clock_inputs: None,
            local_service_inputs: None,
            optimization: match optimization {
                AotOptLevel::None => runmat_native_codegen::aot::NativeOptimization::None,
                AotOptLevel::Size => runmat_native_codegen::aot::NativeOptimization::SpeedAndSize,
                AotOptLevel::Speed => runmat_native_codegen::aot::NativeOptimization::Speed,
            },
            retained_functions: Some(program_link_plan.retained_functions()),
            runtime_binding_mode: program_link_plan.runtime_composition.binding_mode(),
            retained_builtin_bindings: program_link_plan.runtime_composition.bindings().to_vec(),
            interop: program_link_plan.interop.clone(),
            foreign_policy: Some(
                runmat_core::foreign_artifacts::capture_foreign_policy_object(&config.foreign)
                    .map_err(anyhow::Error::msg)?,
            ),
            native_interfaces: foreign_artifacts.native_interfaces,
            mex_artifacts: foreign_artifacts.mex_artifacts,
            java_artifacts: foreign_artifacts.java_artifacts,
            python_artifacts: foreign_artifacts.python_artifacts,
        },
    )?;
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
    match &plan.runtime_composition {
        runmat_aot::compile::RuntimeComposition::Selected { bindings } => {
            println!(
                "Runtime composition: selected ({} exact builtin bindings)",
                bindings.len()
            );
        }
        runmat_aot::compile::RuntimeComposition::Dynamic { reason } => {
            println!("Runtime composition: dynamic ({reason:?})");
        }
    }
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
    if !plan.runtime_composition.bindings().is_empty() {
        println!("Retained runtime builtin bindings:");
        for binding in plan.runtime_composition.bindings() {
            println!(
                "  {}#{}: {}",
                binding.name, binding.variant, binding.native_symbol
            );
        }
    }
    for adapter in &plan.interop.adapters {
        println!(
            "Retained foreign adapter {} v{} with {} exact artifact(s)",
            adapter.adapter,
            adapter.minimum_version,
            adapter.artifact_identities.len()
        );
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
