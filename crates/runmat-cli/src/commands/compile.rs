use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;

use crate::{
    cli::{AotOptLevel, Cli},
    commands::{package::install_project_for_source, script::resolve_script_input},
    presentation,
};

pub struct CompileRequest {
    pub file: PathBuf,
    pub output: Option<PathBuf>,
    pub optimization: AotOptLevel,
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
            "this RunMat build does not contain a native compile runtime; install an official build or build with scripts/build-runmat-with-aot-runtime.sh"
        )
    })?;
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
    let object = runmat_aot::emit_native_object(
        &unit,
        runmat_aot::NativeObjectOptions {
            optimization: match optimization {
                AotOptLevel::None => runmat_native_codegen::aot::NativeOptimization::None,
                AotOptLevel::Size => runmat_native_codegen::aot::NativeOptimization::SpeedAndSize,
                AotOptLevel::Speed => runmat_native_codegen::aot::NativeOptimization::Speed,
            },
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
