use anyhow::{Context, Result};

use crate::cli::MexArgs;
use crate::presentation;

pub fn execute(args: MexArgs) -> Result<()> {
    execute_with_mode(args, false)
}

pub fn execute_cuda(args: MexArgs) -> Result<()> {
    execute_with_mode(args, true)
}

fn execute_with_mode(args: MexArgs, require_cuda: bool) -> Result<()> {
    let verbose = args.verbose;
    let build = prepare_build(args, require_cuda)?;
    if verbose {
        let command = build
            .plan()
            .context("could not prepare MEX build")?
            .command();
        println!("{}", presentation::stdout().muted(command.join(" ")));
    }
    let output = build.compile().context("MEX build failed")?;
    println!(
        "{} {}",
        presentation::stdout().success("Built"),
        presentation::stdout().path(output.module.display())
    );
    Ok(())
}

fn prepare_build(args: MexArgs, require_cuda: bool) -> Result<runmat_mex::MexBuild> {
    let MexArgs {
        mut sources,
        output,
        out_dir,
        compiler,
        r2017b,
        r2018a,
        large_array_dims,
        compatible_array_dims,
        include_directories,
        definitions,
        compiler_argument,
        linker_argument,
        verbose: _,
    } = args;
    let first = sources.remove(0);
    let api_override = if r2018a {
        Some(runmat_mex::MexApi::R2018a)
    } else if large_array_dims {
        Some(runmat_mex::MexApi::LargeArrayDims)
    } else if compatible_array_dims {
        Some(runmat_mex::MexApi::CompatibleArrayDims)
    } else if r2017b {
        Some(runmat_mex::MexApi::R2017b)
    } else {
        None
    };
    let mut build = runmat_mex::MexBuild::new(first, out_dir);
    for source in sources {
        build = build.source(source);
    }
    if require_cuda && build.language() != runmat_mex::MexSourceLanguage::Cuda {
        anyhow::bail!("mexcuda requires at least one CUDA (.cu) source file");
    }
    if let Some(api) = api_override {
        build = build.api(api);
    }
    if let Some(output) = output {
        build = build.output_name(output);
    }
    if let Some(compiler) = compiler {
        build = build.compiler(compiler);
    }
    for directory in include_directories {
        build = build.include_directory(directory);
    }
    for definition in definitions {
        build = build.define(definition);
    }
    for argument in compiler_argument {
        build = build.compiler_argument(argument);
    }
    for argument in linker_argument {
        build = build.linker_argument(argument);
    }
    Ok(build)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Commands};
    use clap::Parser;

    fn prepared_definitions(command: &str, options: &[&str], source: &str) -> Vec<String> {
        let temporary = tempfile::tempdir().unwrap();
        let source_path = temporary.path().join(source);
        std::fs::write(&source_path, "void mexFunction(void) {}").unwrap();
        let mut arguments = vec!["runmat", command];
        arguments.extend_from_slice(options);
        let source_name = source_path.to_str().unwrap();
        arguments.push(source_name);
        let cli = Cli::try_parse_from(arguments).unwrap();
        let (args, cuda) = match cli.command.unwrap() {
            Commands::Mex(args) => (args, false),
            Commands::Mexcuda(args) => (args, true),
            _ => panic!("expected MEX build command"),
        };
        prepare_build(args, cuda)
            .unwrap()
            .plan()
            .unwrap()
            .steps
            .into_iter()
            .flat_map(|step| step.arguments)
            .collect()
    }

    #[test]
    fn source_language_selects_default_api_unless_cli_explicitly_overrides_it() {
        let c = prepared_definitions("mex", &[], "gateway.c");
        assert!(c.iter().any(|arg| arg.ends_with("TARGET_API_VERSION=700")));
        let cpp = prepared_definitions("mex", &[], "gateway.cpp");
        assert!(cpp
            .iter()
            .any(|arg| arg.ends_with("TARGET_API_VERSION=800")));
        let explicit = prepared_definitions("mex", &["--R2017b"], "gateway.cpp");
        assert!(explicit
            .iter()
            .any(|arg| arg.ends_with("TARGET_API_VERSION=700")));
    }
}
