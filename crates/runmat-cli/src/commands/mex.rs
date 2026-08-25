use anyhow::{Context, Result};

use crate::cli::MexArgs;
use crate::presentation;

pub fn execute(args: MexArgs) -> Result<()> {
    let MexArgs {
        mut sources,
        output,
        out_dir,
        compiler,
        separate_complex,
        interleaved_complex: _,
        include_directories,
        definitions,
        compiler_argument,
        linker_argument,
        verbose,
    } = args;
    let first = sources.remove(0);
    let mut build =
        runmat_mex::MexBuild::new(first, out_dir).interleaved_complex(!separate_complex);
    for source in sources {
        build = build.source(source);
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
