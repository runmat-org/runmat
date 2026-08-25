use clap::Args;
use std::path::PathBuf;

#[derive(Args, Clone)]
pub struct MexArgs {
    /// C source files to compile and link into one MEX module
    #[arg(required = true)]
    pub sources: Vec<PathBuf>,
    /// Output module name without the platform MEX extension
    #[arg(short, long)]
    pub output: Option<String>,
    /// Directory in which to write the compiled MEX module
    #[arg(long, default_value = ".")]
    pub out_dir: PathBuf,
    /// C compiler driver (defaults to CC, cc, or cl.exe)
    #[arg(long)]
    pub compiler: Option<PathBuf>,
    /// Build against the separate-complex Matrix API
    #[arg(long = "R2017b", conflicts_with = "interleaved_complex")]
    pub separate_complex: bool,
    /// Build against the interleaved-complex Matrix API (the default)
    #[arg(long = "R2018a", conflicts_with = "separate_complex")]
    pub interleaved_complex: bool,
    /// Add a C header search directory
    #[arg(short = 'I', value_name = "DIRECTORY")]
    pub include_directories: Vec<PathBuf>,
    /// Add a preprocessor definition
    #[arg(short = 'D', value_name = "NAME[=VALUE]")]
    pub definitions: Vec<String>,
    /// Pass one argument directly to the compiler driver
    #[arg(long, allow_hyphen_values = true)]
    pub compiler_argument: Vec<String>,
    /// Pass one argument directly to the linker phase
    #[arg(long, allow_hyphen_values = true)]
    pub linker_argument: Vec<String>,
    /// Print the compiler command before executing it
    #[arg(short, long)]
    pub verbose: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Commands};
    use clap::Parser;

    #[test]
    fn mex_build_options_parse_without_entering_script_mode() {
        let cli = Cli::try_parse_from([
            "runmat",
            "mex",
            "--R2017b",
            "-Ivendor/include",
            "-DFEATURE=1",
            "-o",
            "native_add",
            "gateway.c",
            "support.c",
        ])
        .unwrap();
        let Some(Commands::Mex(args)) = cli.command else {
            panic!("expected mex command");
        };
        assert!(args.separate_complex);
        assert_eq!(args.include_directories, [PathBuf::from("vendor/include")]);
        assert_eq!(args.definitions, ["FEATURE=1"]);
        assert_eq!(args.output.as_deref(), Some("native_add"));
        assert_eq!(
            args.sources,
            [PathBuf::from("gateway.c"), PathBuf::from("support.c")]
        );
    }
}
