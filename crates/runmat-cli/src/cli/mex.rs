use clap::Args;
use std::path::PathBuf;

#[derive(Args, Clone)]
pub struct MexArgs {
    /// C, C++, Fortran, or CUDA source files to compile and link into one MEX module
    #[arg(required = true)]
    pub sources: Vec<PathBuf>,
    /// Output module name without the platform MEX extension
    #[arg(short, long)]
    pub output: Option<String>,
    /// Directory in which to write the compiled MEX module
    #[arg(long, default_value = ".")]
    pub out_dir: PathBuf,
    /// Primary compiler driver (defaults through CC, CXX, FC/F77, NVCC, or MW_NVCC_PATH)
    #[arg(long)]
    pub compiler: Option<PathBuf>,
    /// Use R2017b separate-complex (mex default; mexcuda defaults to R2018a)
    #[arg(
        long = "R2017b",
        conflicts_with_all = ["r2018a", "large_array_dims", "compatible_array_dims"]
    )]
    pub r2017b: bool,
    /// Use the R2018a interleaved-complex, large-array API
    #[arg(
        long = "R2018a",
        conflicts_with_all = ["r2017b", "large_array_dims", "compatible_array_dims"]
    )]
    pub r2018a: bool,
    /// Use the legacy separate-complex, large-array API spelling
    #[arg(
        long = "largeArrayDims",
        conflicts_with_all = ["r2017b", "r2018a", "compatible_array_dims"]
    )]
    pub large_array_dims: bool,
    /// Use the separate-complex API with 32-bit array dimensions
    #[arg(
        long = "compatibleArrayDims",
        conflicts_with_all = ["r2017b", "r2018a", "large_array_dims"]
    )]
    pub compatible_array_dims: bool,
    /// Add a source include search directory
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
        assert!(args.r2017b);
        assert_eq!(args.include_directories, [PathBuf::from("vendor/include")]);
        assert_eq!(args.definitions, ["FEATURE=1"]);
        assert_eq!(args.output.as_deref(), Some("native_add"));
        assert_eq!(
            args.sources,
            [PathBuf::from("gateway.c"), PathBuf::from("support.c")]
        );
    }

    #[test]
    fn mexcuda_is_a_distinct_command_with_the_shared_build_options() {
        let cli = Cli::try_parse_from([
            "runmat",
            "mexcuda",
            "--R2018a",
            "--compiler",
            "nvcc",
            "-o",
            "gpu_add",
            "gateway.cu",
            "support.cpp",
        ])
        .unwrap();
        let Some(Commands::Mexcuda(args)) = cli.command else {
            panic!("expected mexcuda command");
        };
        assert!(args.r2018a);
        assert_eq!(args.compiler.as_deref(), Some(std::path::Path::new("nvcc")));
        assert_eq!(args.output.as_deref(), Some("gpu_add"));
        assert_eq!(
            args.sources,
            [PathBuf::from("gateway.cu"), PathBuf::from("support.cpp")]
        );
    }
}
