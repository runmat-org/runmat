use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Subcommand, Clone)]
pub enum NativeInterfaceCommand {
    /// Prepare a canonical RunMat interface for an existing shared library
    Prepare(NativeInterfacePrepareArgs),
}

impl NativeInterfaceCommand {
    pub(crate) fn machine_output(&self) -> bool {
        match self {
            Self::Prepare(args) => args.json,
        }
    }
}

#[derive(Args, Clone)]
pub struct NativeInterfacePrepareArgs {
    /// Existing platform-native shared library
    #[arg(long)]
    pub library: PathBuf,
    /// Logical library name recorded in normalized interface metadata
    #[arg(long)]
    pub library_name: String,
    /// Primary public C header
    #[arg(long)]
    pub header: PathBuf,
    /// Public namespace used by clib.<interface> calls
    #[arg(long)]
    pub interface_name: String,
    /// Additional public header whose declarations belong to the interface
    #[arg(long = "add-header")]
    pub additional_headers: Vec<PathBuf>,
    /// Add a header include search directory
    #[arg(short = 'I', value_name = "DIRECTORY")]
    pub include_directories: Vec<PathBuf>,
    /// Add a preprocessor definition
    #[arg(short = 'D', value_name = "NAME[=VALUE]")]
    pub definitions: Vec<String>,
    /// Compiler frontend used to read the C declarations
    #[arg(long, default_value = "clang")]
    pub frontend: PathBuf,
    /// Prepared-interface manifest path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Emit the preparation result as JSON
    #[arg(long)]
    pub json: bool,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::cli::{Cli, Commands};

    #[test]
    fn preparation_options_retain_every_typed_input() {
        let cli = Cli::try_parse_from([
            "runmat",
            "native-interface",
            "prepare",
            "--library",
            "native/libfixture.so",
            "--library-name",
            "fixture_binary",
            "--header",
            "include/fixture.h",
            "--interface-name",
            "fixture",
            "--add-header",
            "include/detail.h",
            "-Iinclude",
            "-DFEATURE=1",
            "--frontend",
            "clang-20",
            "--output",
            "native/fixture.runmat.json",
            "--json",
        ])
        .unwrap();
        let Some(Commands::NativeInterface {
            command: NativeInterfaceCommand::Prepare(args),
        }) = cli.command
        else {
            panic!("expected native-interface prepare command");
        };
        assert_eq!(args.library_name, "fixture_binary");
        assert_eq!(args.interface_name, "fixture");
        assert_eq!(args.additional_headers, [PathBuf::from("include/detail.h")]);
        assert_eq!(args.include_directories, [PathBuf::from("include")]);
        assert_eq!(args.definitions, ["FEATURE=1"]);
        assert_eq!(args.frontend, PathBuf::from("clang-20"));
        assert_eq!(
            args.output,
            Some(PathBuf::from("native/fixture.runmat.json"))
        );
        assert!(args.json);
    }
}
