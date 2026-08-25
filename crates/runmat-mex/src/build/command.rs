use std::path::PathBuf;
use std::process::Command;

use super::{MexBuildError, MexBuildPlan};

#[derive(Debug, Clone)]
pub struct MexBuild {
    pub(super) compiler: PathBuf,
    pub(super) sources: Vec<PathBuf>,
    pub(super) output_directory: PathBuf,
    pub(super) output_name: String,
    pub(super) interleaved_complex: bool,
    pub(super) include_directories: Vec<PathBuf>,
    pub(super) definitions: Vec<String>,
    pub(super) compiler_arguments: Vec<String>,
    pub(super) linker_arguments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexBuildOutput {
    pub module: PathBuf,
    pub command: Vec<String>,
}

impl MexBuild {
    pub fn new(source: impl Into<PathBuf>, output_directory: impl Into<PathBuf>) -> Self {
        let source = source.into();
        let output_name = source
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("module")
            .to_string();
        Self {
            compiler: super::default_c_compiler(),
            sources: vec![source],
            output_directory: output_directory.into(),
            output_name,
            interleaved_complex: true,
            include_directories: Vec::new(),
            definitions: Vec::new(),
            compiler_arguments: Vec::new(),
            linker_arguments: Vec::new(),
        }
    }

    pub fn compiler(mut self, compiler: impl Into<PathBuf>) -> Self {
        self.compiler = compiler.into();
        self
    }

    pub fn source(mut self, source: impl Into<PathBuf>) -> Self {
        self.sources.push(source.into());
        self
    }

    pub fn output_name(mut self, name: impl Into<String>) -> Self {
        self.output_name = name.into();
        self
    }

    pub fn interleaved_complex(mut self, enabled: bool) -> Self {
        self.interleaved_complex = enabled;
        self
    }

    pub fn include_directory(mut self, directory: impl Into<PathBuf>) -> Self {
        self.include_directories.push(directory.into());
        self
    }

    pub fn define(mut self, definition: impl Into<String>) -> Self {
        self.definitions.push(definition.into());
        self
    }

    pub fn compiler_argument(mut self, argument: impl Into<String>) -> Self {
        self.compiler_arguments.push(argument.into());
        self
    }

    pub fn linker_argument(mut self, argument: impl Into<String>) -> Self {
        self.linker_arguments.push(argument.into());
        self
    }

    pub fn plan(&self) -> Result<MexBuildPlan, MexBuildError> {
        MexBuildPlan::for_build(self)
    }

    pub fn compile(&self) -> Result<MexBuildOutput, MexBuildError> {
        let plan = self.plan()?;
        std::fs::create_dir_all(&self.output_directory).map_err(|source| {
            MexBuildError::CreateOutputDirectory {
                directory: self.output_directory.clone(),
                source,
            }
        })?;
        let output = Command::new(&plan.compiler)
            .args(&plan.arguments)
            .output()
            .map_err(|source| MexBuildError::CompilerLaunch {
                compiler: plan.compiler.clone(),
                source,
            })?;
        let command = plan.command();
        if !output.status.success() {
            let diagnostics = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return Err(MexBuildError::CompilerFailure {
                command: command.join(" "),
                diagnostics,
            });
        }
        Ok(MexBuildOutput {
            module: plan.module,
            command,
        })
    }
}
