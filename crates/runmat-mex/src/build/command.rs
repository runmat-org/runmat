use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

use super::mex_suffix;

#[derive(Debug, Clone)]
pub struct MexBuild {
    compiler: PathBuf,
    source: PathBuf,
    output_directory: PathBuf,
    output_name: String,
    interleaved_complex: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexBuildOutput {
    pub module: PathBuf,
    pub command: Vec<String>,
}

#[derive(Debug, Error)]
pub enum MexBuildError {
    #[error("C MEX compilation is unavailable for this target")]
    UnsupportedTarget,
    #[error("MEX source does not exist: {0}")]
    MissingSource(PathBuf),
    #[error("failed to invoke C compiler {compiler}: {source}")]
    CompilerLaunch {
        compiler: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("C MEX compilation failed\ncommand: {command}\n{diagnostics}")]
    CompilerFailure {
        command: String,
        diagnostics: String,
    },
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
            compiler: std::env::var_os("CC")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("cc")),
            source,
            output_directory: output_directory.into(),
            output_name,
            interleaved_complex: true,
        }
    }

    pub fn compiler(mut self, compiler: impl Into<PathBuf>) -> Self {
        self.compiler = compiler.into();
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

    pub fn compile(&self) -> Result<MexBuildOutput, MexBuildError> {
        let suffix = mex_suffix().ok_or(MexBuildError::UnsupportedTarget)?;
        if !self.source.is_file() {
            return Err(MexBuildError::MissingSource(self.source.clone()));
        }
        let crate_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
        let include = crate_directory.join("include");
        let shim = crate_directory.join("shim/runmat_mex_shim.c");
        let module = self
            .output_directory
            .join(format!("{}.{}", self.output_name, suffix));

        let mut arguments = Vec::<String>::new();
        if cfg!(target_os = "macos") {
            arguments.push("-dynamiclib".into());
        } else {
            arguments.push("-shared".into());
        }
        if !cfg!(target_os = "windows") {
            arguments.push("-fPIC".into());
        }
        arguments.extend(["-std=c11".into(), "-O2".into()]);
        arguments.push(format!("-I{}", include.display()));
        if self.interleaved_complex {
            arguments.push("-DRUNMAT_MX_INTERLEAVED_COMPLEX=1".into());
        }
        let function_name = self
            .output_name
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || character == '_' {
                    character
                } else {
                    '_'
                }
            })
            .collect::<String>();
        arguments.push(format!("-DRUNMAT_MEX_FUNCTION_NAME=\"{function_name}\""));
        arguments.push(self.source.display().to_string());
        arguments.push(shim.display().to_string());
        arguments.push("-o".into());
        arguments.push(module.display().to_string());

        let output = Command::new(&self.compiler)
            .args(&arguments)
            .output()
            .map_err(|source| MexBuildError::CompilerLaunch {
                compiler: self.compiler.clone(),
                source,
            })?;
        let command = std::iter::once(self.compiler.display().to_string())
            .chain(arguments.iter().cloned())
            .collect::<Vec<_>>();
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
        Ok(MexBuildOutput { module, command })
    }
}
