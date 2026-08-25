use std::path::PathBuf;
use std::process::Command;

use super::{compiler_family, MexArtifactManifest, MexBuildError, MexBuildPlan, MexTarget};

/// C Matrix API selected for a MEX build.
///
/// The release pins select the complex representation as well as the array
/// dimension API. The legacy spellings remain distinct because MATLAB treats
/// all four choices as mutually exclusive command-line API selections.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MexApi {
    /// Separate-complex, large-array API (MATLAB's current default).
    #[default]
    R2017b,
    /// Interleaved-complex, large-array API.
    R2018a,
    /// Legacy spelling for the separate-complex, large-array API.
    LargeArrayDims,
    /// Separate-complex compatibility API with 32-bit array dimensions.
    CompatibleArrayDims,
}

impl MexApi {
    pub fn uses_interleaved_complex(self) -> bool {
        matches!(self, Self::R2018a)
    }
}

#[derive(Debug, Clone)]
pub struct MexBuild {
    pub(super) compiler: PathBuf,
    pub(super) sources: Vec<PathBuf>,
    pub(super) output_directory: PathBuf,
    pub(super) output_name: String,
    pub(super) api: MexApi,
    pub(super) include_directories: Vec<PathBuf>,
    pub(super) definitions: Vec<String>,
    pub(super) compiler_arguments: Vec<String>,
    pub(super) linker_arguments: Vec<String>,
    pub(super) target: MexTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexBuildOutput {
    pub module: PathBuf,
    pub manifest: PathBuf,
    pub command: Vec<String>,
    pub artifact: MexArtifactManifest,
}

impl MexBuild {
    pub fn new(source: impl Into<PathBuf>, output_directory: impl Into<PathBuf>) -> Self {
        let source = source.into();
        let output_name = source
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("module")
            .to_string();
        let target = MexTarget::current().unwrap_or_else(|_| MexTarget {
            triple: target_lexicon::HOST.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            operating_system: std::env::consts::OS.to_string(),
            pointer_width: usize::BITS as u16,
            suffix: String::new(),
        });
        Self {
            compiler: super::default_c_compiler(),
            sources: vec![source],
            output_directory: output_directory.into(),
            output_name,
            api: MexApi::default(),
            include_directories: Vec::new(),
            definitions: Vec::new(),
            compiler_arguments: Vec::new(),
            linker_arguments: Vec::new(),
            target,
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

    pub fn api(mut self, api: MexApi) -> Self {
        self.api = api;
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

    pub fn target(mut self, target: MexTarget) -> Self {
        self.target = target;
        self
    }

    pub fn plan(&self) -> Result<MexBuildPlan, MexBuildError> {
        MexBuildPlan::for_build(self)
    }

    pub fn compile(&self) -> Result<MexBuildOutput, MexBuildError> {
        let plan = self.plan()?;
        if !plan.target.is_current() {
            return Err(MexBuildError::CrossCompilationUnavailable {
                triple: plan.target.triple.clone(),
            });
        }
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
        let module_bytes =
            std::fs::read(&plan.module).map_err(|source| MexBuildError::ReadCompiledModule {
                module: plan.module.clone(),
                source,
            })?;
        let artifact = MexArtifactManifest::from_module(
            &self.output_name,
            plan.target.clone(),
            self.api,
            compiler_family(&plan.compiler),
            &module_bytes,
        )?;
        let manifest = MexArtifactManifest::path_for_module(&plan.module);
        publish_manifest(&manifest, &artifact.canonical_bytes()?)?;
        Ok(MexBuildOutput {
            module: plan.module,
            manifest,
            command,
            artifact,
        })
    }
}

fn publish_manifest(path: &std::path::Path, bytes: &[u8]) -> Result<(), MexBuildError> {
    use std::io::Write as _;

    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|source| {
        MexBuildError::WriteArtifactManifest {
            path: path.to_path_buf(),
            source,
        }
    })?;
    temporary
        .write_all(bytes)
        .and_then(|_| temporary.as_file_mut().sync_all())
        .map_err(|source| MexBuildError::WriteArtifactManifest {
            path: path.to_path_buf(),
            source,
        })?;
    temporary
        .persist(path)
        .map_err(|error| MexBuildError::WriteArtifactManifest {
            path: path.to_path_buf(),
            source: error.error,
        })?;
    Ok(())
}
