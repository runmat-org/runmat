use std::path::PathBuf;
use std::process::Command;

use super::{compiler_family, MexArtifactManifest, MexBuildError, MexBuildPlan, MexTarget};

/// C Matrix API selected for a MEX build.
///
/// The release pins select the complex representation as well as the array
/// dimension API. The legacy spellings remain distinct because the four
/// choices are mutually exclusive command-line API selections.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MexApi {
    /// Separate-complex, large-array API.
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

/// Source-language ABI selected by the gateway translation units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MexSourceLanguage {
    C,
    Cxx,
    Fortran,
    Cuda,
}

impl MexSourceLanguage {
    pub(super) fn detect(path: &std::path::Path) -> Self {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("cc" | "cpp" | "cxx" | "c++") => Self::Cxx,
            Some("f" | "for" | "f77" | "f90" | "f95" | "f03" | "f08") => Self::Fortran,
            Some("cu") => Self::Cuda,
            _ => Self::C,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MexBuild {
    pub(super) compiler: PathBuf,
    pub(super) compiler_explicit: bool,
    pub(super) c_compiler: PathBuf,
    pub(super) cxx_compiler: PathBuf,
    pub(super) fortran_compiler: PathBuf,
    pub(super) cuda_compiler: PathBuf,
    pub(super) sources: Vec<PathBuf>,
    pub(super) output_directory: PathBuf,
    pub(super) output_name: String,
    pub(super) language: MexSourceLanguage,
    pub(super) api: MexApi,
    pub(super) api_explicit: bool,
    pub(super) include_directories: Vec<PathBuf>,
    pub(super) definitions: Vec<String>,
    pub(super) compiler_arguments: Vec<String>,
    pub(super) c_compiler_arguments: Vec<String>,
    pub(super) cxx_compiler_arguments: Vec<String>,
    pub(super) fortran_compiler_arguments: Vec<String>,
    pub(super) cuda_compiler_arguments: Vec<String>,
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
    pub const fn language(&self) -> MexSourceLanguage {
        self.language
    }

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
        let language = MexSourceLanguage::detect(&source);
        let c_compiler = super::default_c_compiler();
        let cxx_compiler = super::default_cxx_compiler();
        let fortran_compiler = super::default_fortran_compiler();
        let cuda_compiler = super::default_cuda_compiler();
        Self {
            compiler: match language {
                MexSourceLanguage::C => c_compiler.clone(),
                MexSourceLanguage::Cxx => cxx_compiler.clone(),
                MexSourceLanguage::Fortran => fortran_compiler.clone(),
                MexSourceLanguage::Cuda => cuda_compiler.clone(),
            },
            compiler_explicit: false,
            c_compiler,
            cxx_compiler,
            fortran_compiler,
            cuda_compiler,
            sources: vec![source],
            output_directory: output_directory.into(),
            output_name,
            language,
            api: match language {
                MexSourceLanguage::C => MexApi::default(),
                MexSourceLanguage::Cxx => MexApi::R2018a,
                MexSourceLanguage::Fortran => MexApi::default(),
                MexSourceLanguage::Cuda => MexApi::R2018a,
            },
            api_explicit: false,
            include_directories: Vec::new(),
            definitions: Vec::new(),
            compiler_arguments: Vec::new(),
            c_compiler_arguments: Vec::new(),
            cxx_compiler_arguments: Vec::new(),
            fortran_compiler_arguments: Vec::new(),
            cuda_compiler_arguments: Vec::new(),
            linker_arguments: Vec::new(),
            target,
        }
    }

    pub fn compiler(mut self, compiler: impl Into<PathBuf>) -> Self {
        self.compiler = compiler.into();
        if self.language == MexSourceLanguage::Cuda {
            self.cuda_compiler = self.compiler.clone();
        }
        self.compiler_explicit = true;
        self
    }

    pub fn c_compiler(mut self, compiler: impl Into<PathBuf>) -> Self {
        self.c_compiler = compiler.into();
        if self.language == MexSourceLanguage::C && !self.compiler_explicit {
            self.compiler = self.c_compiler.clone();
        }
        self
    }

    pub fn cxx_compiler(mut self, compiler: impl Into<PathBuf>) -> Self {
        self.cxx_compiler = compiler.into();
        if self.language == MexSourceLanguage::Cxx && !self.compiler_explicit {
            self.compiler = self.cxx_compiler.clone();
        }
        self
    }

    pub fn fortran_compiler(mut self, compiler: impl Into<PathBuf>) -> Self {
        self.fortran_compiler = compiler.into();
        if self.language == MexSourceLanguage::Fortran && !self.compiler_explicit {
            self.compiler = self.fortran_compiler.clone();
        }
        self
    }

    pub fn cuda_compiler(mut self, compiler: impl Into<PathBuf>) -> Self {
        self.cuda_compiler = compiler.into();
        if self.language == MexSourceLanguage::Cuda && !self.compiler_explicit {
            self.compiler = self.cuda_compiler.clone();
        }
        self
    }

    pub fn source(mut self, source: impl Into<PathBuf>) -> Self {
        let source = source.into();
        let source_language = MexSourceLanguage::detect(&source);
        let selected_language = match (self.language, source_language) {
            (MexSourceLanguage::Cuda, MexSourceLanguage::Fortran)
            | (MexSourceLanguage::Fortran, MexSourceLanguage::Cuda) => self.language,
            (MexSourceLanguage::Cuda, _) | (_, MexSourceLanguage::Cuda) => MexSourceLanguage::Cuda,
            (MexSourceLanguage::Fortran, _) | (_, MexSourceLanguage::Fortran) => {
                MexSourceLanguage::Fortran
            }
            (MexSourceLanguage::Cxx, _) | (_, MexSourceLanguage::Cxx) => MexSourceLanguage::Cxx,
            _ => MexSourceLanguage::C,
        };
        if selected_language != self.language {
            self.language = selected_language;
            if !self.compiler_explicit {
                self.compiler = match selected_language {
                    MexSourceLanguage::C => self.c_compiler.clone(),
                    MexSourceLanguage::Cxx => self.cxx_compiler.clone(),
                    MexSourceLanguage::Fortran => self.fortran_compiler.clone(),
                    MexSourceLanguage::Cuda => self.cuda_compiler.clone(),
                };
            }
            if !self.api_explicit {
                self.api = match selected_language {
                    MexSourceLanguage::Cxx => MexApi::R2018a,
                    MexSourceLanguage::C | MexSourceLanguage::Fortran => MexApi::R2017b,
                    MexSourceLanguage::Cuda => MexApi::R2018a,
                };
            }
        }
        self.sources.push(source);
        self
    }

    pub fn output_name(mut self, name: impl Into<String>) -> Self {
        self.output_name = name.into();
        self
    }

    pub fn api(mut self, api: MexApi) -> Self {
        self.api = api;
        self.api_explicit = true;
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

    pub fn c_compiler_argument(mut self, argument: impl Into<String>) -> Self {
        self.c_compiler_arguments.push(argument.into());
        self
    }

    pub fn cxx_compiler_argument(mut self, argument: impl Into<String>) -> Self {
        self.cxx_compiler_arguments.push(argument.into());
        self
    }

    pub fn fortran_compiler_argument(mut self, argument: impl Into<String>) -> Self {
        self.fortran_compiler_arguments.push(argument.into());
        self
    }

    pub fn cuda_compiler_argument(mut self, argument: impl Into<String>) -> Self {
        self.cuda_compiler_arguments.push(argument.into());
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
        let initial_plan = self.plan()?;
        if !initial_plan.target.is_current() {
            return Err(MexBuildError::CrossCompilationUnavailable {
                triple: initial_plan.target.triple.clone(),
            });
        }
        std::fs::create_dir_all(&self.output_directory).map_err(|source| {
            MexBuildError::CreateOutputDirectory {
                directory: self.output_directory.clone(),
                source,
            }
        })?;
        let object_directory = if compiler_family(&self.compiler) == super::CCompilerFamily::Msvc
            || self.language != MexSourceLanguage::C
        {
            Some(
                tempfile::Builder::new()
                    .prefix(".runmat-mex-objects-")
                    .tempdir_in(&self.output_directory)
                    .map_err(|source| MexBuildError::CreateOutputDirectory {
                        directory: self.output_directory.clone(),
                        source,
                    })?,
            )
        } else {
            None
        };
        let plan = if let Some(directory) = object_directory.as_ref() {
            MexBuildPlan::for_build_with_msvc_object_directory(self, Some(directory.path()))?
        } else {
            initial_plan
        };
        let command = plan.command();
        for step in &plan.steps {
            let output = Command::new(&step.compiler)
                .args(&step.arguments)
                .output()
                .map_err(|source| MexBuildError::CompilerLaunch {
                    compiler: step.compiler.clone(),
                    source,
                })?;
            if !output.status.success() {
                let diagnostics = format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                let failed_command = std::iter::once(step.compiler.display().to_string())
                    .chain(step.arguments.iter().cloned())
                    .collect::<Vec<_>>()
                    .join(" ");
                return Err(MexBuildError::CompilerFailure {
                    command: failed_command,
                    diagnostics,
                });
            }
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
            self.language,
            if self.language == MexSourceLanguage::Cuda {
                compiler_family(&self.c_compiler)
            } else {
                compiler_family(&plan.compiler)
            },
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
