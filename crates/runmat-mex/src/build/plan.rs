use std::path::{Path, PathBuf};

use super::{
    compiler_family, sdk, CCompilerFamily, MexApi, MexBuild, MexBuildError, MexSourceLanguage,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexBuildPlan {
    pub compiler: PathBuf,
    pub arguments: Vec<String>,
    pub steps: Vec<MexBuildStep>,
    pub module: PathBuf,
    pub target: super::MexTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexBuildStep {
    pub compiler: PathBuf,
    pub arguments: Vec<String>,
}

impl MexBuildPlan {
    pub(super) fn for_build(build: &MexBuild) -> Result<Self, MexBuildError> {
        Self::for_build_with_msvc_object_directory(build, None)
    }

    pub(super) fn for_build_with_msvc_object_directory(
        build: &MexBuild,
        object_directory: Option<&Path>,
    ) -> Result<Self, MexBuildError> {
        validate(build)?;
        build.target.validate()?;
        let family = compiler_family(&build.compiler);
        build.target.validate_compiler(family)?;
        let module = build
            .output_directory
            .join(format!("{}.{}", build.output_name, build.target.suffix));
        let function_name = c_identifier(&build.output_name);
        let sdk = sdk::prepare()?;
        let object_directory = object_directory.unwrap_or(&build.output_directory);
        let steps = match (family, build.language) {
            (CCompilerFamily::GnuLike, MexSourceLanguage::C) => vec![MexBuildStep {
                compiler: build.compiler.clone(),
                arguments: gnu_arguments(
                    build,
                    &module,
                    &function_name,
                    &sdk.include_directory,
                    &sdk.support_source,
                ),
            }],
            (CCompilerFamily::Msvc, MexSourceLanguage::C) => vec![MexBuildStep {
                compiler: build.compiler.clone(),
                arguments: msvc_arguments(
                    build,
                    &module,
                    &function_name,
                    &sdk.include_directory,
                    &sdk.support_source,
                    object_directory,
                ),
            }],
            (CCompilerFamily::GnuLike, MexSourceLanguage::Cxx) => cxx_gnu_steps(
                build,
                &module,
                &function_name,
                &sdk.include_directory,
                &sdk.support_source,
                object_directory,
            ),
            (CCompilerFamily::Msvc, MexSourceLanguage::Cxx) => cxx_msvc_steps(
                build,
                &module,
                &function_name,
                &sdk.include_directory,
                &sdk.support_source,
                object_directory,
            ),
            (CCompilerFamily::GnuLike, MexSourceLanguage::Fortran) => fortran_gnu_steps(
                build,
                &module,
                &function_name,
                &sdk.include_directory,
                &sdk.support_source,
                object_directory,
            ),
            (CCompilerFamily::Msvc, MexSourceLanguage::Fortran) => {
                return Err(MexBuildError::UnsupportedFortranCompiler {
                    compiler: build.compiler.clone(),
                });
            }
        };
        let arguments = flatten_step_arguments(&steps);
        Ok(Self {
            compiler: build.compiler.clone(),
            arguments,
            steps,
            module,
            target: build.target.clone(),
        })
    }

    pub fn command(&self) -> Vec<String> {
        flatten_steps(&self.steps)
    }
}

fn flatten_step_arguments(steps: &[MexBuildStep]) -> Vec<String> {
    steps
        .iter()
        .flat_map(|step| step.arguments.iter().cloned())
        .collect()
}

fn flatten_steps(steps: &[MexBuildStep]) -> Vec<String> {
    let mut command = Vec::new();
    for (index, step) in steps.iter().enumerate() {
        if index != 0 {
            command.push("&&".into());
        }
        command.push(step.compiler.display().to_string());
        command.extend(step.arguments.iter().cloned());
    }
    command
}

fn validate(build: &MexBuild) -> Result<(), MexBuildError> {
    if build.sources.is_empty() {
        return Err(MexBuildError::MissingSources);
    }
    for source in &build.sources {
        if !source.is_file() {
            return Err(MexBuildError::MissingSource(source.clone()));
        }
    }
    if build.output_name.trim().is_empty()
        || build.output_name.contains(['/', '\\', '\0'])
        || build.output_name == "."
        || build.output_name == ".."
    {
        return Err(MexBuildError::InvalidOutputName);
    }
    if build
        .definitions
        .iter()
        .any(|definition| definition.is_empty() || definition.contains('\0'))
    {
        return Err(MexBuildError::InvalidDefinition);
    }
    if build.language == MexSourceLanguage::Fortran
        && !super::is_supported_fortran_compiler(&build.compiler)
    {
        return Err(MexBuildError::UnsupportedFortranCompiler {
            compiler: build.compiler.clone(),
        });
    }
    Ok(())
}

fn gnu_arguments(
    build: &MexBuild,
    module: &Path,
    function_name: &str,
    sdk_include: &Path,
    support_source: &Path,
) -> Vec<String> {
    let mut arguments = Vec::new();
    arguments.push(if cfg!(target_os = "macos") {
        "-dynamiclib".into()
    } else {
        "-shared".into()
    });
    if !cfg!(target_os = "windows") {
        arguments.push("-fPIC".into());
    }
    arguments.extend([
        match build.language {
            MexSourceLanguage::C => "-std=c11",
            MexSourceLanguage::Cxx => "-std=c++17",
            MexSourceLanguage::Fortran => unreachable!("Fortran uses a multi-step build"),
        }
        .into(),
        "-O2".into(),
    ]);
    arguments.push(format!("-I{}", sdk_include.display()));
    for include in &build.include_directories {
        arguments.push(format!("-I{}", include.display()));
    }
    push_definitions(build, function_name, "-D", &mut arguments);
    arguments.extend(build.compiler_arguments.iter().cloned());
    arguments.extend(build.c_compiler_arguments.iter().cloned());
    if build.language == MexSourceLanguage::Cxx {
        for source in &build.sources {
            arguments.extend([
                "-x".into(),
                match MexSourceLanguage::detect(source) {
                    MexSourceLanguage::C => "c",
                    MexSourceLanguage::Cxx => "c++",
                    MexSourceLanguage::Fortran => "f95",
                }
                .into(),
                source.display().to_string(),
            ]);
        }
        arguments.extend([
            "-x".into(),
            "c++".into(),
            support_source.display().to_string(),
        ]);
    } else {
        arguments.extend(build.sources.iter().map(|path| path.display().to_string()));
        arguments.push(support_source.display().to_string());
    }
    arguments.extend(build.linker_arguments.iter().cloned());
    arguments.extend(["-o".into(), module.display().to_string()]);
    arguments
}

fn msvc_arguments(
    build: &MexBuild,
    module: &Path,
    function_name: &str,
    sdk_include: &Path,
    support_source: &Path,
    object_directory: &Path,
) -> Vec<String> {
    let mut arguments = vec![
        "/nologo".into(),
        "/LD".into(),
        "/O2".into(),
        match build.language {
            MexSourceLanguage::C => "/std:c11",
            MexSourceLanguage::Cxx => "/std:c++17",
            MexSourceLanguage::Fortran => unreachable!("Fortran does not use MSVC"),
        }
        .into(),
        format!("/I{}", sdk_include.display()),
        format!(
            "/Fo{}{}",
            object_directory.display(),
            std::path::MAIN_SEPARATOR
        ),
    ];
    for include in &build.include_directories {
        arguments.push(format!("/I{}", include.display()));
    }
    push_definitions(build, function_name, "/D", &mut arguments);
    arguments.extend(build.compiler_arguments.iter().cloned());
    arguments.extend(build.c_compiler_arguments.iter().cloned());
    if build.language == MexSourceLanguage::Cxx {
        arguments.extend(build.sources.iter().map(|path| {
            let mode = match MexSourceLanguage::detect(path) {
                MexSourceLanguage::C => "/TC",
                MexSourceLanguage::Cxx => "/TP",
                MexSourceLanguage::Fortran => unreachable!("Fortran does not use MSVC"),
            };
            format!("{mode}{}", path.display())
        }));
        arguments.push(format!("/TP{}", support_source.display()));
    } else {
        arguments.extend(build.sources.iter().map(|path| path.display().to_string()));
        arguments.push(support_source.display().to_string());
    }
    arguments.push("/link".into());
    arguments.extend(build.linker_arguments.iter().cloned());
    arguments.push(format!("/OUT:{}", module.display()));
    arguments
}

fn cxx_gnu_steps(
    build: &MexBuild,
    module: &Path,
    function_name: &str,
    sdk_include: &Path,
    support_source: &Path,
    object_directory: &Path,
) -> Vec<MexBuildStep> {
    let mut steps = Vec::new();
    for (index, source) in build.sources.iter().enumerate() {
        let language = MexSourceLanguage::detect(source);
        let object = object_path(object_directory, index, source, "o");
        let mut arguments = vec![
            "-fPIC".into(),
            "-O2".into(),
            "-pthread".into(),
            "-x".into(),
            match language {
                MexSourceLanguage::C => "c",
                MexSourceLanguage::Cxx => "c++",
                MexSourceLanguage::Fortran => "f95",
            }
            .into(),
            match language {
                MexSourceLanguage::C => "-std=c11",
                MexSourceLanguage::Cxx => "-std=c++17",
                MexSourceLanguage::Fortran => "-std=legacy",
            }
            .into(),
            format!("-I{}", sdk_include.display()),
        ];
        for include in &build.include_directories {
            arguments.push(format!("-I{}", include.display()));
        }
        push_definitions(build, function_name, "-D", &mut arguments);
        if language == MexSourceLanguage::Cxx {
            arguments.extend(build.compiler_arguments.iter().cloned());
        }
        match language {
            MexSourceLanguage::C => {
                arguments.extend(build.c_compiler_arguments.iter().cloned());
            }
            MexSourceLanguage::Cxx => {
                arguments.extend(build.cxx_compiler_arguments.iter().cloned());
            }
            MexSourceLanguage::Fortran => unreachable!("C++ builds do not compile Fortran"),
        }
        arguments.extend([
            "-c".into(),
            source.display().to_string(),
            "-o".into(),
            object.display().to_string(),
        ]);
        steps.push(MexBuildStep {
            compiler: match language {
                MexSourceLanguage::C => build.c_compiler.clone(),
                MexSourceLanguage::Cxx => build.compiler.clone(),
                MexSourceLanguage::Fortran => unreachable!("C++ builds do not compile Fortran"),
            },
            arguments,
        });
    }

    let support_object = object_path(object_directory, build.sources.len(), support_source, "o");
    let mut support_arguments = vec![
        "-fPIC".into(),
        "-O2".into(),
        "-x".into(),
        "c".into(),
        "-std=c11".into(),
        format!("-I{}", sdk_include.display()),
    ];
    push_definitions(build, function_name, "-D", &mut support_arguments);
    support_arguments.extend(build.c_compiler_arguments.iter().cloned());
    support_arguments.extend([
        "-c".into(),
        support_source.display().to_string(),
        "-o".into(),
        support_object.display().to_string(),
    ]);
    steps.push(MexBuildStep {
        compiler: build.c_compiler.clone(),
        arguments: support_arguments,
    });

    let mut link_arguments = vec![if cfg!(target_os = "macos") {
        "-dynamiclib".into()
    } else {
        "-shared".into()
    }];
    link_arguments.push("-pthread".into());
    link_arguments.extend((0..build.sources.len()).map(|index| {
        object_path(object_directory, index, &build.sources[index], "o")
            .display()
            .to_string()
    }));
    link_arguments.push(support_object.display().to_string());
    link_arguments.extend(build.linker_arguments.iter().cloned());
    link_arguments.extend(["-o".into(), module.display().to_string()]);
    steps.push(MexBuildStep {
        compiler: build.compiler.clone(),
        arguments: link_arguments,
    });
    steps
}

fn cxx_msvc_steps(
    build: &MexBuild,
    module: &Path,
    function_name: &str,
    sdk_include: &Path,
    support_source: &Path,
    object_directory: &Path,
) -> Vec<MexBuildStep> {
    let mut steps = Vec::new();
    for (index, source) in build.sources.iter().enumerate() {
        let language = MexSourceLanguage::detect(source);
        let object = object_path(object_directory, index, source, "obj");
        let mut arguments = vec![
            "/nologo".into(),
            "/O2".into(),
            "/c".into(),
            match language {
                MexSourceLanguage::C => "/std:c11",
                MexSourceLanguage::Cxx => "/std:c++17",
                MexSourceLanguage::Fortran => unreachable!("Fortran does not use MSVC"),
            }
            .into(),
            format!("/I{}", sdk_include.display()),
        ];
        for include in &build.include_directories {
            arguments.push(format!("/I{}", include.display()));
        }
        push_definitions(build, function_name, "/D", &mut arguments);
        if language == MexSourceLanguage::Cxx {
            arguments.extend(build.compiler_arguments.iter().cloned());
        }
        match language {
            MexSourceLanguage::C => {
                arguments.extend(build.c_compiler_arguments.iter().cloned());
            }
            MexSourceLanguage::Cxx => {
                arguments.extend(build.cxx_compiler_arguments.iter().cloned());
            }
            MexSourceLanguage::Fortran => unreachable!("C++ builds do not compile Fortran"),
        }
        arguments.push(format!(
            "{}{}",
            match language {
                MexSourceLanguage::C => "/TC",
                MexSourceLanguage::Cxx => "/TP",
                MexSourceLanguage::Fortran => unreachable!("Fortran does not use MSVC"),
            },
            source.display()
        ));
        arguments.push(format!("/Fo{}", object.display()));
        steps.push(MexBuildStep {
            compiler: match language {
                MexSourceLanguage::C => build.c_compiler.clone(),
                MexSourceLanguage::Cxx => build.compiler.clone(),
                MexSourceLanguage::Fortran => unreachable!("C++ builds do not compile Fortran"),
            },
            arguments,
        });
    }

    let support_object = object_path(object_directory, build.sources.len(), support_source, "obj");
    let mut support_arguments = vec![
        "/nologo".into(),
        "/O2".into(),
        "/c".into(),
        "/std:c11".into(),
        format!("/I{}", sdk_include.display()),
    ];
    push_definitions(build, function_name, "/D", &mut support_arguments);
    support_arguments.extend(build.c_compiler_arguments.iter().cloned());
    support_arguments.extend([
        format!("/TC{}", support_source.display()),
        format!("/Fo{}", support_object.display()),
    ]);
    steps.push(MexBuildStep {
        compiler: build.c_compiler.clone(),
        arguments: support_arguments,
    });

    let mut link_arguments = vec!["/nologo".into(), "/LD".into()];
    link_arguments.extend((0..build.sources.len()).map(|index| {
        object_path(object_directory, index, &build.sources[index], "obj")
            .display()
            .to_string()
    }));
    link_arguments.push(support_object.display().to_string());
    link_arguments.push("/link".into());
    link_arguments.extend(build.linker_arguments.iter().cloned());
    link_arguments.push(format!("/OUT:{}", module.display()));
    steps.push(MexBuildStep {
        compiler: build.compiler.clone(),
        arguments: link_arguments,
    });
    steps
}

fn fortran_gnu_steps(
    build: &MexBuild,
    module: &Path,
    function_name: &str,
    sdk_include: &Path,
    support_source: &Path,
    object_directory: &Path,
) -> Vec<MexBuildStep> {
    let mut steps = Vec::new();
    let c_compiler = build.c_compiler.clone();
    let cxx_compiler = build.cxx_compiler.clone();
    let mut objects = Vec::new();
    for (index, source) in build.sources.iter().enumerate() {
        let language = MexSourceLanguage::detect(source);
        let object = object_path(object_directory, index, source, "o");
        let (compiler, mut arguments) = match language {
            MexSourceLanguage::Fortran => (
                build.compiler.clone(),
                vec![
                    "-fPIC".into(),
                    "-O2".into(),
                    "-cpp".into(),
                    "-std=legacy".into(),
                    if fortran_uses_fixed_form(source) {
                        "-ffixed-line-length-none".into()
                    } else {
                        "-ffree-line-length-none".into()
                    },
                    format!("-I{}", sdk_include.display()),
                ],
            ),
            MexSourceLanguage::C => (
                c_compiler.clone(),
                vec![
                    "-fPIC".into(),
                    "-O2".into(),
                    "-std=c11".into(),
                    format!("-I{}", sdk_include.display()),
                ],
            ),
            MexSourceLanguage::Cxx => (
                cxx_compiler.clone(),
                vec![
                    "-fPIC".into(),
                    "-O2".into(),
                    "-std=c++17".into(),
                    format!("-I{}", sdk_include.display()),
                ],
            ),
        };
        for include in &build.include_directories {
            arguments.push(format!("-I{}", include.display()));
        }
        push_definitions(build, function_name, "-D", &mut arguments);
        match language {
            MexSourceLanguage::Fortran => {
                arguments.extend(build.compiler_arguments.iter().cloned());
                arguments.extend(build.fortran_compiler_arguments.iter().cloned());
            }
            MexSourceLanguage::C => {
                arguments.extend(build.c_compiler_arguments.iter().cloned());
            }
            MexSourceLanguage::Cxx => {
                arguments.extend(build.cxx_compiler_arguments.iter().cloned());
            }
        }
        arguments.extend([
            "-c".into(),
            source.display().to_string(),
            "-o".into(),
            object.display().to_string(),
        ]);
        objects.push(object);
        steps.push(MexBuildStep {
            compiler,
            arguments,
        });
    }

    let support_object = object_path(object_directory, build.sources.len(), support_source, "o");
    let mut support_arguments = vec![
        "-fPIC".into(),
        "-O2".into(),
        "-std=c11".into(),
        format!("-I{}", sdk_include.display()),
        "-DRUNMAT_MEX_FORTRAN=1".into(),
    ];
    push_definitions(build, function_name, "-D", &mut support_arguments);
    support_arguments.extend(build.c_compiler_arguments.iter().cloned());
    support_arguments.extend([
        "-c".into(),
        support_source.display().to_string(),
        "-o".into(),
        support_object.display().to_string(),
    ]);
    objects.push(support_object);
    steps.push(MexBuildStep {
        compiler: c_compiler,
        arguments: support_arguments,
    });

    let mut link_arguments = vec![if cfg!(target_os = "macos") {
        "-dynamiclib".into()
    } else {
        "-shared".into()
    }];
    link_arguments.extend(objects.iter().map(|path| path.display().to_string()));
    if build
        .sources
        .iter()
        .any(|source| MexSourceLanguage::detect(source) == MexSourceLanguage::Cxx)
    {
        link_arguments.push(if cfg!(target_os = "macos") {
            "-lc++".into()
        } else {
            "-lstdc++".into()
        });
    }
    link_arguments.extend(build.linker_arguments.iter().cloned());
    link_arguments.extend(["-o".into(), module.display().to_string()]);
    steps.push(MexBuildStep {
        compiler: build.compiler.clone(),
        arguments: link_arguments,
    });
    steps
}

fn fortran_uses_fixed_form(source: &Path) -> bool {
    matches!(
        source
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("f" | "for" | "f77")
    )
}

fn object_path(directory: &Path, index: usize, source: &Path, extension: &str) -> PathBuf {
    let stem = source
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("source")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    directory.join(format!("{index:04}-{stem}.{extension}"))
}

fn push_definitions(
    build: &MexBuild,
    function_name: &str,
    prefix: &str,
    arguments: &mut Vec<String>,
) {
    arguments.push(format!("{prefix}MATLAB_MEX_FILE=1"));
    match build.api {
        MexApi::R2017b => {
            arguments.push(format!("{prefix}MEX_DOUBLE_HANDLE=1"));
            arguments.push(format!("{prefix}TARGET_API_VERSION=700"));
        }
        MexApi::R2018a => {
            arguments.push(format!("{prefix}RUNMAT_MX_INTERLEAVED_COMPLEX=1"));
            arguments.push(format!("{prefix}TARGET_API_VERSION=800"));
        }
        MexApi::CompatibleArrayDims => {
            arguments.push(format!("{prefix}RUNMAT_MX_COMPATIBLE_ARRAY_DIMS=1"));
            arguments.push(format!("{prefix}MX_COMPAT_32=1"));
            arguments.push(format!("{prefix}TARGET_API_VERSION=700"));
        }
        MexApi::LargeArrayDims => {
            arguments.push(format!("{prefix}TARGET_API_VERSION=700"));
        }
    }
    arguments.push(format!(
        "{prefix}RUNMAT_MEX_FUNCTION_NAME=\"{function_name}\""
    ));
    if build.language == MexSourceLanguage::Fortran {
        let gateway = fortran_gateway_name(function_name);
        arguments.push(format!("{prefix}RUNMAT_MEX_FORTRAN_GATEWAY={gateway}"));
    }
    arguments.extend(
        build
            .definitions
            .iter()
            .map(|definition| format!("{prefix}{definition}")),
    );
}

fn fortran_gateway_name(function_name: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in function_name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("rmf_{hash:016x}")
}

fn c_identifier(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MexTarget;

    #[test]
    fn fortran_plan_compiles_language_and_support_units_then_links_with_fortran() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("demo.F");
        std::fs::write(
            &source,
            "#include \"fintrf.h\"\n      subroutine mexFunction(n,p,m,q)\n      end",
        )
        .unwrap();
        let build = MexBuild::new(&source, temporary.path()).compiler("gfortran");
        let plan = build.plan().unwrap();
        assert_eq!(build.language, MexSourceLanguage::Fortran);
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.steps[0].compiler, PathBuf::from("gfortran"));
        assert!(plan.steps[0]
            .arguments
            .contains(&"-ffixed-line-length-none".to_string()));
        assert!(plan.steps[0]
            .arguments
            .iter()
            .any(|argument| argument.contains("RUNMAT_MEX_FORTRAN_GATEWAY=rmf_")));
        assert!(plan.steps[1]
            .arguments
            .contains(&"-DRUNMAT_MEX_FORTRAN=1".to_string()));
        assert_eq!(plan.steps[2].compiler, PathBuf::from("gfortran"));
        assert!(plan.steps[2]
            .arguments
            .iter()
            .any(|argument| argument.ends_with("runmat_mex_support.o")));
    }

    #[test]
    fn fortran_plan_rejects_an_unqualified_driver_before_launch() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("demo.F90");
        std::fs::write(&source, "subroutine mexFunction\nend").unwrap();
        let error = MexBuild::new(&source, temporary.path())
            .compiler("ifort")
            .plan()
            .unwrap_err();
        assert!(matches!(
            error,
            MexBuildError::UnsupportedFortranCompiler { compiler }
                if compiler == PathBuf::from("ifort")
        ));
    }

    #[test]
    fn msvc_plan_uses_native_driver_spelling() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("demo.c");
        std::fs::write(&source, "void mexFunction(void) {}").unwrap();
        let build = MexBuild::new(&source, temporary.path())
            .target(MexTarget {
                triple: "x86_64-pc-windows-msvc".into(),
                architecture: "x86_64".into(),
                operating_system: "windows".into(),
                pointer_width: 64,
                suffix: "mexw64".into(),
            })
            .compiler("cl.exe")
            .include_directory("vendor/include")
            .define("FEATURE=1")
            .linker_argument("vendor.lib");
        let plan = build.plan().unwrap();
        assert!(plan.arguments.contains(&"/LD".to_string()));
        assert!(plan.arguments.contains(&"/Ivendor/include".to_string()));
        assert!(plan.arguments.contains(&"/DFEATURE=1".to_string()));
        assert!(plan
            .arguments
            .iter()
            .any(|argument| argument.starts_with("/Fo")));
        assert!(plan.arguments.contains(&"/link".to_string()));
        assert!(plan.arguments.contains(&"vendor.lib".to_string()));
        assert!(plan
            .arguments
            .iter()
            .any(|argument| argument.starts_with("/OUT:")));
    }

    #[test]
    fn api_pins_emit_distinct_preprocessor_contracts() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("demo.c");
        std::fs::write(&source, "void mexFunction(void) {}").unwrap();

        let definitions = |api| {
            MexBuild::new(&source, temporary.path())
                .api(api)
                .plan()
                .unwrap()
                .arguments
                .into_iter()
                .filter_map(|argument| {
                    argument
                        .strip_prefix("-D")
                        .or_else(|| argument.strip_prefix("/D"))
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>()
        };

        let default = definitions(MexApi::R2017b);
        assert!(default.contains(&"MATLAB_MEX_FILE=1".to_string()));
        assert!(default.contains(&"MEX_DOUBLE_HANDLE=1".to_string()));
        assert!(default.contains(&"TARGET_API_VERSION=700".to_string()));
        assert!(!default
            .iter()
            .any(|argument| argument.contains("INTERLEAVED")));

        let interleaved = definitions(MexApi::R2018a);
        assert!(interleaved.contains(&"RUNMAT_MX_INTERLEAVED_COMPLEX=1".to_string()));
        assert!(interleaved.contains(&"TARGET_API_VERSION=800".to_string()));

        let compatible = definitions(MexApi::CompatibleArrayDims);
        assert!(compatible.contains(&"RUNMAT_MX_COMPATIBLE_ARRAY_DIMS=1".to_string()));
        assert!(compatible.contains(&"MX_COMPAT_32=1".to_string()));
        assert!(compatible.contains(&"TARGET_API_VERSION=700".to_string()));
    }

    #[test]
    fn cpp_plan_uses_the_cxx_driver_contract_and_keeps_native_support_in_c_mode() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("demo.cpp");
        std::fs::write(&source, "class MexFunction {}; ").unwrap();
        let plan = MexBuild::new(&source, temporary.path()).plan().unwrap();
        assert!(plan.arguments.contains(&"-std=c++17".to_string()));
        assert!(plan.arguments.contains(&"c++".to_string()));
        assert!(plan.arguments.contains(&"-std=c11".to_string()));
        assert!(plan
            .arguments
            .iter()
            .any(|argument| argument == "RUNMAT_MX_INTERLEAVED_COMPLEX=1"
                || argument.ends_with("RUNMAT_MX_INTERLEAVED_COMPLEX=1")));
    }

    #[test]
    fn msvc_cpp_plan_compiles_each_language_then_links_once() {
        let temporary = tempfile::tempdir().unwrap();
        let cpp = temporary.path().join("gateway.cpp");
        let c = temporary.path().join("support.c");
        std::fs::write(&cpp, "class MexFunction {}; ").unwrap();
        std::fs::write(&c, "void support(void) {} ").unwrap();
        let build = MexBuild::new(&cpp, temporary.path())
            .source(&c)
            .compiler("cl.exe")
            .target(MexTarget {
                triple: "x86_64-pc-windows-msvc".into(),
                architecture: "x86_64".into(),
                operating_system: "windows".into(),
                pointer_width: 64,
                suffix: "mexw64".into(),
            });
        let plan =
            MexBuildPlan::for_build_with_msvc_object_directory(&build, Some(temporary.path()))
                .unwrap();
        assert_eq!(plan.steps.len(), 4);
        assert!(plan.steps[0]
            .arguments
            .iter()
            .any(|argument| argument.starts_with("/TP")));
        assert!(plan.steps[1]
            .arguments
            .iter()
            .any(|argument| argument.starts_with("/TC")));
        assert!(plan.steps[2]
            .arguments
            .iter()
            .any(|argument| argument.starts_with("/TC")));
        assert!(plan.steps[3].arguments.contains(&"/LD".to_string()));
    }

    #[test]
    fn target_rejects_an_incompatible_compiler_family_before_launch() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("demo.c");
        std::fs::write(&source, "void mexFunction(void) {}").unwrap();
        let target = MexTarget::current().unwrap();
        let incompatible_compiler = if target.operating_system == "windows" {
            "cc"
        } else {
            "cl.exe"
        };

        let error = MexBuild::new(&source, temporary.path())
            .compiler(incompatible_compiler)
            .plan()
            .unwrap_err();
        assert!(matches!(
            error,
            MexBuildError::UnsupportedCompilerForTarget { .. }
        ));
    }
}
