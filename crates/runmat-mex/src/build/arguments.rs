use std::path::{Path, PathBuf};

use thiserror::Error;

use super::{MexApi, MexBuild, MexBuildOutput};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MexArgumentError {
    #[error("mex: expected at least one C, C++, or Fortran source file")]
    MissingSource,
    #[error("mex: option {0} requires a value")]
    MissingOptionValue(String),
    #[error("mex: unsupported option {0}")]
    UnsupportedOption(String),
    #[error(
        "mex: -setup is not interactive; set CC/CXX/FC or pass --compiler to select a toolchain"
    )]
    SetupUnsupported,
    #[error("mex: API options {first} and {second} cannot be combined")]
    ConflictingApiOptions { first: String, second: String },
}

#[derive(Debug, Clone)]
pub struct MexBuildInvocation {
    pub build: MexBuild,
    pub verbose: bool,
}

impl MexBuildInvocation {
    pub fn parse(arguments: &[String], working_directory: &Path) -> Result<Self, MexArgumentError> {
        let mut sources = Vec::new();
        let mut output_name = None;
        let mut output_directory = working_directory.to_path_buf();
        let mut c_compiler = None;
        let mut cxx_compiler = None;
        let mut fortran_compiler = None;
        let mut api = None;
        let mut include_directories = Vec::new();
        let mut definitions = Vec::new();
        let mut compiler_arguments = Vec::new();
        let mut c_compiler_arguments = Vec::new();
        let mut cxx_compiler_arguments = Vec::new();
        let mut fortran_compiler_arguments = Vec::new();
        let mut linker_arguments = Vec::new();
        let mut verbose = false;
        let mut index = 0;
        while index < arguments.len() {
            let argument = &arguments[index];
            match argument.as_str() {
                "-setup" => return Err(MexArgumentError::SetupUnsupported),
                "-R2017b" => select_api(&mut api, MexApi::R2017b, argument)?,
                "-R2018a" => select_api(&mut api, MexApi::R2018a, argument)?,
                "-largeArrayDims" => select_api(&mut api, MexApi::LargeArrayDims, argument)?,
                "-compatibleArrayDims" => {
                    select_api(&mut api, MexApi::CompatibleArrayDims, argument)?
                }
                "-v" | "-verbose" => verbose = true,
                "-silent" => verbose = false,
                "-output" | "-o" => {
                    output_name = Some(next_value(arguments, &mut index, argument)?);
                }
                "-outdir" => {
                    output_directory = resolve_path(
                        working_directory,
                        &next_value(arguments, &mut index, argument)?,
                    );
                }
                "-I" => include_directories.push(resolve_path(
                    working_directory,
                    &next_value(arguments, &mut index, argument)?,
                )),
                "-D" => definitions.push(next_value(arguments, &mut index, argument)?),
                _ if argument.starts_with("-I") && argument.len() > 2 => {
                    include_directories.push(resolve_path(working_directory, &argument[2..]));
                }
                _ if argument.starts_with("-D") && argument.len() > 2 => {
                    definitions.push(argument[2..].to_string());
                }
                _ if argument.starts_with("-L") || argument.starts_with("-l") => {
                    linker_arguments.push(argument.clone());
                }
                _ if argument.starts_with('-') => {
                    compiler_arguments.push(argument.clone());
                }
                _ if argument.starts_with("CC=") => {
                    c_compiler = Some(PathBuf::from(&argument[3..]));
                }
                _ if argument.starts_with("CXX=") => {
                    cxx_compiler = Some(PathBuf::from(&argument[4..]));
                }
                _ if argument.starts_with("FC=") => {
                    fortran_compiler = Some(PathBuf::from(&argument[3..]));
                }
                _ if argument.starts_with("F77=") => {
                    fortran_compiler = Some(PathBuf::from(&argument[4..]));
                }
                _ if argument.starts_with("CFLAGS=") => {
                    c_compiler_arguments.extend(split_driver_arguments(&argument[7..]));
                }
                _ if argument.starts_with("CXXFLAGS=") => {
                    cxx_compiler_arguments.extend(split_driver_arguments(&argument[9..]));
                }
                _ if argument.starts_with("FFLAGS=") => {
                    fortran_compiler_arguments.extend(split_driver_arguments(&argument[7..]));
                }
                _ if argument.starts_with("LDFLAGS=") => {
                    linker_arguments.extend(split_driver_arguments(&argument[8..]));
                }
                _ if argument.contains('=') => {
                    return Err(MexArgumentError::UnsupportedOption(argument.clone()));
                }
                _ => sources.push(resolve_path(working_directory, argument)),
            }
            index += 1;
        }
        let mut sources = sources.into_iter();
        let first = sources.next().ok_or(MexArgumentError::MissingSource)?;
        let mut build = MexBuild::new(first, output_directory);
        if let Some((api, _)) = api {
            build = build.api(api);
        }
        for source in sources {
            build = build.source(source);
        }
        if let Some(output_name) = output_name {
            build = build.output_name(output_name);
        }
        if let Some(compiler) = c_compiler {
            build = build.c_compiler(compiler);
        }
        if let Some(compiler) = cxx_compiler {
            build = build.cxx_compiler(compiler);
        }
        if let Some(compiler) = fortran_compiler {
            build = build.fortran_compiler(compiler);
        }
        for directory in include_directories {
            build = build.include_directory(directory);
        }
        for definition in definitions {
            build = build.define(definition);
        }
        for argument in compiler_arguments {
            build = build.compiler_argument(argument);
        }
        for argument in c_compiler_arguments {
            build = build.c_compiler_argument(argument);
        }
        for argument in cxx_compiler_arguments {
            build = build.cxx_compiler_argument(argument);
        }
        for argument in fortran_compiler_arguments {
            build = build.fortran_compiler_argument(argument);
        }
        for argument in linker_arguments {
            build = build.linker_argument(argument);
        }
        Ok(Self { build, verbose })
    }

    pub fn compile(&self) -> Result<MexBuildOutput, super::MexBuildError> {
        self.build.compile()
    }
}

fn select_api(
    selected: &mut Option<(MexApi, String)>,
    api: MexApi,
    spelling: &str,
) -> Result<(), MexArgumentError> {
    if let Some((_, first)) = selected {
        return Err(MexArgumentError::ConflictingApiOptions {
            first: first.clone(),
            second: spelling.to_string(),
        });
    }
    *selected = Some((api, spelling.to_string()));
    Ok(())
}

fn next_value(
    arguments: &[String],
    index: &mut usize,
    option: &str,
) -> Result<String, MexArgumentError> {
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| MexArgumentError::MissingOptionValue(option.to_string()))
}

fn resolve_path(working_directory: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        working_directory.join(path)
    }
}

fn split_driver_arguments(value: &str) -> impl Iterator<Item = String> + '_ {
    value.split_whitespace().map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_matlab_style_build_arguments_into_one_plan() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("gateway.c");
        let support = temporary.path().join("support.c");
        std::fs::write(&source, "void mexFunction(void) {}").unwrap();
        std::fs::write(&support, "void support(void) {}").unwrap();
        let arguments = vec![
            "-R2017b".into(),
            "-Iinclude".into(),
            "-DFEATURE=1".into(),
            "-output".into(),
            "native_gateway".into(),
            "gateway.c".into(),
            "support.c".into(),
        ];
        let invocation = MexBuildInvocation::parse(&arguments, temporary.path()).unwrap();
        let plan = invocation.build.plan().unwrap();
        assert!(plan
            .arguments
            .iter()
            .any(|argument| argument.contains("FEATURE=1")));
        assert!(plan
            .arguments
            .iter()
            .any(|argument| argument == &support.display().to_string()));
        assert!(plan.module.ends_with(format!(
            "native_gateway.{}",
            super::super::mex_suffix().unwrap()
        )));
    }

    #[test]
    fn release_api_pins_are_mutually_exclusive() {
        let arguments = vec!["-R2017b".into(), "-R2018a".into(), "gateway.c".into()];
        let error = MexBuildInvocation::parse(&arguments, Path::new(".")).unwrap_err();
        assert_eq!(
            error,
            MexArgumentError::ConflictingApiOptions {
                first: "-R2017b".into(),
                second: "-R2018a".into(),
            }
        );
    }

    #[test]
    fn cpp_sources_select_the_modern_api_and_cxx_overrides() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("gateway.cpp");
        std::fs::write(&source, "class MexFunction {}; ").unwrap();
        let arguments = vec![
            "CXX=custom-cxx".into(),
            "CXXFLAGS=-g -fno-exceptions".into(),
            "gateway.cpp".into(),
        ];
        let invocation = MexBuildInvocation::parse(&arguments, temporary.path()).unwrap();
        assert_eq!(
            invocation.build.language,
            crate::build::MexSourceLanguage::Cxx
        );
        assert_eq!(invocation.build.api, MexApi::R2018a);
        assert_eq!(invocation.build.compiler, PathBuf::from("custom-cxx"));
        assert!(invocation
            .build
            .cxx_compiler_arguments
            .contains(&"-fno-exceptions".to_string()));
    }

    #[test]
    fn fortran_sources_select_fc_flags_and_keep_the_default_api_pin() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("gateway.F90");
        std::fs::write(&source, "subroutine mexFunction\nend").unwrap();
        let arguments = vec![
            "FC=custom-fortran".into(),
            "FFLAGS=-g -fcheck=bounds".into(),
            "gateway.F90".into(),
        ];
        let invocation = MexBuildInvocation::parse(&arguments, temporary.path()).unwrap();
        assert_eq!(
            invocation.build.language,
            crate::build::MexSourceLanguage::Fortran
        );
        assert_eq!(invocation.build.api, MexApi::R2017b);
        assert_eq!(invocation.build.compiler, PathBuf::from("custom-fortran"));
        assert!(invocation
            .build
            .fortran_compiler_arguments
            .contains(&"-fcheck=bounds".to_string()));
    }

    #[test]
    fn mixed_source_selection_is_order_independent_and_uses_language_specific_flags() {
        let temporary = tempfile::tempdir().unwrap();
        for name in ["helper.c", "gateway.F90"] {
            std::fs::write(temporary.path().join(name), "").unwrap();
        }
        for sources in [["helper.c", "gateway.F90"], ["gateway.F90", "helper.c"]] {
            let arguments = vec![
                "CC=custom-c".into(),
                "FC=gfortran-14".into(),
                "CFLAGS=-DC_ONLY".into(),
                "FFLAGS=-DFORTRAN_ONLY".into(),
                sources[0].into(),
                sources[1].into(),
            ];
            let invocation = MexBuildInvocation::parse(&arguments, temporary.path()).unwrap();
            assert_eq!(
                invocation.build.language,
                super::super::MexSourceLanguage::Fortran
            );
            assert_eq!(invocation.build.api, MexApi::R2017b);
            assert_eq!(invocation.build.compiler, PathBuf::from("gfortran-14"));
            assert_eq!(invocation.build.c_compiler, PathBuf::from("custom-c"));
            assert_eq!(invocation.build.c_compiler_arguments, ["-DC_ONLY"]);
            assert_eq!(
                invocation.build.fortran_compiler_arguments,
                ["-DFORTRAN_ONLY"]
            );
        }
    }
}
