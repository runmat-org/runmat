use std::path::{Path, PathBuf};

use thiserror::Error;

use super::{MexApi, MexBuild, MexBuildOutput};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MexArgumentError {
    #[error("mex: expected at least one C source file")]
    MissingSource,
    #[error("mex: option {0} requires a value")]
    MissingOptionValue(String),
    #[error("mex: unsupported option {0}")]
    UnsupportedOption(String),
    #[error("mex: -setup is not interactive; set CC or pass --compiler to select a toolchain")]
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
        let mut compiler = None;
        let mut api = None;
        let mut include_directories = Vec::new();
        let mut definitions = Vec::new();
        let mut compiler_arguments = Vec::new();
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
                    compiler = Some(PathBuf::from(&argument[3..]));
                }
                _ if argument.starts_with("CFLAGS=") => {
                    compiler_arguments.extend(split_driver_arguments(&argument[7..]));
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
        let mut build =
            MexBuild::new(first, output_directory).api(api.map(|(api, _)| api).unwrap_or_default());
        for source in sources {
            build = build.source(source);
        }
        if let Some(output_name) = output_name {
            build = build.output_name(output_name);
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
        for argument in compiler_arguments {
            build = build.compiler_argument(argument);
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
}
