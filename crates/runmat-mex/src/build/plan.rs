use std::path::{Path, PathBuf};

use super::{compiler_family, sdk, CCompilerFamily, MexApi, MexBuild, MexBuildError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexBuildPlan {
    pub compiler: PathBuf,
    pub arguments: Vec<String>,
    pub module: PathBuf,
    pub target: super::MexTarget,
}

impl MexBuildPlan {
    pub(super) fn for_build(build: &MexBuild) -> Result<Self, MexBuildError> {
        validate(build)?;
        build.target.validate()?;
        let family = compiler_family(&build.compiler);
        build.target.validate_compiler(family)?;
        let module = build
            .output_directory
            .join(format!("{}.{}", build.output_name, build.target.suffix));
        let function_name = c_identifier(&build.output_name);
        let sdk = sdk::prepare()?;
        let arguments = match family {
            CCompilerFamily::GnuLike => gnu_arguments(
                build,
                &module,
                &function_name,
                &sdk.include_directory,
                &sdk.shim,
            ),
            CCompilerFamily::Msvc => msvc_arguments(
                build,
                &module,
                &function_name,
                &sdk.include_directory,
                &sdk.shim,
            ),
        };
        Ok(Self {
            compiler: build.compiler.clone(),
            arguments,
            module,
            target: build.target.clone(),
        })
    }

    pub fn command(&self) -> Vec<String> {
        std::iter::once(self.compiler.display().to_string())
            .chain(self.arguments.iter().cloned())
            .collect()
    }
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
    Ok(())
}

fn gnu_arguments(
    build: &MexBuild,
    module: &Path,
    function_name: &str,
    sdk_include: &Path,
    shim: &Path,
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
    arguments.extend(["-std=c11".into(), "-O2".into()]);
    arguments.push(format!("-I{}", sdk_include.display()));
    for include in &build.include_directories {
        arguments.push(format!("-I{}", include.display()));
    }
    push_definitions(build, function_name, "-D", &mut arguments);
    arguments.extend(build.compiler_arguments.iter().cloned());
    arguments.extend(build.sources.iter().map(|path| path.display().to_string()));
    arguments.push(shim.display().to_string());
    arguments.extend(build.linker_arguments.iter().cloned());
    arguments.extend(["-o".into(), module.display().to_string()]);
    arguments
}

fn msvc_arguments(
    build: &MexBuild,
    module: &Path,
    function_name: &str,
    sdk_include: &Path,
    shim: &Path,
) -> Vec<String> {
    let mut arguments = vec![
        "/nologo".into(),
        "/LD".into(),
        "/O2".into(),
        "/std:c11".into(),
        format!("/I{}", sdk_include.display()),
    ];
    for include in &build.include_directories {
        arguments.push(format!("/I{}", include.display()));
    }
    push_definitions(build, function_name, "/D", &mut arguments);
    arguments.extend(build.compiler_arguments.iter().cloned());
    arguments.extend(build.sources.iter().map(|path| path.display().to_string()));
    arguments.push(shim.display().to_string());
    arguments.push("/link".into());
    arguments.extend(build.linker_arguments.iter().cloned());
    arguments.push(format!("/OUT:{}", module.display()));
    arguments
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
    arguments.extend(
        build
            .definitions
            .iter()
            .map(|definition| format!("{prefix}{definition}")),
    );
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
