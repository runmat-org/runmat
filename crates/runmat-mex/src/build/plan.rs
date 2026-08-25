use std::path::{Path, PathBuf};

use super::{compiler_family, mex_suffix, sdk, CCompilerFamily, MexBuild, MexBuildError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexBuildPlan {
    pub compiler: PathBuf,
    pub arguments: Vec<String>,
    pub module: PathBuf,
}

impl MexBuildPlan {
    pub(super) fn for_build(build: &MexBuild) -> Result<Self, MexBuildError> {
        let suffix = mex_suffix().ok_or(MexBuildError::UnsupportedTarget)?;
        validate(build)?;
        let module = build
            .output_directory
            .join(format!("{}.{}", build.output_name, suffix));
        let function_name = c_identifier(&build.output_name);
        let sdk = sdk::prepare()?;
        let family = compiler_family(&build.compiler);
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
    if build.interleaved_complex {
        arguments.push(format!("{prefix}RUNMAT_MX_INTERLEAVED_COMPLEX=1"));
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

    #[test]
    fn msvc_plan_uses_native_driver_spelling() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("demo.c");
        std::fs::write(&source, "void mexFunction(void) {}").unwrap();
        let build = MexBuild::new(&source, temporary.path())
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
}
