use serde::{Deserialize, Serialize};

use super::{CCompilerFamily, MexBuildError};

/// Native target identity for a C MEX module.
///
/// MEX modules are process-native dynamic libraries. A target identity is
/// therefore part of both build admission and durable artifact identity; a
/// matching filename suffix alone is not sufficient evidence of compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexTarget {
    pub triple: String,
    pub architecture: String,
    pub operating_system: String,
    pub pointer_width: u16,
    pub suffix: String,
}

impl MexTarget {
    pub fn current() -> Result<Self, MexBuildError> {
        let suffix = super::mex_suffix().ok_or(MexBuildError::UnsupportedTarget)?;
        let target = Self {
            triple: target_lexicon::HOST.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            operating_system: std::env::consts::OS.to_string(),
            pointer_width: usize::BITS as u16,
            suffix: suffix.to_string(),
        };
        target.validate()?;
        Ok(target)
    }

    pub fn validate(&self) -> Result<(), MexBuildError> {
        let supported = matches!(
            (
                self.architecture.as_str(),
                self.operating_system.as_str(),
                self.pointer_width,
                self.suffix.as_str(),
            ),
            ("aarch64", "macos", 64, "mexmaca64")
                | ("x86_64", "macos", 64, "mexmaci64")
                | ("x86_64", "linux", 64, "mexa64")
                | ("x86_64", "windows", 64, "mexw64")
        );
        if !supported
            || self.triple.is_empty()
            || self.triple.len() > 128
            || !self.triple.is_ascii()
            || self.triple.chars().any(char::is_control)
        {
            return Err(MexBuildError::UnsupportedTargetIdentity {
                triple: self.triple.clone(),
            });
        }
        Ok(())
    }

    /// Admit the operating-system ABI used by the CUDA MEX toolchain.
    /// Callers may check this before source or SDK access; the build plan
    /// repeats it against the artifact target rather than trusting the caller.
    pub fn validate_cuda_build(&self) -> Result<(), MexBuildError> {
        self.validate()?;
        if matches!(self.operating_system.as_str(), "linux" | "windows") {
            Ok(())
        } else {
            Err(MexBuildError::UnsupportedCudaTarget)
        }
    }

    pub(super) fn validate_compiler(&self, family: CCompilerFamily) -> Result<(), MexBuildError> {
        let compatible = match self.operating_system.as_str() {
            "macos" | "linux" => family == CCompilerFamily::GnuLike,
            "windows" => family == CCompilerFamily::Msvc,
            _ => false,
        };
        if compatible {
            Ok(())
        } else {
            Err(MexBuildError::UnsupportedCompilerForTarget {
                compiler_family: family.as_str(),
                triple: self.triple.clone(),
            })
        }
    }

    pub(super) fn is_current(&self) -> bool {
        Self::current().is_ok_and(|current| current == *self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(operating_system: &str, suffix: &str) -> MexTarget {
        MexTarget {
            triple: format!("x86_64-unknown-{operating_system}"),
            architecture: "x86_64".into(),
            operating_system: operating_system.into(),
            pointer_width: 64,
            suffix: suffix.into(),
        }
    }

    #[test]
    fn cuda_target_admission_matches_build_plan_supported_hosts() {
        assert!(target("linux", "mexa64").validate_cuda_build().is_ok());
        assert!(target("windows", "mexw64").validate_cuda_build().is_ok());
        assert!(matches!(
            target("macos", "mexmaci64").validate_cuda_build(),
            Err(MexBuildError::UnsupportedCudaTarget)
        ));
        assert!(matches!(
            target("wasm", "").validate_cuda_build(),
            Err(MexBuildError::UnsupportedTargetIdentity { .. })
        ));
    }
}
