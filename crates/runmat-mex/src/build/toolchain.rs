use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CCompilerFamily {
    GnuLike,
    Msvc,
}

impl CCompilerFamily {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::GnuLike => "gnu-like",
            Self::Msvc => "msvc",
        }
    }
}

pub(super) fn default_c_compiler() -> PathBuf {
    std::env::var_os("CC")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if cfg!(target_os = "windows") {
                PathBuf::from("cl.exe")
            } else {
                PathBuf::from("cc")
            }
        })
}

pub(super) fn default_cxx_compiler() -> PathBuf {
    std::env::var_os("CXX")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if cfg!(target_os = "windows") {
                PathBuf::from("cl.exe")
            } else {
                PathBuf::from("c++")
            }
        })
}

pub(super) fn default_fortran_compiler() -> PathBuf {
    std::env::var_os("FC")
        .or_else(|| std::env::var_os("F77"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("gfortran"))
}

pub(super) fn default_cuda_compiler() -> PathBuf {
    std::env::var_os("NVCC")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("MW_NVCC_PATH").map(|directory| {
                PathBuf::from(directory).join(if cfg!(target_os = "windows") {
                    "nvcc.exe"
                } else {
                    "nvcc"
                })
            })
        })
        .unwrap_or_else(|| {
            PathBuf::from(if cfg!(target_os = "windows") {
                "nvcc.exe"
            } else {
                "nvcc"
            })
        })
}

pub(super) fn is_supported_cuda_compiler(compiler: &Path) -> bool {
    compiler
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name.to_ascii_lowercase().as_str(), "nvcc" | "nvcc.exe"))
}

pub(super) fn is_supported_fortran_compiler(compiler: &Path) -> bool {
    compiler
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().starts_with("gfortran"))
}

pub(super) fn compiler_family(compiler: &Path) -> CCompilerFamily {
    let executable = compiler
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if matches!(
        executable.as_str(),
        "cl" | "cl.exe" | "clang-cl" | "clang-cl.exe"
    ) {
        CCompilerFamily::Msvc
    } else {
        CCompilerFamily::GnuLike
    }
}
