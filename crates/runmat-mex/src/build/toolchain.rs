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
