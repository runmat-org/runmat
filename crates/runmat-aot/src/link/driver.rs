use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{AotError, AotResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LinkerFamily {
    UnixCc,
    Msvc,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct LinkerDriver {
    pub path: PathBuf,
    pub family: LinkerFamily,
}

pub fn discover_linker(explicit: Option<&Path>) -> AotResult<LinkerDriver> {
    let family = if cfg!(target_os = "windows") {
        LinkerFamily::Msvc
    } else {
        LinkerFamily::UnixCc
    };
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(LinkerDriver {
                path: path.to_path_buf(),
                family,
            });
        }
        return Err(AotError::contract(
            "aot.linker.explicit",
            format!("configured linker `{}` does not exist", path.display()),
        ));
    }
    if let Some(path) = std::env::var_os("RUNMAT_LINKER").map(PathBuf::from) {
        return discover_linker(Some(&path));
    }
    let candidates: &[&str] = if cfg!(target_os = "windows") {
        &["lld-link.exe", "link.exe"]
    } else {
        &["cc", "clang", "gcc"]
    };
    for candidate in candidates {
        for path in find_all_on_path(candidate) {
            if family != LinkerFamily::Msvc || is_msvc_linker(&path) {
                return Ok(LinkerDriver { path, family });
            }
        }
    }
    Err(AotError::contract(
        "aot.linker.missing",
        "no supported system linker driver was found; install a C toolchain or set RUNMAT_LINKER",
    ))
}

fn find_all_on_path(name: &str) -> Vec<PathBuf> {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths)
                .map(|directory| directory.join(name))
                .filter(|path| path.is_file())
                .collect()
        })
        .unwrap_or_default()
}

fn is_msvc_linker(path: &Path) -> bool {
    let Ok(output) = Command::new(path).arg("/?").output() else {
        return false;
    };
    let mut diagnostic = String::from_utf8_lossy(&output.stdout).into_owned();
    diagnostic.push_str(&String::from_utf8_lossy(&output.stderr));
    is_msvc_linker_diagnostic(&diagnostic)
}

fn is_msvc_linker_diagnostic(diagnostic: &str) -> bool {
    let diagnostic = diagnostic.to_ascii_lowercase();
    diagnostic.contains("microsoft (r) incremental linker")
        || diagnostic.contains("overview: llvm linker")
}

#[cfg(test)]
mod tests {
    use super::is_msvc_linker_diagnostic;

    #[test]
    fn recognizes_supported_windows_linker_banners() {
        assert!(is_msvc_linker_diagnostic(
            "Microsoft (R) Incremental Linker Version 14.44"
        ));
        assert!(is_msvc_linker_diagnostic(
            "OVERVIEW: LLVM Linker\nUSAGE: lld-link [options] file..."
        ));
    }

    #[test]
    fn rejects_unrelated_path_commands_named_link() {
        assert!(!is_msvc_linker_diagnostic(
            "/usr/bin/link: missing operand after '/?'"
        ));
    }
}
