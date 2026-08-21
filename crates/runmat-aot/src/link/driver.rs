use std::path::{Path, PathBuf};

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
        &["link.exe", "lld-link.exe"]
    } else {
        &["cc", "clang", "gcc"]
    };
    for candidate in candidates {
        if let Some(path) = find_on_path(candidate) {
            return Ok(LinkerDriver { path, family });
        }
    }
    Err(AotError::contract(
        "aot.linker.missing",
        "no supported system linker driver was found; install a C toolchain or set RUNMAT_LINKER",
    ))
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|directory| directory.join(name))
            .find(|path| path.is_file())
    })
}
