//! Installed Windows shader compiler paths; no ambient SDK/PATH dependency.
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Component, Path, PathBuf};

pub(super) fn resolve() -> Result<wgpu::Dx12Compiler> {
    let selected = wgpu::Dx12Compiler::from_env().unwrap_or(wgpu::Dx12Compiler::DynamicDxc {
        dxil_path: "dxil.dll".into(),
        dxc_path: "dxcompiler.dll".into(),
    });
    match selected {
        wgpu::Dx12Compiler::DynamicDxc {
            dxil_path,
            dxc_path,
        } => {
            let executable = std::env::current_exe()
                .context("cannot locate installed Windows shader compiler")?;
            let installation = executable
                .parent()
                .context("Windows executable has no installation directory")?;
            Ok(wgpu::Dx12Compiler::DynamicDxc {
                dxil_path: library_path(installation, dxil_path)?,
                dxc_path: library_path(installation, dxc_path)?,
            })
        }
        // Explicit upstream compiler selections remain explicit; missing DXC
        // never causes us to select a different compiler or compute backend.
        explicit @ (wgpu::Dx12Compiler::Fxc | wgpu::Dx12Compiler::StaticDxc) => Ok(explicit),
    }
}

fn library_path(installation: &Path, requested: String) -> Result<String> {
    installed_path(installation, requested.into())?
        .into_os_string()
        .into_string()
        .map_err(|_| anyhow!("installed Windows shader compiler path is not Unicode"))
}

fn installed_path(installation: &Path, requested: PathBuf) -> Result<PathBuf> {
    if requested.is_absolute() {
        return Ok(requested);
    }
    if requested.as_os_str().is_empty()
        || requested
            .components()
            .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
    {
        bail!("relative Windows shader compiler paths must remain inside the installation");
    }
    Ok(installation.join(requested))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installation_paths_are_lossless_and_reject_ambiguous_relative_paths() {
        let root = Path::new(r"C:\Program Files\RunMat");
        assert_eq!(
            library_path(root, "dxcompiler.dll".into()).unwrap(),
            r"C:\Program Files\RunMat\dxcompiler.dll"
        );
        assert_eq!(
            installed_path(root, "dxcompiler.dll".into()).unwrap(),
            root.join("dxcompiler.dll")
        );
        let explicit = PathBuf::from(r"D:\approved\dxil.dll");
        assert_eq!(installed_path(root, explicit.clone()).unwrap(), explicit);
        for invalid in [r"C:dxcompiler.dll", r"\dxil.dll", r"..\dxcompiler.dll", ""] {
            assert!(installed_path(root, invalid.into()).is_err());
        }
    }
}
