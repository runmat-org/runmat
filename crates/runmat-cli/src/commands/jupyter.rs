use anyhow::{Context, Result};
use log::info;
use std::fs;
use std::path::{Path, PathBuf};

pub async fn install_jupyter_kernel() -> Result<()> {
    info!("Installing RunMat as a Jupyter kernel");

    let current_exe = std::env::current_exe().context("Failed to get current executable path")?;
    let kernel_dir =
        find_jupyter_kernel_dir().context("Failed to find Jupyter kernel directory")?;
    let runmat_kernel_dir = kernel_dir.join("runmat");

    fs::create_dir_all(&runmat_kernel_dir).with_context(|| {
        format!(
            "Failed to create kernel directory: {}",
            runmat_kernel_dir.display()
        )
    })?;

    let kernel_json = format!(
        r#"{{
  "argv": [
    "{}",
    "kernel-connection",
    "{{connection_file}}"
  ],
  "display_name": "RunMat",
  "language": "matlab",
  "metadata": {{
    "debugger": false
  }}
}}"#,
        current_exe.display()
    );

    let kernel_json_path = runmat_kernel_dir.join("kernel.json");
    fs::write(&kernel_json_path, kernel_json).with_context(|| {
        format!(
            "Failed to write kernel.json to {}",
            kernel_json_path.display()
        )
    })?;

    create_kernel_logos(&runmat_kernel_dir)?;

    println!("RunMat Jupyter kernel installed successfully!");
    println!("Kernel directory: {}", runmat_kernel_dir.display());
    println!();
    println!("You can now start Jupyter and select 'RunMat' as a kernel:");
    println!("  jupyter notebook");
    println!("  # or");
    println!("  jupyter lab");
    println!();
    println!("To verify the installation:");
    println!("  jupyter kernelspec list");

    Ok(())
}

fn find_jupyter_kernel_dir() -> Result<PathBuf> {
    if let Ok(output) = std::process::Command::new("jupyter")
        .args(["--data-dir"])
        .output()
    {
        if output.status.success() {
            let data_dir_str = String::from_utf8_lossy(&output.stdout);
            let data_dir = data_dir_str.trim();
            let kernels_dir = PathBuf::from(data_dir).join("kernels");
            if kernels_dir.exists() || kernels_dir.parent().is_some_and(|p| p.exists()) {
                return Ok(kernels_dir);
            }
        }
    }

    if let Some(home_dir) = dirs::home_dir() {
        let user_kernels = home_dir.join(".local/share/jupyter/kernels");
        if user_kernels.exists() || user_kernels.parent().is_some_and(|p| p.exists()) {
            return Ok(user_kernels);
        }

        #[cfg(target_os = "macos")]
        {
            let macos_kernels = home_dir.join("Library/Jupyter/kernels");
            if macos_kernels.exists() || macos_kernels.parent().is_some_and(|p| p.exists()) {
                return Ok(macos_kernels);
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(appdata) = std::env::var("APPDATA") {
                let windows_kernels = PathBuf::from(appdata).join("jupyter/kernels");
                if windows_kernels.exists() || windows_kernels.parent().is_some_and(|p| p.exists())
                {
                    return Ok(windows_kernels);
                }
            }
        }

        let default_kernels = home_dir.join(".local/share/jupyter/kernels");
        return Ok(default_kernels);
    }

    Err(anyhow::anyhow!(
        "Could not determine Jupyter kernel directory. Please install Jupyter first."
    ))
}

fn create_kernel_logos(kernel_dir: &Path) -> Result<()> {
    let logo_info = kernel_dir.join("logo-readme.txt");
    fs::write(
        logo_info,
        "RunMat kernel logos can be added here:\n- logo-32x32.png\n- logo-64x64.png",
    )
    .context("Failed to create logo info file")?;

    Ok(())
}
