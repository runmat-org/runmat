//! Existing smoke's opt-in product mode; no second window loop or renderer.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Child(std::process::Child);
impl Drop for Child {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) fn run(executable: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !executable.is_absolute() || !executable.is_file() {
        return Err(std::io::Error::other(
            "CLI smoke requires an existing absolute executable path",
        )
        .into());
    }
    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory =
        std::env::temp_dir().join(format!("runmat-native-gui-{}-{unique}", std::process::id()));
    std::fs::create_dir(&directory)?;
    let fixture = Fixture(directory);
    let script = fixture.0.join("native_graphics.m");
    std::fs::write(&script, "f = figure();\nplot([0 1 2], [0 1 4]);\ndrawnow;\ntitle('updated native scene');\ndrawnow('nocallbacks');\nclose(f);\ndrawnow;\ndisp('RUNMAT_NATIVE_GUI_PRODUCT_OK');\n")?;
    let config = fixture.0.join("runtime.toml");
    std::fs::write(
        &config,
        "[runtime.plotting]\nmode = 'gui'\nforce_headless = false\n[runtime.telemetry]\nenabled = false\n",
    )?;
    let stdout = std::fs::File::create(fixture.0.join("stdout"))?;
    let stderr = std::fs::File::create(fixture.0.join("stderr"))?;
    let mut child = Child(
        Command::new(executable)
            .arg("--config")
            .arg(&config)
            .arg("--plot-mode")
            .arg("gui")
            .arg("run")
            .arg(&script)
            .env_remove("RUNMAT_DISABLE_INTERACTIVE_PLOTS")
            .env_remove("RUNMAT_HOST_MANAGED_PLOTS")
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(stderr)
            .spawn()?,
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.0.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.0.kill();
            let _ = child.0.wait();
            return Err(std::io::Error::other(
                "native CLI graphics did not finish within 30 seconds",
            )
            .into());
        }
        std::thread::sleep(Duration::from_millis(16));
    };
    let stdout = std::fs::read_to_string(fixture.0.join("stdout"))?;
    let stderr = std::fs::read_to_string(fixture.0.join("stderr"))?;
    if !status.success() || !stdout.contains("RUNMAT_NATIVE_GUI_PRODUCT_OK") {
        return Err(std::io::Error::other(format!(
            "native CLI graphics failed ({status}):\n{stdout}\n{stderr}"
        ))
        .into());
    }
    println!("native CLI/Core/host-loop drawnow, scene update and close completed with validated receipts");
    Ok(())
}
