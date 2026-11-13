use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::io::AsyncWriteExt;

static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

struct TemporaryFile(Option<PathBuf>);

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Publish a complete file in its destination directory, replacing only that entry.
pub(super) async fn write_atomic(target: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = target.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "artifact target has no parent directory",
        )
    })?;
    tokio::fs::create_dir_all(parent).await?;

    let (mut temporary, mut file) = loop {
        let nonce = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".runmat-artifact-{}-{nonce}.tmp",
            std::process::id()
        ));
        // Create synchronously so cancellation cannot occur after the file is
        // created but before its RAII cleanup guard takes ownership.
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => break (TemporaryFile(Some(path)), tokio::fs::File::from_std(file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    file.write_all(bytes).await?;
    file.sync_all().await?;
    drop(file);
    std::fs::rename(temporary.0.as_ref().expect("live temporary path"), target)?;
    temporary.0.take();
    Ok(())
}
