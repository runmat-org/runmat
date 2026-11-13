//! A native stage lives in a newly created private sibling directory.
//! Neither its name nor its cleanup path is exposed to tabular consumers.

use crate::{FsPublicationFailure, FsStagedOutput};
use async_trait::async_trait;
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub(crate) fn begin(destination: &Path) -> io::Result<Box<dyn FsStagedOutput>> {
    if !destination.is_absolute() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "staged replacement requires an absolute destination",
        ));
    }
    let parent = destination.parent().ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidInput,
            "destination has no parent directory",
        )
    })?;
    for _ in 0..16 {
        let directory = parent.join(format!(".runmat-stage-{}", Uuid::new_v4()));
        match begin_at(destination, directory) {
            Ok(stage) => return Ok(Box::new(stage)),
            Err(source) if source.kind() == ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(source),
        }
    }
    Err(io::Error::new(
        ErrorKind::AlreadyExists,
        "could not acquire a unique staged output",
    ))
}

fn begin_at(destination: &Path, directory: PathBuf) -> io::Result<NativeStagedOutput> {
    create_private_dir(&directory)?;
    let mut stage = NativeStagedOutput {
        destination: destination.to_path_buf(),
        payload: directory.join("payload"),
        directory,
        file: None,
        owned_payload: false,
        published: false,
    };
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stage.payload)?;
    stage.file = Some(file);
    stage.owned_payload = true;
    Ok(stage)
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700).create(path)
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> io::Result<()> {
    fs::create_dir(path)
}

struct NativeStagedOutput {
    destination: PathBuf,
    directory: PathBuf,
    payload: PathBuf,
    file: Option<File>,
    owned_payload: bool,
    published: bool,
}

#[async_trait(?Send)]
impl FsStagedOutput for NativeStagedOutput {
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.file
            .as_mut()
            .ok_or_else(|| io::Error::other("staged output is closed"))?
            .write(bytes)
    }

    async fn flush_and_sync(&mut self) -> io::Result<()> {
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("staged output is closed"))?;
        file.flush()?;
        file.sync_all()
    }

    async fn publish(mut self: Box<Self>) -> Result<(), FsPublicationFailure> {
        // The native implementation performs no await after rename. Dropping
        // a pending future before it reaches that syscall still drops `self`.
        self.flush_and_sync()
            .await
            .map_err(FsPublicationFailure::not_published)?;
        self.file.take();
        fs::rename(&self.payload, &self.destination)
            .map_err(FsPublicationFailure::not_published)?;
        self.published = true;
        Ok(())
    }
}

impl Drop for NativeStagedOutput {
    fn drop(&mut self) {
        self.file.take();
        if self.owned_payload && !self.published {
            let _ = fs::remove_file(&self.payload);
        }
        // Never recurse: unexpected material in the private directory is not
        // ours to delete. An I/O failure may leave an inert staging directory.
        let _ = fs::remove_dir(&self.directory);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn collided_stage_candidate_never_removes_existing_bytes() {
        let root = tempfile::tempdir().expect("fixture root");
        let destination = root.path().join("output.csv");
        fs::write(&destination, b"original").expect("destination");
        let occupied = root.path().join("occupied-stage");
        fs::create_dir(&occupied).expect("occupied directory");
        fs::write(occupied.join("payload"), b"user bytes").expect("sentinel");

        let Err(error) = begin_at(&destination, occupied.clone()) else {
            panic!("occupied directory must reject stage acquisition");
        };
        assert_eq!(error.kind(), ErrorKind::AlreadyExists);
        assert_eq!(fs::read(occupied.join("payload")).unwrap(), b"user bytes");
        assert_eq!(fs::read(&destination).unwrap(), b"original");
    }

    #[test]
    fn dropped_and_failed_stages_preserve_previous_file() {
        let root = tempfile::tempdir().expect("fixture root");
        let destination = root.path().join("output.csv");
        fs::write(&destination, b"original").expect("destination");
        let stage_dir = root.path().join("owned-stage");
        let mut stage = begin_at(&destination, stage_dir.clone()).expect("stage");
        assert_eq!(block_on(stage.write_chunk(b"replacement")).unwrap(), 11);
        drop(stage);
        assert!(!stage_dir.exists());
        assert_eq!(fs::read(&destination).unwrap(), b"original");

        let directory_target = root.path().join("directory-target");
        fs::create_dir(&directory_target).expect("directory target");
        let stage_dir = root.path().join("failed-stage");
        let mut stage = begin_at(&directory_target, stage_dir.clone()).expect("stage");
        block_on(stage.write_chunk(b"payload")).expect("write");
        let failure = block_on(Box::new(stage).publish()).expect_err("directory target");
        assert_eq!(
            failure.disposition(),
            crate::FsPublicationDisposition::NotPublished
        );
        assert!(!stage_dir.exists());
        assert!(directory_target.is_dir());
    }

    #[test]
    fn completed_publish_replaces_only_captured_destination() {
        let root = tempfile::tempdir().expect("fixture root");
        let destination = root.path().join("output.csv");
        let neighbor = root.path().join("neighbor.csv");
        fs::write(&destination, b"original").expect("destination");
        fs::write(&neighbor, b"neighbor").expect("neighbor");
        let mut stage = begin(&destination).expect("stage");
        assert_eq!(block_on(stage.write_chunk(b"replacement")).unwrap(), 11);
        block_on(stage.publish()).expect("publish");
        assert_eq!(fs::read(&destination).unwrap(), b"replacement");
        assert_eq!(fs::read(&neighbor).unwrap(), b"neighbor");
    }
}
