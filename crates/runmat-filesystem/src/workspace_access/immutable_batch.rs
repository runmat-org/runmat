//! Group commit of immutable files: one durability barrier and one commit record per batch.
//!
//! Publishing an immutable file one at a time costs a full device flush for the staged file
//! and another for each directory that exposes it. A batch stages every file first, makes all
//! of them durable with one barrier, records the batch, and only then links the files into
//! place. The record is the commit point:
//! - a crash before the record is durable leaves no batch file visible (only hidden staging
//!   files, which never carry a published name);
//! - a crash after it is rolled forward by the next scope operation, so every batch file
//!   becomes visible, each with its complete durable content.
//!
//! A published name therefore always refers to durable, complete content, as for a single
//! `atomic_create_new` followed by `synchronize_immutable`.
use super::WorkspaceAccess;
use std::{
    io,
    path::{Path, PathBuf},
};

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub(super) use native::{recover, NativeImmutableBatch};

/// Where a staged immutable file stands before its batch commits.
#[derive(Debug, PartialEq, Eq)]
pub enum ImmutableStage {
    /// New content; visible once the batch commits.
    Staged,
    /// The name already exists with these bytes; the caller decides whether they match.
    /// A committed batch also confirms its durability.
    Existing(Vec<u8>),
}

/// A group of immutable files committed together. Dropping it uncommitted publishes nothing.
pub trait ImmutableBatch {
    fn stage(
        &mut self,
        path: &Path,
        bytes: &[u8],
        maximum_existing_bytes: u64,
    ) -> io::Result<ImmutableStage>;
    /// Makes every staged file durable and visible, and confirms existing ones.
    fn commit(self: Box<Self>) -> io::Result<()>;
}

/// Providers without a native group commit publish and confirm each file on its own.
pub(super) struct SequentialImmutableBatch<'a, A: WorkspaceAccess + ?Sized> {
    pub(super) access: &'a A,
    pub(super) staged: Vec<PathBuf>,
}

impl<A: WorkspaceAccess + ?Sized> ImmutableBatch for SequentialImmutableBatch<'_, A> {
    fn stage(
        &mut self,
        path: &Path,
        bytes: &[u8],
        maximum_existing_bytes: u64,
    ) -> io::Result<ImmutableStage> {
        let stage = match self
            .access
            .atomic_create_new(path, bytes, maximum_existing_bytes)?
        {
            crate::AtomicCreateResult::Published { .. } => ImmutableStage::Staged,
            crate::AtomicCreateResult::Existing(existing) => ImmutableStage::Existing(existing),
        };
        self.staged.push(path.to_path_buf());
        Ok(stage)
    }

    fn commit(self: Box<Self>) -> io::Result<()> {
        for path in &self.staged {
            self.access.synchronize_immutable(path)?;
        }
        Ok(())
    }
}
