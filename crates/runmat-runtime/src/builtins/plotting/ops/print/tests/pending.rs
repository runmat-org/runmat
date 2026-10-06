//! Public print-future source/scene ownership through a suspended stage.

use super::*;
use async_trait::async_trait;
use futures::channel::oneshot;
use futures::future::{select, Either};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelIntegerDownloadFuture, AccelNumericDownloadFuture, AccelProvider,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataView, HostNumericTensorView, HostTensorView,
    ThreadProviderGuard,
};
use runmat_filesystem::{
    self as fs, DirEntry, FileHandle, FsMetadata, FsProvider, FsPublicationFailure,
    FsResourceContract, FsResourceLease, FsResourceRevision, FsStagedOutput, OpenFlags,
};
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[path = "pending/fixtures.rs"]
mod fixtures;
#[path = "pending/publication.rs"]
mod publication;
#[path = "pending/resident.rs"]
mod resident;

use fixtures::{await_operation_start, PausedDownloadProvider, PausedProvider, RevocableLease};
