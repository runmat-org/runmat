use std::future::Future;
use std::pin::Pin;

use runmat_test::identity::RunId;

use super::StoredArtifact;
use crate::RunnerResult;

pub type ArtifactFuture<'a, T> = Pin<Box<dyn Future<Output = RunnerResult<T>> + 'a>>;

pub trait ArtifactStore {
    fn put<'a>(
        &'a self,
        run_id: &'a RunId,
        name: &'a str,
        media_type: &'a str,
        bytes: &'a [u8],
    ) -> ArtifactFuture<'a, StoredArtifact>;

    fn remove_run<'a>(&'a self, run_id: &'a RunId) -> ArtifactFuture<'a, ()>;
}
