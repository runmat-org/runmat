use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use runmat_builtins::TEMPNAME_ERROR_UNABLE_TO_GENERATE;

use super::super::error;

const IDENTITY: &str = "tempname";
const MAX_ATTEMPTS: usize = 64;
static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) async fn unused_path(base: &Path) -> crate::BuiltinResult<PathBuf> {
    unused_path_with(base, |path| async move {
        runmat_filesystem::symlink_metadata_async(&path).await
    })
    .await
}

async fn unused_path_with<F, Fut>(base: &Path, mut metadata: F) -> crate::BuiltinResult<PathBuf>
where
    F: FnMut(PathBuf) -> Fut,
    Fut: std::future::Future<Output = std::io::Result<runmat_filesystem::FsMetadata>>,
{
    for _ in 0..MAX_ATTEMPTS {
        let candidate = base.join(token());
        match metadata(candidate.clone()).await {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(candidate),
            Err(source) => {
                return Err(error::message(
                    IDENTITY,
                    &TEMPNAME_ERROR_UNABLE_TO_GENERATE,
                    format!("tempname: unable to check candidate: {source}"),
                ));
            }
            Ok(_) => {}
        }
    }
    Err(error::builtin(IDENTITY, &TEMPNAME_ERROR_UNABLE_TO_GENERATE))
}

fn token() -> String {
    let now = runmat_time::system_time_now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let counter = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "tp{:016x}{:08x}{:08x}{:016x}",
        now.as_secs(),
        now.subsec_nanos(),
        process_id(),
        counter
    )
}

fn process_id() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        u64::from(std::process::id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error, ErrorKind};

    #[test]
    fn only_not_found_establishes_an_unused_candidate() {
        for kind in [
            ErrorKind::PermissionDenied,
            ErrorKind::Unsupported,
            ErrorKind::Interrupted,
        ] {
            let mut calls = 0;
            let error = futures::executor::block_on(unused_path_with(Path::new("base"), |_| {
                calls += 1;
                std::future::ready(Err(Error::new(kind, "injected metadata failure")))
            }))
            .unwrap_err();
            assert_eq!(calls, 1);
            assert_eq!(error.identifier(), Some("RunMat:tempname:UnableToGenerate"));
        }
        let path = futures::executor::block_on(unused_path_with(Path::new("base"), |_| {
            std::future::ready(Err(Error::from(ErrorKind::NotFound)))
        }))
        .unwrap();
        assert_eq!(path.parent(), Some(Path::new("base")));
    }

    #[test]
    fn existing_candidates_retry_with_distinct_names_and_a_bounded_budget() {
        let mut candidates = std::collections::HashSet::new();
        let error = futures::executor::block_on(unused_path_with(Path::new("base"), |path| {
            assert!(candidates.insert(path));
            std::future::ready(Ok(runmat_filesystem::FsMetadata::new(
                runmat_filesystem::FsFileType::File,
                0,
                None,
                false,
            )))
        }))
        .unwrap_err();
        assert_eq!(candidates.len(), MAX_ATTEMPTS);
        assert_eq!(error.identifier(), Some("RunMat:tempname:UnableToGenerate"));
    }
}
