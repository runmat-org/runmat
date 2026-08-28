use std::path::Path;

use runmat_test::identity::RunId;

use crate::NativeRunnerResult;

use super::paths::run_directory;

pub async fn remove_artifact_run(root: &Path, run_id: &RunId) -> NativeRunnerResult<()> {
    let directory = run_directory(root, run_id);
    match tokio::fs::remove_dir_all(directory).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}
