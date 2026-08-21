use runmat_execution::identity::ArtifactId;

use super::PortFuture;
use crate::RunnerResult;

pub trait ArtifactPort {
    fn ensure_available<'a>(
        &'a mut self,
        artifact_id: ArtifactId,
    ) -> PortFuture<'a, RunnerResult<()>>;
}
