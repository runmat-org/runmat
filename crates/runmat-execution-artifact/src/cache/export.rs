use crate::{ArtifactResult, LogicalObject};

/// Content-addressed sink for canonical execution objects.
///
/// Implementations must be idempotent for identical objects and reject any attempt to bind the
/// same content identity to different bytes. Publication and visibility remain runner concerns.
pub trait CacheExport {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()>;
}
