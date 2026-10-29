use crate::{ArtifactResult, LogicalObject};

/// Content-addressed sink for canonical execution objects.
///
/// Implementations must be idempotent for identical objects and reject any attempt to bind the
/// same content identity to different bytes. Publication and visibility remain runner concerns.
pub trait CacheExport {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()>;

    /// Writes several objects with the same per-object guarantees as `write_verified`.
    /// Durable stores override this to share one device flush across the whole batch.
    fn write_verified_batch<'a>(
        &mut self,
        objects: impl IntoIterator<Item = &'a LogicalObject>,
    ) -> ArtifactResult<()>
    where
        Self: Sized,
    {
        for object in objects {
            self.write_verified(object)?;
        }
        Ok(())
    }
}
