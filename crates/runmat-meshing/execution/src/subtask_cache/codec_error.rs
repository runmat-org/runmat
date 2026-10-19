//! Cache-specific error mapping stays at the cache owner.
use super::MeshingSubtaskCacheError;
use crate::codec_control::CodecControl;
impl CodecControl<'_> {
    pub(crate) fn error(&self, error: impl std::fmt::Display) -> MeshingSubtaskCacheError {
        match self.take_failure() {
            Some(error) => MeshingSubtaskCacheError::Control(error),
            None => MeshingSubtaskCacheError::Storage(error.to_string()),
        }
    }
}
