use crate::{SharedClassIdentity, Utf8Text};
use runmat_gc_api::GcHandle;

/// Event listener handle for events
#[derive(Debug, Clone, PartialEq)]
pub struct Listener {
    pub id: u64,
    pub target: GcHandle,
    pub target_class_name: SharedClassIdentity,
    pub event_name: Utf8Text,
    pub callback: GcHandle,
    pub enabled: bool,
    pub valid: bool,
}

impl Listener {
    pub fn class_name(&self) -> &runmat_types::ClassIdentity {
        self.target_class_name.as_identity()
    }
}
