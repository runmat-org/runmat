use runmat_gc_api::GcHandle;
use runmat_types::ClassIdentity;

/// Event listener handle for events
#[derive(Debug, Clone, PartialEq)]
pub struct Listener {
    pub id: u64,
    pub target: GcHandle,
    pub target_class_name: ClassIdentity,
    pub event_name: String,
    pub callback: GcHandle,
    pub enabled: bool,
    pub valid: bool,
}

impl Listener {
    pub fn class_name(&self) -> &ClassIdentity {
        &self.target_class_name
    }
}
