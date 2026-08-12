use runmat_gc_api::GcHandle;

/// Event listener handle for events
#[derive(Debug, Clone, PartialEq)]
pub struct Listener {
    pub id: u64,
    pub target: GcHandle,
    pub target_class_name: String,
    pub event_name: String,
    pub callback: GcHandle,
    pub enabled: bool,
    pub valid: bool,
}

impl Listener {
    pub fn class_name(&self) -> String {
        self.target_class_name.clone()
    }
}
