use runmat_gc_api::GcHandle;

#[derive(Debug, Clone)]
pub struct HandleRef {
    pub class_name: String,
    pub target: GcHandle,
    pub valid: bool,
}

impl PartialEq for HandleRef {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target
    }
}
