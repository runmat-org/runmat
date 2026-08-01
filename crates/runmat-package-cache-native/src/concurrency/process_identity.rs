#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessIdentity {
    pub pid: u32,
}

impl ProcessIdentity {
    pub fn current() -> Self {
        Self {
            pid: std::process::id(),
        }
    }
}
