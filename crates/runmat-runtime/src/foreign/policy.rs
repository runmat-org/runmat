#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignTrust {
    Trusted,
    Untrusted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignIsolation {
    InProcess,
    IsolatedProcess,
    RemoteHost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignPlatform {
    Native,
    Wasm { host_bridge_available: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForeignExecutionPolicy {
    pub trust: ForeignTrust,
    pub isolation: ForeignIsolation,
}

impl ForeignExecutionPolicy {
    pub const fn trusted_in_process() -> Self {
        Self {
            trust: ForeignTrust::Trusted,
            isolation: ForeignIsolation::InProcess,
        }
    }

    pub const fn requires_process_boundary(self) -> bool {
        matches!(self.trust, ForeignTrust::Untrusted)
            || !matches!(self.isolation, ForeignIsolation::InProcess)
    }
}
