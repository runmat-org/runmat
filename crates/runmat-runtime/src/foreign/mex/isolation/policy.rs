use std::time::Duration;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnmanifestedMexPolicy {
    #[default]
    Isolate,
    Deny,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MexIsolationPolicy {
    pub unmanifested: UnmanifestedMexPolicy,
    pub invocation_timeout: Option<Duration>,
}
