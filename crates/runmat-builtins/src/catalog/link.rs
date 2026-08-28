use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinReachability {
    Always,
    Dynamic,
    Feature(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinLinkPolicy {
    PortableRuntime,
    HostRuntime,
    NativeSymbol,
    ForeignRuntime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinLinkContract {
    pub reachability: BuiltinReachability,
    pub policy: BuiltinLinkPolicy,
    pub artifact_dependencies: &'static [&'static str],
}
