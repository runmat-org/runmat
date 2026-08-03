use super::OverlayRoute;

/// Validated direct-route intent. Socket/TLS composition lands with secure
/// delivery in E12; callers cannot silently downgrade to relay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectQuicRoute(pub OverlayRoute);

impl DirectQuicRoute {
    pub fn new(authority: impl Into<String>) -> Option<Self> {
        let authority = authority.into();
        (!authority.trim().is_empty()).then_some(Self(OverlayRoute::DirectQuic { authority }))
    }
}
