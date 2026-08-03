use super::OverlayRoute;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueRelayRoute(pub OverlayRoute);

impl OpaqueRelayRoute {
    pub fn new(authority: impl Into<String>) -> Option<Self> {
        let authority = authority.into();
        (!authority.trim().is_empty()).then_some(Self(OverlayRoute::OpaqueRelay { authority }))
    }
}
