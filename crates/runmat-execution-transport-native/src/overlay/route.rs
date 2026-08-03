#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayRoute {
    DirectQuic { authority: String },
    OpaqueRelay { authority: String },
}

impl OverlayRoute {
    pub fn authority(&self) -> &str {
        match self {
            Self::DirectQuic { authority } | Self::OpaqueRelay { authority } => authority,
        }
    }
}
