/// Textual operation dimension accepted by a fully connected layer. Numeric
/// dimensions are admitted separately at the constructor boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeepLearningDenseOperationDimension {
    SpatialChannel,
}

impl DeepLearningDenseOperationDimension {
    pub const ALL: [Self; 1] = [Self::SpatialChannel];
    pub const NAMES: &'static [&'static str] = &[Self::SpatialChannel.canonical_name()];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|dimension| name.eq_ignore_ascii_case(dimension.canonical_name()))
    }

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::SpatialChannel => "spatial-channel",
        }
    }
}
