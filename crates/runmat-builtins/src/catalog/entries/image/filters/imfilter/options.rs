#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImfilterOption {
    Replicate,
    Symmetric,
    Circular,
    Fill,
    Same,
    Full,
    Valid,
    Convolution,
    Correlation,
}

impl ImfilterOption {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "replicate" => Some(Self::Replicate),
            "symmetric" => Some(Self::Symmetric),
            "circular" => Some(Self::Circular),
            "fill" => Some(Self::Fill),
            "same" => Some(Self::Same),
            "full" => Some(Self::Full),
            "valid" => Some(Self::Valid),
            "conv" => Some(Self::Convolution),
            "corr" => Some(Self::Correlation),
            _ => None,
        }
    }
}
