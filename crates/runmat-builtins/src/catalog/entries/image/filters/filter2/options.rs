#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter2Option {
    Same,
    Full,
    Valid,
    Convolution,
    Correlation,
}

impl Filter2Option {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "same" => Some(Self::Same),
            "full" => Some(Self::Full),
            "valid" => Some(Self::Valid),
            "conv" => Some(Self::Convolution),
            "corr" => Some(Self::Correlation),
            _ => None,
        }
    }
}
