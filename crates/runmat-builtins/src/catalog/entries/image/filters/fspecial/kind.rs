#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FspecialFilterKind {
    Average,
    Disk,
    Gaussian,
    Laplacian,
    Log,
    Motion,
    Prewitt,
    Sobel,
    Unsharp,
}

impl FspecialFilterKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "average" => Some(Self::Average),
            "disk" => Some(Self::Disk),
            "gaussian" => Some(Self::Gaussian),
            "laplacian" => Some(Self::Laplacian),
            "log" => Some(Self::Log),
            "motion" => Some(Self::Motion),
            "prewitt" => Some(Self::Prewitt),
            "sobel" => Some(Self::Sobel),
            "unsharp" => Some(Self::Unsharp),
            _ => None,
        }
    }

    pub fn maximum_arguments(self) -> usize {
        match self {
            Self::Prewitt | Self::Sobel => 1,
            Self::Gaussian | Self::Log | Self::Motion => 3,
            _ => 2,
        }
    }
}
