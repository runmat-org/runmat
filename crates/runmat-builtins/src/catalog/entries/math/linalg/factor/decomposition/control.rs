//! Closed source controls shared by static inference and Runtime admission.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecompositionKind {
    Auto,
    Lu,
    Qr,
    Cod,
    Chol,
    Triangular,
    Diagonal,
}

impl DecompositionKind {
    pub fn parse(text: &str) -> Option<Self> {
        [
            Self::Auto,
            Self::Lu,
            Self::Qr,
            Self::Cod,
            Self::Chol,
            Self::Triangular,
            Self::Diagonal,
        ]
        .into_iter()
        .find(|kind| text.eq_ignore_ascii_case(kind.as_str()))
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Lu => "lu",
            Self::Qr => "qr",
            Self::Cod => "cod",
            Self::Chol => "chol",
            Self::Triangular => "triangular",
            Self::Diagonal => "diagonal",
        }
    }

    pub const fn requires_square(self) -> bool {
        matches!(
            self,
            Self::Lu | Self::Chol | Self::Triangular | Self::Diagonal
        )
    }

    pub const fn accepts_triangle(self) -> bool {
        matches!(self, Self::Chol | Self::Triangular)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecompositionOption {
    CheckCondition,
    RankTolerance,
}

impl DecompositionOption {
    pub fn parse(text: &str) -> Option<Self> {
        [Self::CheckCondition, Self::RankTolerance]
            .into_iter()
            .find(|option| text.eq_ignore_ascii_case(option.as_str()))
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CheckCondition => "checkcondition",
            Self::RankTolerance => "ranktolerance",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecompositionTriangle {
    Upper,
    Lower,
}

impl DecompositionTriangle {
    pub fn parse(text: &str) -> Option<Self> {
        [Self::Upper, Self::Lower]
            .into_iter()
            .find(|triangle| text.eq_ignore_ascii_case(triangle.as_str()))
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Upper => "upper",
            Self::Lower => "lower",
        }
    }
}
