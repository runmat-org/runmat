/// Name-value identities are decoded once, before any option value is interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizableVariableOption {
    Optimize,
    Transform,
    Type,
}

impl OptimizableVariableOption {
    pub const fn public_name(self) -> &'static str {
        match self {
            Self::Optimize => "Optimize",
            Self::Transform => "Transform",
            Self::Type => "Type",
        }
    }

    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "type" => Self::Type,
            "transform" => Self::Transform,
            "optimize" => Self::Optimize,
            _ => return None,
        })
    }
}

/// Semantic variable kind shared by construction, object decoding and analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationVariableType {
    Real,
    Integer,
    Categorical,
}

impl OptimizationVariableType {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "real" => Self::Real,
            "integer" => Self::Integer,
            "categorical" => Self::Categorical,
            _ => return None,
        })
    }

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Integer => "integer",
            Self::Categorical => "categorical",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationVariableTransform {
    None,
    Log,
}

impl OptimizationVariableTransform {
    pub fn from_public_name(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "none" => Self::None,
            "log" => Self::Log,
            _ => return None,
        })
    }

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Log => "log",
        }
    }
}
