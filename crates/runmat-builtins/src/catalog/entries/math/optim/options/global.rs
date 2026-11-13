//! Closed predefined global-optimizer roles; runtime keeps the selected source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimSwarmCreation {
    Uniform,
}
impl OptimSwarmCreation {
    pub fn parse(name: &str) -> Option<Self> {
        name.trim()
            .eq_ignore_ascii_case("pswcreationuniform")
            .then_some(Self::Uniform)
    }
    pub const fn name(self) -> &'static str {
        "pswcreationuniform"
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimHybridSolver {
    Fminsearch,
    Fminunc,
    Fmincon,
    Patternsearch,
}
impl OptimHybridSolver {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "fminsearch" => Some(Self::Fminsearch),
            "fminunc" => Some(Self::Fminunc),
            "fmincon" => Some(Self::Fmincon),
            "patternsearch" => Some(Self::Patternsearch),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fminsearch => "fminsearch",
            Self::Fminunc => "fminunc",
            Self::Fmincon => "fmincon",
            Self::Patternsearch => "patternsearch",
        }
    }
    pub const fn accepts_bounds(self) -> bool {
        matches!(self, Self::Fmincon | Self::Patternsearch)
    }
}
