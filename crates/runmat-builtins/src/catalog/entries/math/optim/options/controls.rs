//! Canonical option control values, independent of storage and dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimDisplay {
    Off,
    None,
    Iter,
    Notify,
    Final,
    Diagnose,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimSwitch {
    Off,
    On,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimDifferenceType {
    Forward,
    Central,
}
impl OptimDifferenceType {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "forward" => Some(Self::Forward),
            "central" => Some(Self::Central),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Central => "central",
        }
    }
}

impl OptimSwitch {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" => Some(Self::On),
            "off" | "false" | "no" => Some(Self::Off),
            _ => None,
        }
    }

    pub const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

impl OptimDisplay {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "none" => Some(Self::None),
            "iter" => Some(Self::Iter),
            "notify" => Some(Self::Notify),
            "final" => Some(Self::Final),
            "diagnose" => Some(Self::Diagnose),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::None => "none",
            Self::Iter => "iter",
            Self::Notify => "notify",
            Self::Final => "final",
            Self::Diagnose => "diagnose",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimHessianSource {
    FiniteDifference,
    Objective,
}

impl OptimHessianSource {
    pub fn parse(text: &str) -> Option<Self> {
        text.trim()
            .eq_ignore_ascii_case("objective")
            .then_some(Self::Objective)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimSubproblemAlgorithm {
    Cg,
    Factorization,
}

impl OptimSubproblemAlgorithm {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "cg" => Some(Self::Cg),
            "factorization" => Some(Self::Factorization),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cg => "cg",
            Self::Factorization => "factorization",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimAlgorithm {
    QuasiNewton,
    Bfgs,
    InteriorPoint,
    InteriorPointConvex,
    LevenbergMarquardt,
    TrustRegionReflective,
    TrustRegion,
    Sqp,
    ActiveSet,
}

impl OptimAlgorithm {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "quasi-newton" => Some(Self::QuasiNewton),
            "bfgs" => Some(Self::Bfgs),
            "interior-point" => Some(Self::InteriorPoint),
            "interior-point-convex" => Some(Self::InteriorPointConvex),
            "levenberg-marquardt" => Some(Self::LevenbergMarquardt),
            "trust-region-reflective" => Some(Self::TrustRegionReflective),
            "trust-region" => Some(Self::TrustRegion),
            "sqp" => Some(Self::Sqp),
            "active-set" => Some(Self::ActiveSet),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::QuasiNewton => "quasi-newton",
            Self::Bfgs => "bfgs",
            Self::InteriorPoint => "interior-point",
            Self::InteriorPointConvex => "interior-point-convex",
            Self::LevenbergMarquardt => "levenberg-marquardt",
            Self::TrustRegionReflective => "trust-region-reflective",
            Self::TrustRegion => "trust-region",
            Self::Sqp => "sqp",
            Self::ActiveSet => "active-set",
        }
    }
}
