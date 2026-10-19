//! Shared semantic identities for optimization option builders.
//!
//! Public spellings are decoded here. Option structs retain their historical
//! field names for solver interoperability, but neither builder dispatches on
//! those presentation strings after admission.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimOptionField {
    TolX,
    TolFun,
    MaxIter,
    MaxFunEvals,
    Display,
    Algorithm,
    SpecifyObjectiveGradient,
}

impl OptimOptionField {
    pub const fn name(self) -> &'static str {
        match self {
            Self::TolX => "TolX",
            Self::TolFun => "TolFun",
            Self::MaxIter => "MaxIter",
            Self::MaxFunEvals => "MaxFunEvals",
            Self::Display => "Display",
            Self::Algorithm => "Algorithm",
            Self::SpecifyObjectiveGradient => "SpecifyObjectiveGradient",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "tolx" | "steptolerance" => Some(Self::TolX),
            "tolfun" | "functiontolerance" | "optimalitytolerance" => Some(Self::TolFun),
            "maxiter" => Some(Self::MaxIter),
            "maxfunevals" | "maxfunctionevaluations" => Some(Self::MaxFunEvals),
            "display" => Some(Self::Display),
            "algorithm" => Some(Self::Algorithm),
            "specifyobjectivegradient" | "gradobj" => Some(Self::SpecifyObjectiveGradient),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimSolver {
    Coneprog,
    Fminbnd,
    Fminunc,
    Fzero,
    Fsolve,
    Lsqcurvefit,
    Lsqnonlin,
    Generic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimDisplay {
    Off,
    None,
    Iter,
    Notify,
    Final,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimSwitch {
    Off,
    On,
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
        }
    }
}

impl OptimSolver {
    /// Existing positional options-struct recognition uses canonical field
    /// spellings; aliases remain available at the option-name parser boundary.
    pub fn recognizes_options_field(self, name: &str) -> bool {
        name.eq_ignore_ascii_case("Solver")
            || OptimOptionField::parse(name)
                .is_some_and(|field| self.accepts(field) && name.eq_ignore_ascii_case(field.name()))
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Coneprog => "coneprog",
            Self::Fminbnd => "fminbnd",
            Self::Fminunc => "fminunc",
            Self::Fzero => "fzero",
            Self::Fsolve => "fsolve",
            Self::Lsqcurvefit => "lsqcurvefit",
            Self::Lsqnonlin => "lsqnonlin",
            Self::Generic => "",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "coneprog" => Some(Self::Coneprog),
            "fminbnd" => Some(Self::Fminbnd),
            "fminunc" => Some(Self::Fminunc),
            "fzero" => Some(Self::Fzero),
            "fsolve" => Some(Self::Fsolve),
            "lsqcurvefit" => Some(Self::Lsqcurvefit),
            "lsqnonlin" => Some(Self::Lsqnonlin),
            _ => None,
        }
    }

    pub const fn default_display(self) -> &'static str {
        match self {
            Self::Fminbnd => "notify",
            _ => "off",
        }
    }

    pub const fn accepts(self, field: OptimOptionField) -> bool {
        use OptimOptionField as Field;
        match field {
            Field::TolX | Field::MaxIter | Field::MaxFunEvals | Field::Display => true,
            Field::TolFun => matches!(
                self,
                Self::Coneprog
                    | Self::Fminunc
                    | Self::Fsolve
                    | Self::Lsqcurvefit
                    | Self::Lsqnonlin
                    | Self::Generic
            ),
            Field::Algorithm => matches!(
                self,
                Self::Coneprog
                    | Self::Fminunc
                    | Self::Lsqcurvefit
                    | Self::Lsqnonlin
                    | Self::Generic
            ),
            Field::SpecifyObjectiveGradient => matches!(self, Self::Fminunc | Self::Generic),
        }
    }

    pub const fn accepts_display(self, display: OptimDisplay) -> bool {
        match self {
            Self::Fminbnd | Self::Fminunc | Self::Generic => {
                matches!(
                    display,
                    OptimDisplay::Off
                        | OptimDisplay::None
                        | OptimDisplay::Iter
                        | OptimDisplay::Notify
                        | OptimDisplay::Final
                )
            }
            _ => matches!(
                display,
                OptimDisplay::Off | OptimDisplay::None | OptimDisplay::Iter | OptimDisplay::Final
            ),
        }
    }

    pub const fn accepts_algorithm(self, algorithm: OptimAlgorithm) -> bool {
        match self {
            Self::Fminunc => matches!(
                algorithm,
                OptimAlgorithm::QuasiNewton | OptimAlgorithm::Bfgs
            ),
            Self::Coneprog => matches!(
                algorithm,
                OptimAlgorithm::InteriorPoint | OptimAlgorithm::InteriorPointConvex
            ),
            Self::Lsqcurvefit | Self::Lsqnonlin => matches!(
                algorithm,
                OptimAlgorithm::LevenbergMarquardt | OptimAlgorithm::TrustRegionReflective
            ),
            Self::Generic => matches!(
                algorithm,
                OptimAlgorithm::QuasiNewton
                    | OptimAlgorithm::Bfgs
                    | OptimAlgorithm::LevenbergMarquardt
                    | OptimAlgorithm::TrustRegionReflective
            ),
            _ => false,
        }
    }
}
