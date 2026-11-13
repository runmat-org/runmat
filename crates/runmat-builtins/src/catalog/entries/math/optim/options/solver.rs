//! Solver applicability profiles shared by runtime and inference.
use super::{OptimAlgorithm, OptimDisplay, OptimOptionField};
mod applicability;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimSolver {
    Coneprog,
    Quadprog,
    Lsqlin,
    Linprog,
    Intlinprog,
    Fminbnd,
    Fminunc,
    Fmincon,
    Fseminf,
    Particleswarm,
    Ga,
    Simulannealbnd,
    Surrogateopt,
    Fminimax,
    Fgoalattain,
    Patternsearch,
    Fzero,
    Fsolve,
    Lsqcurvefit,
    Lsqnonlin,
    Generic,
}

impl OptimSolver {
    pub const fn accepts_zero_iterations(self) -> bool {
        matches!(
            self,
            Self::Fminunc
                | Self::Fmincon
                | Self::Fseminf
                | Self::Particleswarm
                | Self::Ga
                | Self::Simulannealbnd
                | Self::Fminimax
                | Self::Fgoalattain
                | Self::Fsolve
                | Self::Linprog
                | Self::Coneprog
                | Self::Quadprog
                | Self::Lsqlin
                | Self::Patternsearch
        )
    }
    /// Existing positional options-struct recognition uses canonical field
    /// spellings; aliases remain available at the option-name parser boundary.
    pub fn recognizes_options_field(self, name: &str) -> bool {
        name.eq_ignore_ascii_case("Solver")
            || OptimOptionField::parse(name).is_some_and(|field| {
                self.accepts(field) && name.eq_ignore_ascii_case(field.name_for_solver(self))
            })
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Coneprog => "coneprog",
            Self::Quadprog => "quadprog",
            Self::Lsqlin => "lsqlin",
            Self::Linprog => "linprog",
            Self::Intlinprog => "intlinprog",
            Self::Fminbnd => "fminbnd",
            Self::Fminunc => "fminunc",
            Self::Fmincon => "fmincon",
            Self::Fseminf => "fseminf",
            Self::Particleswarm => "particleswarm",
            Self::Ga => "ga",
            Self::Simulannealbnd => "simulannealbnd",
            Self::Surrogateopt => "surrogateopt",
            Self::Fminimax => "fminimax",
            Self::Fgoalattain => "fgoalattain",
            Self::Patternsearch => "patternsearch",
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
            "quadprog" => Some(Self::Quadprog),
            "lsqlin" => Some(Self::Lsqlin),
            "linprog" => Some(Self::Linprog),
            "intlinprog" => Some(Self::Intlinprog),
            "fminbnd" => Some(Self::Fminbnd),
            "fminunc" => Some(Self::Fminunc),
            "fmincon" => Some(Self::Fmincon),
            "particleswarm" => Some(Self::Particleswarm),
            "ga" => Some(Self::Ga),
            "simulannealbnd" => Some(Self::Simulannealbnd),
            "surrogateopt" => Some(Self::Surrogateopt),
            "fseminf" => Some(Self::Fseminf),
            "fminimax" => Some(Self::Fminimax),
            "fgoalattain" => Some(Self::Fgoalattain),
            "patternsearch" => Some(Self::Patternsearch),
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
            Self::Quadprog | Self::Lsqlin | Self::Patternsearch => "final",
            Self::Ga
            | Self::Particleswarm
            | Self::Simulannealbnd
            | Self::Surrogateopt
            | Self::Fmincon
            | Self::Fseminf
            | Self::Fminimax
            | Self::Fgoalattain => "final",
            Self::Intlinprog => "iter",
            _ => "off",
        }
    }

    pub const fn accepts_display(self, display: OptimDisplay) -> bool {
        if matches!(self, Self::Ga) {
            return matches!(
                display,
                OptimDisplay::Off
                    | OptimDisplay::None
                    | OptimDisplay::Iter
                    | OptimDisplay::Final
                    | OptimDisplay::Diagnose
            );
        }
        if matches!(self, Self::Patternsearch) {
            return matches!(
                display,
                OptimDisplay::Off
                    | OptimDisplay::None
                    | OptimDisplay::Iter
                    | OptimDisplay::Final
                    | OptimDisplay::Diagnose
            );
        }
        if matches!(self, Self::Simulannealbnd) {
            return matches!(
                display,
                OptimDisplay::Off
                    | OptimDisplay::Final
                    | OptimDisplay::Iter
                    | OptimDisplay::Diagnose
            );
        }
        match self {
            Self::Fminbnd | Self::Fminunc | Self::Fseminf | Self::Generic => {
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
            Self::Linprog | Self::Intlinprog => false,
            Self::Quadprog => matches!(
                algorithm,
                OptimAlgorithm::InteriorPointConvex | OptimAlgorithm::ActiveSet
            ),
            Self::Lsqlin => matches!(
                algorithm,
                OptimAlgorithm::InteriorPoint
                    | OptimAlgorithm::TrustRegionReflective
                    | OptimAlgorithm::ActiveSet
            ),
            // Only implemented mechanisms, never algorithm aliases.
            Self::Fmincon => matches!(
                algorithm,
                OptimAlgorithm::Sqp
                    | OptimAlgorithm::InteriorPoint
                    | OptimAlgorithm::ActiveSet
                    | OptimAlgorithm::TrustRegionReflective
            ),
            Self::Fminunc => matches!(
                algorithm,
                OptimAlgorithm::QuasiNewton | OptimAlgorithm::Bfgs | OptimAlgorithm::TrustRegion
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
                    | OptimAlgorithm::TrustRegion
            ),
            _ => false,
        }
    }
}
