/// Source of parameter uncertainty for nonlinear confidence intervals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NlparciSource {
    Jacobian,
    Covariance,
}

impl NlparciSource {
    pub fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Jacobian") {
            Some(Self::Jacobian)
        } else if name.eq_ignore_ascii_case("Covar") {
            Some(Self::Covariance)
        } else {
            None
        }
    }
}
