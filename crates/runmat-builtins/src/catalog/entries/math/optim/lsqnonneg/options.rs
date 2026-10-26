//! The actually consumed NNLS controls, resolved through the existing legacy grammar.
use super::super::{
    FminsearchOptionName as Legacy, OptimOptionField as Modern, OptimParameterName,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LsqnonnegOption {
    Tolerance,
    MaxIterations,
    Display,
}
impl LsqnonnegOption {
    pub const fn from_parameter(name: OptimParameterName) -> Option<Self> {
        Some(match name {
            OptimParameterName::Legacy(Legacy::TolX) | OptimParameterName::Modern(Modern::TolX) => {
                Self::Tolerance
            }
            OptimParameterName::Legacy(Legacy::MaxIter)
            | OptimParameterName::Modern(Modern::MaxIter) => Self::MaxIterations,
            OptimParameterName::Legacy(Legacy::Display)
            | OptimParameterName::Modern(Modern::Display) => Self::Display,
            _ => return None,
        })
    }
}
