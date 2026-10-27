#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KsdensityOptionName {
    Bandwidth,
    NumPoints,
    Function,
    Kernel,
    Support,
}
impl KsdensityOptionName {
    pub fn from_public(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Bandwidth") {
            Some(Self::Bandwidth)
        } else if name.eq_ignore_ascii_case("NumPoints") {
            Some(Self::NumPoints)
        } else if name.eq_ignore_ascii_case("Function") {
            Some(Self::Function)
        } else if name.eq_ignore_ascii_case("Kernel") {
            Some(Self::Kernel)
        } else if name.eq_ignore_ascii_case("Support") {
            Some(Self::Support)
        } else {
            None
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KsdensityFunction {
    Pdf,
    Cdf,
    Survivor,
}
impl KsdensityFunction {
    pub fn from_public(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("pdf") {
            Some(Self::Pdf)
        } else if name.eq_ignore_ascii_case("cdf") {
            Some(Self::Cdf)
        } else if name.eq_ignore_ascii_case("survivor") {
            Some(Self::Survivor)
        } else {
            None
        }
    }
}
