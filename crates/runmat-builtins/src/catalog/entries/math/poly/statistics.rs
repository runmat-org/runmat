/// The public fields of fit statistics consumed by prediction-interval forms.
/// Field spellings are serialized structure keys; semantic consumers carry
/// this closed identity rather than reinterpreting arbitrary labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyfitStatField {
    R,
    DegreesOfFreedom,
    ResidualNorm,
}

impl PolyfitStatField {
    pub const ALL: [Self; 3] = [Self::R, Self::DegreesOfFreedom, Self::ResidualNorm];

    pub const fn public_name(self) -> &'static str {
        match self {
            Self::R => "R",
            Self::DegreesOfFreedom => "df",
            Self::ResidualNorm => "normr",
        }
    }
}
