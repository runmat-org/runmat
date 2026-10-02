/// Sampling of a generated cosine-sum window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowSampling {
    Symmetric,
    Periodic,
}

/// Class of the generated window, independent of the length's class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowOutputType {
    Double,
    Single,
}

/// The two public option roles. Decode text once at the language boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowOption {
    Sampling(WindowSampling),
    OutputType(WindowOutputType),
}

impl WindowOption {
    pub fn parse(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("symmetric") {
            Some(Self::Sampling(WindowSampling::Symmetric))
        } else if text.eq_ignore_ascii_case("periodic") {
            Some(Self::Sampling(WindowSampling::Periodic))
        } else if text.eq_ignore_ascii_case("double") {
            Some(Self::OutputType(WindowOutputType::Double))
        } else if text.eq_ignore_ascii_case("single") {
            Some(Self::OutputType(WindowOutputType::Single))
        } else {
            None
        }
    }
}
