//! Canonical semantic controls accepted by the one-sample KS test.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KstestOptionName {
    Alpha,
    Cdf,
    Tail,
}

impl KstestOptionName {
    pub fn from_public(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("alpha") {
            Some(Self::Alpha)
        } else if text.eq_ignore_ascii_case("cdf") {
            Some(Self::Cdf)
        } else if text.eq_ignore_ascii_case("tail") {
            Some(Self::Tail)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KstestTail {
    Unequal,
    Larger,
    Smaller,
}

impl KstestTail {
    pub fn from_public(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("unequal") || text.eq_ignore_ascii_case("both") {
            Some(Self::Unequal)
        } else if text.eq_ignore_ascii_case("larger") || text.eq_ignore_ascii_case("right") {
            Some(Self::Larger)
        } else if text.eq_ignore_ascii_case("smaller") || text.eq_ignore_ascii_case("left") {
            Some(Self::Smaller)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KstestNamedCdf {
    StandardNormal,
}

impl KstestNamedCdf {
    pub fn from_public(text: &str) -> Option<Self> {
        let normalized = text
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        matches!(
            normalized.as_str(),
            "normcdf" | "normal" | "norm" | "gaussian"
        )
        .then_some(Self::StandardNormal)
    }
}
