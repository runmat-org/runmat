//! Public ttest2 controls decode into these canonical semantic identities.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ttest2OptionName {
    Alpha,
    Dim,
    Tail,
    VarType,
}

impl Ttest2OptionName {
    pub fn from_public(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("alpha") {
            Some(Self::Alpha)
        } else if text.eq_ignore_ascii_case("dim") {
            Some(Self::Dim)
        } else if text.eq_ignore_ascii_case("tail") {
            Some(Self::Tail)
        } else if text.eq_ignore_ascii_case("vartype") {
            Some(Self::VarType)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ttest2Tail {
    Both,
    Right,
    Left,
}

impl Ttest2Tail {
    pub fn from_public(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("both") {
            Some(Self::Both)
        } else if text.eq_ignore_ascii_case("right") || text.eq_ignore_ascii_case("larger") {
            Some(Self::Right)
        } else if text.eq_ignore_ascii_case("left") || text.eq_ignore_ascii_case("smaller") {
            Some(Self::Left)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ttest2Variance {
    Equal,
    Unequal,
}

impl Ttest2Variance {
    pub fn from_public(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("equal") {
            Some(Self::Equal)
        } else if text.eq_ignore_ascii_case("unequal") {
            Some(Self::Unequal)
        } else {
            None
        }
    }
}
