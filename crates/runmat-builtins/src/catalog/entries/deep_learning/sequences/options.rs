//! Public padsequences controls, decoded at the source/runtime boundary.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PadOptionKey {
    DirectionAlias,
    Length,
    PaddingDirection,
    PaddingValue,
    UniformOutput,
}

impl PadOptionKey {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "paddingvalue" => Some(Self::PaddingValue),
            "paddingdirection" => Some(Self::PaddingDirection),
            "direction" => Some(Self::DirectionAlias),
            "length" => Some(Self::Length),
            "uniformoutput" => Some(Self::UniformOutput),
            _ => None,
        }
    }

    /// The lowered input spelling is retained only for user diagnostics.
    pub fn spelling(self) -> &'static str {
        match self {
            Self::PaddingValue => "paddingvalue",
            Self::PaddingDirection => "paddingdirection",
            Self::DirectionAlias => "direction",
            Self::Length => "length",
            Self::UniformOutput => "uniformoutput",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadDirection {
    Left,
    Right,
    Both,
}

impl PadDirection {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "left" => Some(Self::Left),
            "right" => Some(Self::Right),
            "both" => Some(Self::Both),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadLengthKeyword {
    Longest,
    Shortest,
}

impl PadLengthKeyword {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "longest" => Some(Self::Longest),
            "shortest" => Some(Self::Shortest),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadBooleanKeyword {
    True,
    False,
}

impl PadBooleanKeyword {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "true" | "on" | "yes" => Some(Self::True),
            "false" | "off" | "no" => Some(Self::False),
            _ => None,
        }
    }

    pub fn value(self) -> bool {
        matches!(self, Self::True)
    }
}
