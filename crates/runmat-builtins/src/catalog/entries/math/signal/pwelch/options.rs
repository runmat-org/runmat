//! Canonical public Welch selector identities shared by inference and execution.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PwelchKeyword {
    Parent,
    Onesided,
    Twosided,
    Centered,
    Psd,
    Power,
    Mean,
    MaxHold,
    MinHold,
    Unknown(String),
}

impl PwelchKeyword {
    pub fn decode(text: &str) -> Self {
        let normalized = text.trim().to_ascii_lowercase();
        match normalized.as_str() {
            // The Parent name-value key historically did not trim whitespace;
            // selector spellings do. Keep those distinct public boundaries.
            "parent" if text.eq_ignore_ascii_case("parent") => Self::Parent,
            "onesided" => Self::Onesided,
            "twosided" => Self::Twosided,
            "centered" => Self::Centered,
            "psd" => Self::Psd,
            "power" => Self::Power,
            "mean" => Self::Mean,
            "maxhold" => Self::MaxHold,
            "minhold" => Self::MinHold,
            _ => Self::Unknown(normalized),
        }
    }

    pub fn is_selector(&self) -> bool {
        matches!(
            self,
            Self::Onesided
                | Self::Twosided
                | Self::Centered
                | Self::Psd
                | Self::Power
                | Self::Mean
                | Self::MaxHold
                | Self::MinHold
        )
    }

    pub fn label(&self) -> &str {
        match self {
            Self::Parent => "parent",
            Self::Onesided => "onesided",
            Self::Twosided => "twosided",
            Self::Centered => "centered",
            Self::Psd => "psd",
            Self::Power => "power",
            Self::Mean => "mean",
            Self::MaxHold => "maxhold",
            Self::MinHold => "minhold",
            Self::Unknown(text) => text,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PwelchControlRole {
    Window,
    Overlap,
    Grid,
    SampleRate,
    Option,
}

/// Ordered positional/selector grammar. A source-known selector begins the
/// selector tail; subsequent inputs cannot become positional controls again.
#[derive(Default)]
pub struct PwelchControlCursor {
    positional: usize,
    options_started: bool,
}

impl PwelchControlCursor {
    pub fn next(&mut self, token: Option<&PwelchKeyword>) -> PwelchControlRole {
        if self.options_started || token.is_some_and(PwelchKeyword::is_selector) {
            self.options_started = true;
            return PwelchControlRole::Option;
        }
        let role = match self.positional {
            0 => PwelchControlRole::Window,
            1 => PwelchControlRole::Overlap,
            2 => PwelchControlRole::Grid,
            3 => PwelchControlRole::SampleRate,
            _ => PwelchControlRole::Option,
        };
        self.positional += 1;
        role
    }
}
