//! Typed spelling boundary shared by warning execution and catalog inference.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarningMode {
    On,
    Off,
    Once,
    Error,
}

impl WarningMode {
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("on") {
            Some(Self::On)
        } else if text.eq_ignore_ascii_case("off") {
            Some(Self::Off)
        } else if text.eq_ignore_ascii_case("once") {
            Some(Self::Once)
        } else if text.eq_ignore_ascii_case("error") {
            Some(Self::Error)
        } else {
            None
        }
    }

    pub fn keyword(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
            Self::Once => "once",
            Self::Error => "error",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarningCommand {
    SetMode(WarningMode),
    Default,
    Reset,
    Query,
    Status,
    Backtrace,
}

impl WarningCommand {
    pub fn parse(text: &str) -> Option<Self> {
        if let Some(mode) = WarningMode::parse(text) {
            return Some(Self::SetMode(mode));
        }
        let text = text.trim();
        if text.eq_ignore_ascii_case("default") {
            Some(Self::Default)
        } else if text.eq_ignore_ascii_case("reset") {
            Some(Self::Reset)
        } else if text.eq_ignore_ascii_case("query") {
            Some(Self::Query)
        } else if text.eq_ignore_ascii_case("status") {
            Some(Self::Status)
        } else if text.eq_ignore_ascii_case("backtrace") {
            Some(Self::Backtrace)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarningDisplayMode {
    Backtrace,
    Verbose,
}

impl WarningDisplayMode {
    pub fn keyword(self) -> &'static str {
        match self {
            Self::Backtrace => "backtrace",
            Self::Verbose => "verbose",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarningStateSetting {
    Mode(WarningMode),
    Default,
}

impl WarningStateSetting {
    pub fn parse(text: &str) -> Option<Self> {
        if let Some(mode) = WarningMode::parse(text) {
            Some(Self::Mode(mode))
        } else if text.trim().eq_ignore_ascii_case("default") {
            Some(Self::Default)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarningTarget<'a> {
    All,
    Last,
    Backtrace,
    Verbose,
    Identifier(&'a str),
}

impl<'a> WarningTarget<'a> {
    pub fn parse(text: &'a str) -> Self {
        let trimmed = text.trim();
        if trimmed.eq_ignore_ascii_case("all") {
            Self::All
        } else if trimmed.eq_ignore_ascii_case("last") {
            Self::Last
        } else if trimmed.eq_ignore_ascii_case("backtrace") {
            Self::Backtrace
        } else if trimmed.eq_ignore_ascii_case("verbose") {
            Self::Verbose
        } else {
            Self::Identifier(trimmed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_mode_and_target_spelling_boundaries() {
        assert_eq!(
            WarningCommand::parse(" On "),
            Some(WarningCommand::SetMode(WarningMode::On))
        );
        assert_eq!(
            WarningCommand::parse(" BaCkTrAcE "),
            Some(WarningCommand::Backtrace)
        );
        assert_eq!(WarningMode::parse(" oNcE "), Some(WarningMode::Once));
        assert_eq!(WarningMode::parse("default"), None);
        assert_eq!(
            WarningStateSetting::parse(" DeFaUlT "),
            Some(WarningStateSetting::Default)
        );
        assert_eq!(
            WarningStateSetting::parse(" oFf "),
            Some(WarningStateSetting::Mode(WarningMode::Off))
        );
        assert_eq!(WarningStateSetting::parse("reset"), None);
        assert_eq!(WarningCommand::parse("some warning"), None);
        assert_eq!(WarningTarget::parse(" ALL "), WarningTarget::All);
        assert_eq!(WarningTarget::parse(" LaSt "), WarningTarget::Last);
        assert_eq!(
            WarningTarget::parse(" BACKTRACE "),
            WarningTarget::Backtrace
        );
        assert_eq!(WarningTarget::parse(" verbose "), WarningTarget::Verbose);
        assert_eq!(
            WarningTarget::parse("  RunMat:x  "),
            WarningTarget::Identifier("RunMat:x")
        );
        assert_eq!(WarningTarget::parse(""), WarningTarget::Identifier(""));
    }
}
