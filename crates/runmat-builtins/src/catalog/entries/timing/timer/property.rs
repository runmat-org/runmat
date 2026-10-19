//! Typed timer property and public option spelling boundaries.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimerProperty {
    TimerFcn,
    StartFcn,
    StopFcn,
    ErrorFcn,
    Period,
    StartDelay,
    TasksToExecute,
    BusyMode,
    ExecutionMode,
    Name,
    Tag,
    ObjectVisibility,
    UserData,
    AveragePeriod,
    InstantPeriod,
    Running,
    TasksExecuted,
    Type,
}

impl TimerProperty {
    pub const ALL: [Self; 18] = [
        Self::TimerFcn,
        Self::StartFcn,
        Self::StopFcn,
        Self::ErrorFcn,
        Self::Period,
        Self::StartDelay,
        Self::TasksToExecute,
        Self::BusyMode,
        Self::ExecutionMode,
        Self::Name,
        Self::Tag,
        Self::ObjectVisibility,
        Self::UserData,
        Self::AveragePeriod,
        Self::InstantPeriod,
        Self::Running,
        Self::TasksExecuted,
        Self::Type,
    ];

    pub fn parse(spelling: &str) -> Option<Self> {
        let spelling = spelling.trim();
        Self::ALL
            .into_iter()
            .find(|property| property.name().eq_ignore_ascii_case(spelling))
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::TimerFcn => "TimerFcn",
            Self::StartFcn => "StartFcn",
            Self::StopFcn => "StopFcn",
            Self::ErrorFcn => "ErrorFcn",
            Self::Period => "Period",
            Self::StartDelay => "StartDelay",
            Self::TasksToExecute => "TasksToExecute",
            Self::BusyMode => "BusyMode",
            Self::ExecutionMode => "ExecutionMode",
            Self::Name => "Name",
            Self::Tag => "Tag",
            Self::ObjectVisibility => "ObjectVisibility",
            Self::UserData => "UserData",
            Self::AveragePeriod => "AveragePeriod",
            Self::InstantPeriod => "InstantPeriod",
            Self::Running => "Running",
            Self::TasksExecuted => "TasksExecuted",
            Self::Type => "Type",
        }
    }

    pub const fn is_readonly(self) -> bool {
        matches!(
            self,
            Self::AveragePeriod
                | Self::InstantPeriod
                | Self::Running
                | Self::TasksExecuted
                | Self::Type
        )
    }

    pub const fn is_numeric(self) -> bool {
        matches!(self, Self::Period | Self::StartDelay | Self::TasksToExecute)
    }

    pub const fn frozen_while_running(self) -> bool {
        matches!(
            self,
            Self::BusyMode | Self::ExecutionMode | Self::StartDelay
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimerBusyMode {
    Drop,
    Error,
    Queue,
}

impl TimerBusyMode {
    pub fn parse(text: &str) -> Option<Self> {
        if text.trim().eq_ignore_ascii_case("drop") {
            Some(Self::Drop)
        } else if text.trim().eq_ignore_ascii_case("error") {
            Some(Self::Error)
        } else if text.trim().eq_ignore_ascii_case("queue") {
            Some(Self::Queue)
        } else {
            None
        }
    }

    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Drop => "drop",
            Self::Error => "error",
            Self::Queue => "queue",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimerExecutionMode {
    SingleShot,
    FixedRate,
    FixedDelay,
    FixedSpacing,
}

impl TimerExecutionMode {
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("singleShot") {
            Some(Self::SingleShot)
        } else if text.eq_ignore_ascii_case("fixedRate") {
            Some(Self::FixedRate)
        } else if text.eq_ignore_ascii_case("fixedDelay") {
            Some(Self::FixedDelay)
        } else if text.eq_ignore_ascii_case("fixedSpacing") {
            Some(Self::FixedSpacing)
        } else {
            None
        }
    }

    pub const fn spelling(self) -> &'static str {
        match self {
            Self::SingleShot => "singleShot",
            Self::FixedRate => "fixedRate",
            Self::FixedDelay => "fixedDelay",
            Self::FixedSpacing => "fixedSpacing",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimerVisibility {
    On,
    Off,
}

impl TimerVisibility {
    pub fn parse(text: &str) -> Option<Self> {
        if text.trim().eq_ignore_ascii_case("on") {
            Some(Self::On)
        } else if text.trim().eq_ignore_ascii_case("off") {
            Some(Self::Off)
        } else {
            None
        }
    }

    pub const fn spelling(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_and_modes_decode_only_public_spellings() {
        assert_eq!(
            TimerProperty::parse("  tImErFcN  "),
            Some(TimerProperty::TimerFcn)
        );
        assert_eq!(TimerProperty::parse("unknown"), None);
        assert!(TimerProperty::Running.is_readonly());
        assert!(TimerProperty::Period.is_numeric());
        assert_eq!(TimerBusyMode::parse(" QUEUE "), Some(TimerBusyMode::Queue));
        assert_eq!(TimerBusyMode::parse("defer"), None);
        assert_eq!(
            TimerExecutionMode::parse("FixedSpacing"),
            Some(TimerExecutionMode::FixedSpacing)
        );
        assert_eq!(TimerExecutionMode::parse("fixed space"), None);
        assert_eq!(TimerVisibility::parse("OFF"), Some(TimerVisibility::Off));
        assert_eq!(TimerVisibility::parse("hidden"), None);
    }
}
