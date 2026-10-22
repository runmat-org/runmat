//! Public dateshift control grammar, shared by Runtime and static inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DateShiftBoundary {
    Start,
    End,
    DayOfWeek,
}
impl DateShiftBoundary {
    pub fn parse_text(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "start" => Some(Self::Start),
            "end" => Some(Self::End),
            "dayofweek" => Some(Self::DayOfWeek),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DateShiftUnit {
    Year,
    Quarter,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
}
impl DateShiftUnit {
    pub fn parse_text(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "year" => Some(Self::Year),
            "quarter" => Some(Self::Quarter),
            "month" => Some(Self::Month),
            "week" => Some(Self::Week),
            "day" => Some(Self::Day),
            "hour" => Some(Self::Hour),
            "minute" => Some(Self::Minute),
            "second" => Some(Self::Second),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DateShiftRuleKeyword {
    Current,
    Next,
    Previous,
    Nearest,
}
impl DateShiftRuleKeyword {
    pub fn parse_text(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "current" => Some(Self::Current),
            "next" => Some(Self::Next),
            "previous" => Some(Self::Previous),
            "nearest" => Some(Self::Nearest),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DateShiftDayTarget {
    Sunday,
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Weekday,
    Weekend,
}
impl DateShiftDayTarget {
    pub fn parse_text(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sun" | "sunday" => Some(Self::Sunday),
            "mon" | "monday" => Some(Self::Monday),
            "tue" | "tues" | "tuesday" => Some(Self::Tuesday),
            "wed" | "wednesday" => Some(Self::Wednesday),
            "thu" | "thur" | "thurs" | "thursday" => Some(Self::Thursday),
            "fri" | "friday" => Some(Self::Friday),
            "sat" | "saturday" => Some(Self::Saturday),
            "weekday" => Some(Self::Weekday),
            "weekend" => Some(Self::Weekend),
            _ => None,
        }
    }
}

// Exceeds the number of any supported weekdays over Chrono's complete date
// range; keeps the bounded whole-week offset representable as a TimeDelta.
pub const MAX_DATESHIFT_DAY_OCCURRENCE: u64 = 200_000_000;

/// Exact double control conversion. The rounded binary64 i64::MAX is 2^63,
/// so the upper endpoint is exclusive; -2^63 remains inclusive.
pub fn dateshift_exact_f64_to_i64(value: f64) -> Option<i64> {
    const LOWER: f64 = -9_223_372_036_854_775_808.0;
    const UPPER: f64 = 9_223_372_036_854_775_808.0;
    (value.is_finite() && value.fract() == 0.0 && (LOWER..UPPER).contains(&value))
        .then_some(value as i64)
}
