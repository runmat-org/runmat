use std::cmp::Ordering;

use runmat_value::IntValue;

/// One exact component of a grouping key.
///
#[derive(Clone, Debug)]
pub(crate) enum KeyAtom {
    Missing,
    Logical(bool),
    Number(f64),
    Integer(IntValue),
    CalendarDuration(f64, f64),
    Text(String),
}

impl KeyAtom {
    pub(crate) fn label(&self) -> String {
        match self {
            Self::Missing => "<missing>".into(),
            Self::Logical(flag) => flag.to_string(),
            Self::Number(value) => format_number(*value),
            Self::Integer(value) => format_integer(value),
            Self::CalendarDuration(months, days) => format!("{months}mo {days}d"),
            Self::Text(text) => text.clone(),
        }
    }

    fn rank(&self) -> u8 {
        match self {
            Self::Logical(_) => 0,
            Self::Number(_) => 1,
            Self::Integer(_) => 2,
            Self::CalendarDuration(_, _) => 3,
            Self::Text(_) => 4,
            Self::Missing => 5,
        }
    }
}

impl PartialEq for KeyAtom {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for KeyAtom {}

impl PartialOrd for KeyAtom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KeyAtom {
    fn cmp(&self, other: &Self) -> Ordering {
        let rank = self.rank().cmp(&other.rank());
        if rank != Ordering::Equal {
            return rank;
        }
        match (self, other) {
            (Self::Missing, Self::Missing) => Ordering::Equal,
            (Self::Logical(left), Self::Logical(right)) => left.cmp(right),
            (Self::Number(left), Self::Number(right)) => {
                left.partial_cmp(right).unwrap_or(Ordering::Equal)
            }
            (Self::Integer(left), Self::Integer(right)) => compare_integers(left, right),
            (
                Self::CalendarDuration(left_months, left_days),
                Self::CalendarDuration(right_months, right_days),
            ) => left_months
                .partial_cmp(right_months)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left_days.partial_cmp(right_days).unwrap_or(Ordering::Equal)),
            (Self::Text(left), Self::Text(right)) => left.cmp(right),
            _ => Ordering::Equal,
        }
    }
}

fn compare_integers(left: &IntValue, right: &IntValue) -> Ordering {
    let left = integer_sign_and_magnitude(left);
    let right = integer_sign_and_magnitude(right);
    match (left.0, right.0) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => left.1.cmp(&right.1),
        (true, true) => right.1.cmp(&left.1),
    }
}

fn integer_sign_and_magnitude(value: &IntValue) -> (bool, u64) {
    match value {
        IntValue::I8(value) => (*value < 0, value.unsigned_abs() as u64),
        IntValue::I16(value) => (*value < 0, value.unsigned_abs() as u64),
        IntValue::I32(value) => (*value < 0, value.unsigned_abs() as u64),
        IntValue::I64(value) => (*value < 0, value.unsigned_abs()),
        IntValue::U8(value) => (false, *value as u64),
        IntValue::U16(value) => (false, *value as u64),
        IntValue::U32(value) => (false, *value as u64),
        IntValue::U64(value) => (false, *value),
    }
}

pub(crate) fn format_integer(value: &IntValue) -> String {
    match value {
        IntValue::I8(value) => value.to_string(),
        IntValue::I16(value) => value.to_string(),
        IntValue::I32(value) => value.to_string(),
        IntValue::I64(value) => value.to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}
