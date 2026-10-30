use std::cmp::Ordering;

use runmat_value::{IntValue, NumericScalar};

/// One exact component of a grouping key.
#[derive(Clone, Debug)]
pub(crate) enum KeyAtom {
    Missing,
    Logical(bool),
    Number(f64),
    Integer(IntValue),
    CalendarDuration(f64, f64),
    Text(String),
}

/// Approximate the work of visiting and copying one typed group key. This is
/// scheduling work, not a memory reservation or a bound on map comparisons.
pub(crate) fn group_key_work_units(key: &[KeyAtom]) -> u64 {
    key.iter().fold(1u64, |units, atom| {
        let component = match atom {
            KeyAtom::Text(text) => u64::try_from(text.len()).unwrap_or(u64::MAX).max(1),
            _ => 1,
        };
        units.saturating_add(component)
    })
}

impl KeyAtom {
    pub(crate) fn from_numeric(value: NumericScalar) -> Option<Self> {
        match value {
            NumericScalar::F64(value) if value.is_nan() => None,
            NumericScalar::F64(value) => Some(Self::Number(value)),
            NumericScalar::F32(value) if value.is_nan() => None,
            NumericScalar::F32(value) => Some(Self::Number(f64::from(value))),
            value => value.into_int_value().map(Self::Integer),
        }
    }

    pub(crate) fn from_text(value: &str) -> Option<Self> {
        Some(Self::Text(value.into()))
    }

    pub(crate) fn from_string(value: &runmat_value::StringElement) -> Option<Self> {
        value.as_str().map(|text| Self::Text(text.into()))
    }

    pub(crate) fn string_label(&self) -> runmat_value::StringElement {
        match self {
            Self::Missing => runmat_value::StringElement::Missing,
            Self::Text(text) => runmat_value::StringElement::text(text.clone()),
            _ => runmat_value::StringElement::text(self.label()),
        }
    }

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
