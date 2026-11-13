//! Exact class identities shared by cross-validation split constructors.

/// Group identity is a value, not a display string. In particular, wide
/// integer classes never pass through binary64 or decimal formatting.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) enum PartitionLabel {
    Number(u64),
    Integer(i128),
    Logical(bool),
    Text(runmat_value::StringElement),
}

impl PartitionLabel {
    pub(super) fn text(value: runmat_value::StringElement) -> Option<Self> {
        (!value.is_missing()).then_some(Self::Text(value))
    }

    pub(super) fn number(value: f64) -> Option<Self> {
        if value.is_nan() {
            None
        } else {
            // IEEE signed zero compares equal as a class label.
            Some(Self::Number(if value == 0.0 { 0 } else { value.to_bits() }))
        }
    }

    pub(super) fn integer(value: &runmat_value::IntValue) -> Self {
        Self::Integer(value.to_i128())
    }
}

#[cfg(test)]
mod tests {
    use super::PartitionLabel;
    use runmat_value::IntValue;

    #[test]
    fn wide_integer_and_zero_labels_remain_exact() {
        assert_ne!(
            PartitionLabel::integer(&IntValue::U64((1u64 << 53) + 1)),
            PartitionLabel::integer(&IntValue::U64((1u64 << 53) + 2))
        );
        assert_eq!(PartitionLabel::number(-0.0), PartitionLabel::number(0.0));
        assert_eq!(PartitionLabel::number(f64::NAN), None);
    }
}
