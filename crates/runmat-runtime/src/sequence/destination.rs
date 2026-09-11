use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::Value;

pub(crate) mod endpoint;
mod path;

pub use endpoint::{PreparedSequenceEndpoint, SequenceEndpointSpec};
pub use path::{AssignmentStepSpec, PreparedSequenceDestination, SequenceDestinationBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DestinationCardinality {
    Fixed,
    Discard,
    Sequence(usize),
}

impl DestinationCardinality {
    const fn count(self) -> usize {
        match self {
            Self::Fixed | Self::Discard => 1,
            Self::Sequence(count) => count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DestinationRange {
    start: usize,
    len: usize,
}

impl DestinationRange {
    pub const fn start(self) -> usize {
        self.start
    }

    pub const fn len(self) -> usize {
        self.len
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestinationLayout {
    ranges: Vec<DestinationRange>,
    total: usize,
}

impl DestinationLayout {
    pub fn new(
        cardinalities: impl IntoIterator<Item = DestinationCardinality>,
    ) -> Result<Self, RuntimeError> {
        let mut ranges = Vec::new();
        let mut total = 0usize;
        for cardinality in cardinalities {
            let len = cardinality.count();
            ranges.push(DestinationRange { start: total, len });
            total = total.checked_add(len).ok_or_else(|| {
                semantic_error(
                    "DestinationCardinalityOverflow",
                    "output destination cardinality exceeds the supported limit",
                )
            })?;
        }
        Ok(Self { ranges, total })
    }

    pub const fn total(&self) -> usize {
        self.total
    }

    pub fn ranges(&self) -> &[DestinationRange] {
        &self.ranges
    }

    pub fn distribute(self, mut values: Vec<Value>) -> Result<Vec<Vec<Value>>, RuntimeError> {
        if values.len() < self.total {
            return Err(semantic_error(
                "CommaSeparatedListOutputShortage",
                format!(
                    "sequence assignment requires exactly one value per destination element (expected {}, received {})",
                    self.total, values.len()
                ),
            ));
        }
        values.truncate(self.total);
        let mut remaining = values.into_iter();
        self.ranges
            .into_iter()
            .map(|range| Ok(remaining.by_ref().take(range.len).collect()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_layout_preserves_target_order_and_zero_length_sequences() {
        let layout = DestinationLayout::new([
            DestinationCardinality::Fixed,
            DestinationCardinality::Sequence(0),
            DestinationCardinality::Discard,
            DestinationCardinality::Sequence(2),
        ])
        .unwrap();
        assert_eq!(layout.total(), 4);
        assert_eq!(layout.ranges()[1].len(), 0);
        assert_eq!(layout.ranges()[3].start(), 2);

        let values = layout
            .distribute(vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
                Value::Num(5.0),
            ])
            .unwrap();
        assert_eq!(values[0], vec![Value::Num(1.0)]);
        assert!(values[1].is_empty());
        assert_eq!(values[2], vec![Value::Num(2.0)]);
        assert_eq!(values[3], vec![Value::Num(3.0), Value::Num(4.0)]);
    }

    #[test]
    fn distribution_reports_shortage_without_partial_output() {
        let layout = DestinationLayout::new([
            DestinationCardinality::Fixed,
            DestinationCardinality::Sequence(2),
        ])
        .unwrap();
        let error = layout
            .distribute(vec![Value::Num(1.0), Value::Num(2.0)])
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:CommaSeparatedListOutputShortage")
        );
        assert!(error
            .to_string()
            .contains("exactly one value per destination element (expected 3, received 2)"));
    }
}
