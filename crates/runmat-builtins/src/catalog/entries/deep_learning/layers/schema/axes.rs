//! Semantic dimension labels and reversible physical-axis mappings.
//!
//! This schema describes axes, not tensor storage. In particular, changing an
//! `DeepLearningAxisLayout` never transposes or relabels the bytes of an existing array.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeepLearningAxisLabel {
    Channel,
    Batch,
    Time,
    Spatial,
    Unspecified,
}

impl DeepLearningAxisLabel {
    pub const fn label(self) -> char {
        match self {
            Self::Channel => 'C',
            Self::Batch => 'B',
            Self::Time => 'T',
            Self::Spatial => 'S',
            Self::Unspecified => 'U',
        }
    }

    fn from_label(label: char) -> Option<Self> {
        match label {
            'C' => Some(Self::Channel),
            'B' => Some(Self::Batch),
            'T' => Some(Self::Time),
            'S' => Some(Self::Spatial),
            'U' => Some(Self::Unspecified),
            _ => None,
        }
    }

    const fn unique(self) -> bool {
        matches!(self, Self::Channel | Self::Batch | Self::Time)
    }
}

/// The input layer decides an unformatted array's documented physical order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepLearningInputKind {
    Feature,
    VectorSequence,
    Spatial1d,
    SpatialSequence1d,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepLearningAxisError {
    RankMismatch {
        labels: usize,
        rank: usize,
    },
    InvalidLabel(char),
    DuplicateLabel(char),
    NoDefault {
        kind: DeepLearningInputKind,
        rank: usize,
    },
    MissingAxis(char),
    UnexpectedAxis(char),
    SpatialAxisCount {
        expected: usize,
        actual: usize,
    },
    IncompatiblePermutation,
}

impl fmt::Display for DeepLearningAxisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch { labels, rank } => {
                write!(formatter, "format has {labels} labels for rank-{rank} data")
            }
            Self::InvalidLabel(label) => write!(formatter, "invalid axis label '{label}'"),
            Self::DuplicateLabel(label) => write!(formatter, "axis label '{label}' is repeated"),
            Self::NoDefault { kind, rank } => {
                write!(
                    formatter,
                    "no unformatted {kind:?} input convention for rank {rank}"
                )
            }
            Self::MissingAxis(label) => write!(formatter, "required axis '{label}' is absent"),
            Self::UnexpectedAxis(label) => {
                write!(formatter, "axis '{label}' is not valid for this input kind")
            }
            Self::SpatialAxisCount { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} spatial axes, found {actual}"
                )
            }
            Self::IncompatiblePermutation => {
                write!(formatter, "target axes do not match the source axes")
            }
        }
    }
}

impl std::error::Error for DeepLearningAxisError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepLearningAxisLayout {
    axes: Vec<DeepLearningAxisLabel>,
}

impl DeepLearningAxisLayout {
    pub fn parse(format: &str, rank: usize) -> Result<Self, DeepLearningAxisError> {
        let labels = format.chars().collect::<Vec<_>>();
        if labels.len() != rank {
            return Err(DeepLearningAxisError::RankMismatch {
                labels: labels.len(),
                rank,
            });
        }
        let axes = labels
            .into_iter()
            .map(|label| {
                DeepLearningAxisLabel::from_label(label)
                    .ok_or(DeepLearningAxisError::InvalidLabel(label))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_axes(&axes)
    }

    /// Construct an internal layout from semantic labels without serializing
    /// them through display text. Public format parsing delegates here too.
    pub fn from_axes(axes: &[DeepLearningAxisLabel]) -> Result<Self, DeepLearningAxisError> {
        let mut admitted = Vec::with_capacity(axes.len());
        for &axis in axes {
            if axis.unique() && admitted.contains(&axis) {
                return Err(DeepLearningAxisError::DuplicateLabel(axis.label()));
            }
            admitted.push(axis);
        }
        Ok(Self { axes: admitted })
    }

    /// Empty format means unformatted input; its convention is selected by
    /// the declared input layer, never by matching coincidental dimensions.
    pub fn for_input(
        format: &str,
        kind: DeepLearningInputKind,
        rank: usize,
    ) -> Result<Self, DeepLearningAxisError> {
        if !format.is_empty() {
            let layout = Self::parse(format, rank)?;
            layout.validate_input_kind(kind)?;
            return Ok(layout);
        }
        use DeepLearningAxisLabel::{Batch, Channel, Spatial, Time};
        let default: &[DeepLearningAxisLabel] = match (kind, rank) {
            (DeepLearningInputKind::Feature, 1) => &[Channel],
            (DeepLearningInputKind::Feature, 2) => &[Batch, Channel],
            (DeepLearningInputKind::VectorSequence, 2) => &[Channel, Time],
            (DeepLearningInputKind::VectorSequence, 3) => &[Channel, Batch, Time],
            (DeepLearningInputKind::Spatial1d, 2) => &[Spatial, Channel],
            (DeepLearningInputKind::Spatial1d, 3) => &[Spatial, Channel, Batch],
            (DeepLearningInputKind::SpatialSequence1d, 3) => &[Spatial, Channel, Time],
            (DeepLearningInputKind::SpatialSequence1d, 4) => &[Spatial, Channel, Batch, Time],
            _ => return Err(DeepLearningAxisError::NoDefault { kind, rank }),
        };
        let layout = Self::from_axes(default)?;
        layout.validate_input_kind(kind)?;
        Ok(layout)
    }

    pub fn validate_input_kind(
        &self,
        kind: DeepLearningInputKind,
    ) -> Result<(), DeepLearningAxisError> {
        self.require(DeepLearningAxisLabel::Channel)?;
        let needs_time = matches!(
            kind,
            DeepLearningInputKind::VectorSequence | DeepLearningInputKind::SpatialSequence1d
        );
        let has_time = self.index(DeepLearningAxisLabel::Time).is_some();
        if needs_time && !has_time {
            return Err(DeepLearningAxisError::MissingAxis('T'));
        }
        if !needs_time && has_time {
            return Err(DeepLearningAxisError::UnexpectedAxis('T'));
        }
        let expected_spatial = usize::from(matches!(
            kind,
            DeepLearningInputKind::Spatial1d | DeepLearningInputKind::SpatialSequence1d
        ));
        let actual_spatial = self
            .axes
            .iter()
            .filter(|axis| **axis == DeepLearningAxisLabel::Spatial)
            .count();
        if actual_spatial != expected_spatial {
            return Err(DeepLearningAxisError::SpatialAxisCount {
                expected: expected_spatial,
                actual: actual_spatial,
            });
        }
        Ok(())
    }

    pub fn labels(&self) -> String {
        self.axes.iter().map(|axis| axis.label()).collect()
    }

    pub fn axes(&self) -> &[DeepLearningAxisLabel] {
        &self.axes
    }

    pub fn index(&self, kind: DeepLearningAxisLabel) -> Option<usize> {
        self.axes.iter().position(|axis| *axis == kind)
    }

    pub fn require(&self, kind: DeepLearningAxisLabel) -> Result<usize, DeepLearningAxisError> {
        self.index(kind)
            .ok_or(DeepLearningAxisError::MissingAxis(kind.label()))
    }

    /// Maps each target physical axis to its source physical axis. Repeated
    /// spatial/unspecified labels are paired by occurrence, not conflated.
    pub fn permutation_to(
        &self,
        target: &Self,
    ) -> Result<DeepLearningAxisPermutation, DeepLearningAxisError> {
        if self.axes.len() != target.axes.len() {
            return Err(DeepLearningAxisError::IncompatiblePermutation);
        }
        let mut used = vec![false; self.axes.len()];
        let mut target_to_source = Vec::with_capacity(target.axes.len());
        for target_kind in &target.axes {
            let source = self
                .axes
                .iter()
                .enumerate()
                .find(|(index, kind)| !used[*index] && *kind == target_kind)
                .map(|(index, _)| index)
                .ok_or(DeepLearningAxisError::IncompatiblePermutation)?;
            used[source] = true;
            target_to_source.push(source);
        }
        Ok(DeepLearningAxisPermutation { target_to_source })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepLearningAxisPermutation {
    target_to_source: Vec<usize>,
}

impl DeepLearningAxisPermutation {
    pub fn target_to_source(&self) -> &[usize] {
        &self.target_to_source
    }

    pub fn inverse(&self) -> Self {
        let mut source_to_target = vec![0; self.target_to_source.len()];
        for (target, source) in self.target_to_source.iter().copied().enumerate() {
            source_to_target[source] = target;
        }
        Self {
            target_to_source: source_to_target,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_and_default_axes_are_distinct() {
        assert_eq!(
            DeepLearningAxisLayout::for_input("", DeepLearningInputKind::Feature, 2)
                .unwrap()
                .labels(),
            "BC"
        );
        assert_eq!(
            DeepLearningAxisLayout::for_input("CB", DeepLearningInputKind::Feature, 2)
                .unwrap()
                .labels(),
            "CB"
        );
        assert_eq!(
            DeepLearningAxisLayout::for_input("", DeepLearningInputKind::VectorSequence, 3)
                .unwrap()
                .labels(),
            "CBT"
        );
        assert_eq!(
            DeepLearningAxisLayout::for_input("", DeepLearningInputKind::Spatial1d, 3)
                .unwrap()
                .labels(),
            "SCB"
        );
        assert_eq!(
            DeepLearningAxisLayout::for_input("", DeepLearningInputKind::SpatialSequence1d, 4)
                .unwrap()
                .labels(),
            "SCBT"
        );
    }

    #[test]
    fn malformed_labels_and_unsupported_defaults_reject() {
        use DeepLearningAxisLabel::{Batch, Channel, Spatial};
        assert_eq!(
            DeepLearningAxisLayout::from_axes(&[Batch, Channel, Spatial])
                .unwrap()
                .axes(),
            &[Batch, Channel, Spatial]
        );
        assert_eq!(
            DeepLearningAxisLayout::from_axes(&[Channel, Batch, Channel]),
            Err(DeepLearningAxisError::DuplicateLabel('C'))
        );
        assert_eq!(
            DeepLearningAxisLayout::parse("CC", 2),
            Err(DeepLearningAxisError::DuplicateLabel('C'))
        );
        assert_eq!(
            DeepLearningAxisLayout::parse("XU", 2),
            Err(DeepLearningAxisError::InvalidLabel('X'))
        );
        assert!(matches!(
            DeepLearningAxisLayout::parse("CB", 3),
            Err(DeepLearningAxisError::RankMismatch { .. })
        ));
        assert!(matches!(
            DeepLearningAxisLayout::for_input("", DeepLearningInputKind::Feature, 3),
            Err(DeepLearningAxisError::NoDefault { .. })
        ));
        assert_eq!(
            DeepLearningAxisLayout::for_input("UU", DeepLearningInputKind::Feature, 2),
            Err(DeepLearningAxisError::MissingAxis('C'))
        );
        assert_eq!(
            DeepLearningAxisLayout::for_input("CBT", DeepLearningInputKind::Feature, 3),
            Err(DeepLearningAxisError::UnexpectedAxis('T'))
        );
    }

    #[test]
    fn repeated_spatial_and_unknown_axes_have_reversible_mapping() {
        let source = DeepLearningAxisLayout::parse("SSCBUU", 6).unwrap();
        let target = DeepLearningAxisLayout::parse("UCSUBS", 6).unwrap();
        let permutation = source.permutation_to(&target).unwrap();
        assert_eq!(permutation.target_to_source(), &[4, 2, 0, 5, 3, 1]);
        assert_eq!(
            permutation.inverse().target_to_source(),
            &[2, 5, 1, 4, 0, 3]
        );
        assert!(source
            .permutation_to(&DeepLearningAxisLayout::parse("CBSUUU", 6).unwrap())
            .is_err());
    }
}
