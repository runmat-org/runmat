//! Partition authored outputs by the analysis field they ultimately read, for executing one
//! authored study as several analyses (each publishes only the outputs it can produce).
use super::{AuthoredOutput, AuthoredOutputExpression};
use crate::{AuthoredAssessment, ObservationQuantity};

impl AuthoredOutputExpression {
    /// The analysis field this expression reads, or the output it references.
    fn root(&self) -> Result<(&str, ObservationQuantity), &str> {
        match self {
            Self::AnalysisField {
                step_id, quantity, ..
            }
            | Self::LineParameter {
                step_id, quantity, ..
            } => Ok((step_id, *quantity)),
            Self::OutputReference { output_id } => Err(output_id),
            Self::Projection { source, .. }
            | Self::Recovery { source, .. }
            | Self::SpatialReduction { source, .. }
            | Self::PortReduction { source, .. }
            | Self::SpatialSelection { source, .. }
            | Self::Interpolation { source, .. }
            | Self::Sampling { source, .. }
            | Self::SeriesReduction { source, .. } => source.root(),
        }
    }
}

/// Analysis step and field quantity an output ultimately reads, following output references.
pub fn authored_output_root(
    outputs: &[AuthoredOutput],
    output_id: &str,
) -> Option<(String, ObservationQuantity)> {
    let mut visited = std::collections::BTreeSet::new();
    let mut current = outputs.iter().find(|o| o.output_id == output_id)?;
    loop {
        if !visited.insert(current.output_id.as_str()) {
            return None;
        }
        match current.expression.root() {
            Ok((step, quantity)) => return Some((step.to_string(), quantity)),
            Err(next) => current = outputs.iter().find(|o| o.output_id == next)?,
        }
    }
}

/// Keep the outputs whose root field satisfies `keep`, and the assessments of kept outputs.
pub fn retain_authored_outputs(
    outputs: &mut Vec<AuthoredOutput>,
    assessments: &mut Vec<AuthoredAssessment>,
    keep: impl Fn(&str, ObservationQuantity) -> bool,
) {
    let all = outputs.clone();
    outputs.retain(|output| {
        authored_output_root(&all, &output.output_id)
            .is_some_and(|(step, quantity)| keep(&step, quantity))
    });
    let kept: std::collections::BTreeSet<_> = outputs.iter().map(|o| o.output_id.clone()).collect();
    assessments.retain(|assessment| kept.contains(&assessment.output_id));
}
