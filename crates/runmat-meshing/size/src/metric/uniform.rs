//! Spatially constant resolution is owned by the shared metric field.
use super::*;

impl ResolvedMetricField {
    /// Return one exact evaluation when no entity-scoped contribution exists.
    /// A validated grading ratio is at least one; equal positive characteristic
    /// lengths therefore form a fixed point of conservative grading on any graph.
    /// This is a construction-time result, never a serialized trust flag.
    pub fn uniform_evaluation(
        &self,
    ) -> Result<Option<ResolvedMetricEvaluation>, MetricContractError> {
        if !self.request.contributions.is_empty() {
            return Ok(None);
        }
        let evaluation = self.resolve(&BTreeSet::new())?;
        evaluation.metric.conservative_minimum_length_m()?;
        Ok(Some(evaluation))
    }
}
