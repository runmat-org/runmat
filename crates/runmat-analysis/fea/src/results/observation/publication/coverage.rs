//! One counted index owns declaration kind and completion; history samples never discharge histories.
use super::*;
use crate::topology::storage::{reserve, slots, SortedValues};

pub(super) struct PublicationCoverage<'a>(SortedValues<&'a str, (bool, bool)>);
impl<'a> PublicationCoverage<'a> {
    pub(super) fn resident_bytes(count: usize) -> AnalysisExecutionResult<u64> {
        slots::<(&str, (bool, bool))>(count).map_err(invalid)
    }
    pub(super) fn derive(
        declarations: &'a [ObservationRequest],
        plan: &ResolvedObservationPlan,
    ) -> AnalysisExecutionResult<Self> {
        if declarations.len() != plan.observations.len() {
            return Err(invalid(
                "authored declaration coverage differs from resolved plan",
            ));
        }
        Self::resident_bytes(declarations.len())?;
        let mut entries = reserve(declarations.len()).map_err(invalid)?;
        for (declaration, resolved) in declarations.iter().zip(&plan.observations) {
            let spec = declaration.spec().ok_or_else(|| {
                invalid("diagnostic/monitor publication is not supported by this catalog")
            })?;
            if spec != &resolved.spec {
                return Err(invalid(
                    "authored declaration differs from resolved plan or duplicates ID",
                ));
            }
            entries.push((
                spec.observation_id.0.as_str(),
                (matches!(declaration, ObservationRequest::History(_)), false),
            ));
        }
        Ok(Self(SortedValues::from_entries(entries).map_err(invalid)?))
    }
    pub(super) fn sample(&mut self, id: &str) -> AnalysisExecutionResult<bool> {
        match self.0.get_mut(id) {
            Some((true, _)) => Ok(false),
            Some((false, seen)) if !*seen => {
                *seen = true;
                Ok(true)
            }
            _ => Err(invalid(
                "sample duplicates or contains an undeclared observation",
            )),
        }
    }
    pub(super) fn history(&mut self, id: &str) -> AnalysisExecutionResult<bool> {
        match self.0.get_mut(id) {
            Some((true, seen)) if !*seen => {
                *seen = true;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    pub(super) fn complete(&self) -> bool {
        self.0.entries().all(|entry| entry.1 .1)
    }
}
#[cfg(test)]
mod tests;
