//! Canonical field membership, independent of partition layout and execution placement.
use std::collections::BTreeSet;

use super::*;
use crate::results::field::RecoveredFieldIdentity;
use crate::results::observation::ObservationFieldSource;

#[cfg(test)]
mod tests;

/// Structural recovery includes every sample of a required definition; thermal evaluation
/// selects exactly one authored source sample. This is not a partition or CPU coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationSelectionCoordinate {
    StructuralRecovery,
    ThermalSample { ordinal: usize },
}

impl ObservationFieldCatalog {
    /// Require exact canonical membership and order. This does not replace source metadata
    /// authentication, selection integrity validation, or authored-plan reprojection.
    pub fn validate_canonical_selection(
        &self,
        plan: &ResolvedObservationPlan,
        selection: &ObservationFieldSelection,
    ) -> AnalysisExecutionResult<()> {
        let coordinate = match selection.source() {
            ObservationFieldSource::StructuralRecovery { .. } => {
                ObservationSelectionCoordinate::StructuralRecovery
            }
            ObservationFieldSource::ThermalResult { sample_ordinal, .. } => {
                ObservationSelectionCoordinate::ThermalSample {
                    ordinal: sample_ordinal,
                }
            }
        };
        let expected = self.canonical_selected_fields(plan, coordinate)?;
        if !selection
            .fields()
            .iter()
            .map(|entry| entry.field)
            .eq(expected)
        {
            return Err(invalid(
                "observation selection differs from canonical required fields",
            ));
        }
        Ok(())
    }

    /// Select in authoritative catalog order, including all and only required definitions.
    /// Source authentication and authored-plan reprojection remain separate mandatory checks.
    pub fn canonical_selected_fields(
        &self,
        plan: &ResolvedObservationPlan,
        coordinate: ObservationSelectionCoordinate,
    ) -> AnalysisExecutionResult<Vec<RecoveredFieldIdentity>> {
        let entries: &[_] = match (&self.source, coordinate) {
            (
                ObservationFieldCatalogSource::Structural { manifest, .. },
                ObservationSelectionCoordinate::StructuralRecovery,
            ) => &manifest.fields,
            (
                ObservationFieldCatalogSource::Thermal { manifest },
                ObservationSelectionCoordinate::ThermalSample { ordinal },
            ) => manifest
                .fields
                .get(ordinal)
                .ok_or_else(|| invalid("canonical selection sample is absent"))?,
            _ => {
                return Err(invalid(
                    "canonical selection coordinate differs from catalog kind",
                ))
            }
        };
        let required = plan
            .observations
            .iter()
            .map(|observation| observation.source_field_definition_identity)
            .collect::<BTreeSet<_>>();
        Ok(entries
            .iter()
            .filter(|entry| required.contains(&entry.definition.identity()))
            .map(|entry| entry.field)
            .collect())
    }
}
