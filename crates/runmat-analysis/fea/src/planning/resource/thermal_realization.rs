//! Realized excitation inventory and lookup payload before thermal assembly.
use runmat_analysis_core::{
    AnalysisModel, ExcitationCaseId, IndependentDomain, PhysicalSignal, ProcedureStep,
};
use serde::{
    ser::{SerializeSeq, SerializeTuple},
    Serialize, Serializer,
};

use super::{ByteCount, Count, ResourcePlanningError};
use crate::excitation::{active_actions, ActiveAction, RealizedExcitation};

#[derive(Clone, Copy, Debug)]
pub struct ThermalRealizationResourceFacts {
    pub active_actions: Count,
    pub case_memberships: Count,
    pub vector_bytes: ByteCount,
    pub metadata_estimated_bytes: ByteCount,
    pub signal_lookup_bytes: ByteCount,
}

impl ThermalRealizationResourceFacts {
    pub fn derive(
        model: &AnalysisModel,
        step: &ProcedureStep,
    ) -> Result<Self, ResourcePlanningError> {
        if !matches!(
            step.independent_domain,
            IndependentDomain::Static | IndependentDomain::TimeSeconds { .. }
        ) {
            return Err(ResourcePlanningError::Invalid(
                "thermal realization requires static or time samples".into(),
            ));
        }
        let active = active_actions(model, Some(step)).map_err(invalid)?;
        let mut actions = 0_u64;
        let mut memberships = 0_u64;
        for selected in active.clone() {
            actions = add(actions, 1)?;
            memberships = add(memberships, selected.case_ids().count() as u64)?;
        }
        // One streaming estimate over all selected metadata avoids per-action fixed overhead.
        // Full authored targets/frames include nested resolved entities and assembly paths.
        // Static/time samples add no heap-owned parameter name. Numerical values fit in slots.
        let metadata = runmat_canonical_codec::serialized_materialization_resident_bytes_estimate(
            &Metadata(active),
            crate::execution::THERMAL_WORKLOAD_LIMITS,
        )
        .map_err(invalid)?;
        Ok(Self {
            active_actions: Count(actions),
            case_memberships: Count(memberships),
            vector_bytes: ByteCount(add(
                scale(actions, std::mem::size_of::<RealizedExcitation>() as u64)?,
                scale(memberships, std::mem::size_of::<ExcitationCaseId>() as u64)?,
            )?),
            metadata_estimated_bytes: ByteCount(metadata),
            signal_lookup_bytes: ByteCount(scale(
                model.signals.len() as u64,
                std::mem::size_of::<&PhysicalSignal>() as u64,
            )?),
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        add(
            add(self.vector_bytes.0, self.metadata_estimated_bytes.0)?,
            self.signal_lookup_bytes.0,
        )
        .map(ByteCount)
    }
}

fn scale(count: u64, width: u64) -> Result<u64, ResourcePlanningError> {
    count
        .checked_mul(width)
        .ok_or(ResourcePlanningError::Overflow(
            "thermal realization payload",
        ))
}

fn add(left: u64, right: u64) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(
            "thermal realization payload",
        ))
}

fn invalid(error: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(error.to_string())
}

struct Metadata<I>(I);
impl<'a, I: Iterator<Item = ActiveAction<'a>> + Clone> Serialize for Metadata<I> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        for selected in self.0.clone() {
            sequence.serialize_element(&SelectedMetadata(selected))?;
        }
        sequence.end()
    }
}

struct SelectedMetadata<'a>(ActiveAction<'a>);
impl Serialize for SelectedMetadata<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut pair = serializer.serialize_tuple(2)?;
        pair.serialize_element(self.0.action.excitation())?;
        pair.serialize_element(&Cases(self.0))?;
        pair.end()
    }
}

struct Cases<'a>(ActiveAction<'a>);
impl Serialize for Cases<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        for id in self.0.case_ids() {
            sequence.serialize_element(id)?;
        }
        sequence.end()
    }
}

#[cfg(test)]
mod tests;
