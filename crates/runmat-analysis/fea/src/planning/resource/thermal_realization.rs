//! Realized excitation inventory and lookup payload before thermal assembly.
use runmat_analysis_core::{
    AnalysisModel, ExcitationAction, ExcitationCaseId, FieldLocation, HeatGenerationDensity,
    IndependentDomain, PhysicalSignal, ProcedureStep, Signal, SignalKnotValue, SignalOrdinate,
    SignalRepresentation, SignalShape, SignalValue, TabulatedSignalDomain,
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
    /// Retained element scalars plus simultaneous sampling tensors. Input model storage is
    /// reserved separately; this is requested heap payload, not RSS or allocator overhead.
    pub field_payload_bytes: ByteCount,
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
        let mut field_payload = 0_u64;
        for selected in active.clone() {
            actions = add(actions, 1)?;
            memberships = add(memberships, selected.case_ids().count() as u64)?;
            if let ExcitationAction::HeatGeneration(action) = selected.action {
                let mut matching = model
                    .signals
                    .iter()
                    .filter(|signal| signal.signal_id() == &action.signal_id);
                let signal = matching
                    .next()
                    .ok_or_else(|| invalid("thermal source signal is missing"))?;
                if matching.next().is_some() {
                    return Err(invalid("thermal source signal ID is duplicated"));
                }
                if matches!(signal.shape(), SignalShape::TopologyField { .. }) {
                    let PhysicalSignal::HeatGenerationDensity(signal) = signal else {
                        return Err(invalid("thermal element density quantity mismatch"));
                    };
                    field_payload = add(field_payload, element_scalar_payload(signal)?)?;
                }
            }
        }
        // One streaming estimate over all selected metadata avoids per-action fixed overhead.
        // Full authored targets/frames include nested resolved entities and assembly paths.
        // Static/time samples add no heap-owned parameter name. Scalar numerical values fit
        // in slots; element-field vectors and sampled tensors are explicitly reserved below.
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
            field_payload_bytes: ByteCount(field_payload),
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        add(
            add(self.vector_bytes.0, self.metadata_estimated_bytes.0)?,
            add(self.signal_lookup_bytes.0, self.field_payload_bytes.0)?,
        )
        .map(ByteCount)
    }
}

fn element_scalar_payload(
    signal: &Signal<HeatGenerationDensity>,
) -> Result<u64, ResourcePlanningError> {
    let SignalShape::TopologyField {
        topology_id,
        location: FieldLocation::Element,
        component_count: 1,
    } = &signal.shape
    else {
        return Err(invalid(
            "thermal scalar field requires element location and one component",
        ));
    };
    topology_id
        .parse::<runmat_execution::Digest>()
        .map_err(invalid)?;
    let (rows, tensors) = match &signal.representation {
        SignalRepresentation::Constant(value) => (tensor_rows(value)?, 1),
        SignalRepresentation::Tabulated(table) => {
            if table.domain != TabulatedSignalDomain::TimeSeconds
                || table.knots.is_empty()
                || table.knots.iter().any(|knot| !knot.abscissa.is_finite())
                || table
                    .knots
                    .windows(2)
                    .any(|pair| pair[0].abscissa >= pair[1].abscissa)
            {
                return Err(invalid(
                    "thermal scalar field table requires ordered finite time knots",
                ));
            }
            let mut rows = None;
            for knot in &table.knots {
                let count = match &knot.value {
                    SignalKnotValue::Continuous(value) => ordinate_rows(value)?,
                    SignalKnotValue::Jump {
                        left_limit,
                        right_limit,
                        ..
                    } => {
                        let left = ordinate_rows(left_limit)?;
                        if left != ordinate_rows(right_limit)? {
                            return Err(invalid("thermal scalar field jump shape differs"));
                        }
                        left
                    }
                };
                if rows.is_some_and(|previous| previous != count) {
                    return Err(invalid("thermal scalar field table changes row count"));
                }
                rows = Some(count);
            }
            // Linear time interpolation may simultaneously hold both sampled endpoints
            // and its output. Static sampling holds at most two tensors, bounded by three.
            (
                rows.ok_or_else(|| invalid("thermal scalar field table has no values"))?,
                3,
            )
        }
        _ => {
            return Err(invalid(
                "thermal scalar field representation is not materialized",
            ))
        }
    };
    scalar_field_payload(rows, tensors)
}

fn tensor_rows(value: &SignalValue<HeatGenerationDensity>) -> Result<u64, ResourcePlanningError> {
    let SignalValue::Tensor { shape, values } = value else {
        return Err(invalid("thermal scalar field requires a tensor"));
    };
    let [rows, 1] = shape.as_slice() else {
        return Err(invalid("thermal scalar field requires [elements, 1] shape"));
    };
    if *rows == 0
        || *rows != values.len()
        || values
            .iter()
            .any(|value| !value.canonical_value().is_finite())
    {
        return Err(invalid(
            "thermal scalar field requires exact positive finite tensor data",
        ));
    }
    u64::try_from(*rows)
        .map_err(|_| ResourcePlanningError::Overflow("thermal scalar field row count"))
}

fn ordinate_rows(
    value: &SignalOrdinate<HeatGenerationDensity>,
) -> Result<u64, ResourcePlanningError> {
    match value {
        SignalOrdinate::Real(value) => tensor_rows(value),
        SignalOrdinate::Complex { real, imaginary } => {
            let rows = tensor_rows(real)?;
            if rows != tensor_rows(imaginary)? {
                return Err(invalid("thermal scalar field complex shapes differ"));
            }
            Ok(rows)
        }
    }
}

fn scalar_field_payload(rows: u64, tensors: u64) -> Result<u64, ResourcePlanningError> {
    // Each retained row is ComplexScalar (two f64s), each sampled tensor has separate
    // real/imaginary f64 vectors and a two-usize shape. Sum scratch across active actions
    // conservatively even though they sample sequentially. No numerical tensor is built here.
    let row_bytes = scale(rows, 2 * std::mem::size_of::<f64>() as u64)?;
    let shape_bytes = 2 * std::mem::size_of::<usize>() as u64;
    add(row_bytes, scale(tensors, add(row_bytes, shape_bytes)?)?)
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
