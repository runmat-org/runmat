//! Native complex SI field copies from the original retained Maxwell result only.
use super::*;
use crate::{
    execution::{
        FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoveryRequirement,
        FieldRecoverySource,
    },
    physics::maxwell::{ComplexPhasor, MaxwellFieldSample},
    results::field::{
        RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition,
        RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
};
use runmat_analysis_core::{AnalysisField, FieldLocation, ObservationQuantity, RecoveryPolicy};

pub const MAXWELL_NATIVE_FIELD_COUNT: usize = 11;

#[derive(Clone, Copy)]
pub(in crate::execution) enum NativeMaxwellValues<'a> {
    Circulation(&'a [ComplexPhasor]),
    Electric(&'a [MaxwellFieldSample]),
    Flux(&'a [MaxwellFieldSample]),
    Intensity(&'a [MaxwellFieldSample]),
    Displacement(&'a [MaxwellFieldSample]),
    Current(&'a [MaxwellFieldSample]),
    /// Cycle-averaged Joule loss density (1/2) Re(E . conj(J)), real.
    Loss(&'a [MaxwellFieldSample]),
    /// Element averages (1/V)∫ of a sampled quantity over each element's Keast15 samples.
    ElementElectric(&'a [MaxwellFieldSample]),
    ElementFlux(&'a [MaxwellFieldSample]),
    ElementCurrent(&'a [MaxwellFieldSample]),
    ElementLoss(&'a [MaxwellFieldSample]),
}

fn sample_loss(sample: &MaxwellFieldSample) -> f64 {
    0.5 * (0..3)
        .map(|axis| {
            let e = sample.electric_field_v_per_m.canonical_value()[axis];
            let j = sample.conduction_current_density_a_per_m2.canonical_value()[axis];
            e.real * j.real + e.imaginary * j.imaginary
        })
        .sum::<f64>()
}

fn element_average<'a>(
    samples: &'a [MaxwellFieldSample],
    value: impl Fn(&MaxwellFieldSample) -> [ComplexPhasor; 3] + 'a,
) -> impl Iterator<Item = f64> + 'a {
    samples.chunks(15).flat_map(move |element| {
        let volume: f64 = element.iter().map(|s| s.volume_weight_m3).sum();
        let mut average = [ComplexPhasor::ZERO; 3];
        for sample in element {
            let v = value(sample);
            for axis in 0..3 {
                average[axis] += v[axis] * (sample.volume_weight_m3 / volume);
            }
        }
        average.into_iter().flat_map(|v| [v.real, v.imaginary])
    })
}
impl<'a> NativeMaxwellValues<'a> {
    pub(in crate::execution) fn values(self) -> Box<dyn Iterator<Item = f64> + 'a> {
        match self {
            Self::Loss(samples) => return Box::new(samples.iter().map(sample_loss)),
            Self::ElementElectric(samples) => {
                return Box::new(element_average(samples, |s| {
                    s.electric_field_v_per_m.canonical_value()
                }))
            }
            Self::ElementFlux(samples) => {
                return Box::new(element_average(samples, |s| {
                    s.flux_density_t.canonical_value()
                }))
            }
            Self::ElementCurrent(samples) => {
                return Box::new(element_average(samples, |s| {
                    s.conduction_current_density_a_per_m2.canonical_value()
                }))
            }
            Self::ElementLoss(samples) => {
                return Box::new(samples.chunks(15).map(|element| {
                    let volume: f64 = element.iter().map(|s| s.volume_weight_m3).sum();
                    element
                        .iter()
                        .map(|s| sample_loss(s) * s.volume_weight_m3)
                        .sum::<f64>()
                        / volume
                }))
            }
            _ => {}
        }
        let entities = match self {
            Self::Circulation(v) => v.len(),
            Self::Electric(v)
            | Self::Flux(v)
            | Self::Intensity(v)
            | Self::Displacement(v)
            | Self::Current(v) => v.len(),
            Self::Loss(_)
            | Self::ElementElectric(_)
            | Self::ElementFlux(_)
            | Self::ElementCurrent(_)
            | Self::ElementLoss(_) => unreachable!("handled above"),
        };
        let components = if matches!(self, Self::Circulation(_)) {
            1
        } else {
            3
        };
        Box::new(
            (0..entities)
                .flat_map(move |i| {
                    let values = match self {
                        Self::Circulation(v) => [v[i], ComplexPhasor::ZERO, ComplexPhasor::ZERO],
                        Self::Electric(v) => v[i].electric_field_v_per_m.canonical_value(),
                        Self::Flux(v) => v[i].flux_density_t.canonical_value(),
                        Self::Intensity(v) => v[i].magnetic_field_a_per_m.canonical_value(),
                        Self::Displacement(v) => {
                            v[i].electric_displacement_c_per_m2.canonical_value()
                        }
                        Self::Current(v) => {
                            v[i].conduction_current_density_a_per_m2.canonical_value()
                        }
                        _ => unreachable!("handled above"),
                    };
                    values.into_iter().take(components)
                })
                .flat_map(|value| [value.real, value.imaginary]),
        )
    }
}
impl Serialize for NativeMaxwellValues<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.values())
    }
}
pub(in crate::execution) struct NativeMaxwellField<'a> {
    pub field_id: &'static str,
    pub values: NativeMaxwellValues<'a>,
    pub shape: Vec<usize>,
}
pub(in crate::execution) fn native_maxwell_fields(
    result: &MaxwellHarmonicSolverResult,
) -> [NativeMaxwellField<'_>; MAXWELL_NATIVE_FIELD_COUNT] {
    let r = result.response();
    let samples = &r.samples;
    [
        NativeMaxwellField {
            field_id: "maxwell.electric_edge_circulation",
            values: NativeMaxwellValues::Circulation(r.edge_circulations_v.canonical_values()),
            shape: vec![r.edge_circulations_v.canonical_values().len(), 2],
        },
        NativeMaxwellField {
            field_id: "maxwell.electric_field",
            values: NativeMaxwellValues::Electric(samples),
            shape: vec![samples.len(), 3, 2],
        },
        NativeMaxwellField {
            field_id: "maxwell.flux_density",
            values: NativeMaxwellValues::Flux(samples),
            shape: vec![samples.len(), 3, 2],
        },
        NativeMaxwellField {
            field_id: "maxwell.magnetic_field",
            values: NativeMaxwellValues::Intensity(samples),
            shape: vec![samples.len(), 3, 2],
        },
        NativeMaxwellField {
            field_id: "maxwell.electric_displacement",
            values: NativeMaxwellValues::Displacement(samples),
            shape: vec![samples.len(), 3, 2],
        },
        NativeMaxwellField {
            field_id: "maxwell.conduction_current_density",
            values: NativeMaxwellValues::Current(samples),
            shape: vec![samples.len(), 3, 2],
        },
        NativeMaxwellField {
            field_id: "maxwell.joule_power_density",
            values: NativeMaxwellValues::Loss(samples),
            shape: vec![samples.len()],
        },
        NativeMaxwellField {
            field_id: "maxwell.element_electric_field",
            values: NativeMaxwellValues::ElementElectric(samples),
            shape: vec![samples.len() / 15, 3, 2],
        },
        NativeMaxwellField {
            field_id: "maxwell.element_flux_density",
            values: NativeMaxwellValues::ElementFlux(samples),
            shape: vec![samples.len() / 15, 3, 2],
        },
        NativeMaxwellField {
            field_id: "maxwell.element_conduction_current_density",
            values: NativeMaxwellValues::ElementCurrent(samples),
            shape: vec![samples.len() / 15, 3, 2],
        },
        NativeMaxwellField {
            field_id: "maxwell.element_joule_power_density",
            values: NativeMaxwellValues::ElementLoss(samples),
            shape: vec![samples.len() / 15],
        },
    ]
}

impl MaxwellHarmonicSolverResult {
    pub fn native_field_request(
        &self,
        workload: &MaxwellHarmonicWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        self.validate_for(workload, mesh)?;
        self.native_field_request_from_result()
    }
    pub fn native_field_request_from_result(
        &self,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        self.validate()?;
        FieldRecoveryRequest::new(
            self.producer_identity(),
            FieldRecoverySource::SolverResult(self.identity()?),
            self.sampling_topology().identity().map_err(invalid)?,
            native_maxwell_field_plan(self)?,
        )
    }
    pub fn recover_native_fields(
        &self,
        workload: &MaxwellHarmonicWorkload,
        mesh: &SolverMeshArtifact,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        self.validate_for(workload, mesh)?;
        self.recover_native_fields_from_result(request)
    }
    /// Requires the original authenticated result. Never evaluates an operator or regenerates a
    /// lost physical field; each scalar is copied in canonical real/imaginary interleaved order.
    pub fn recover_native_fields_from_result(
        &self,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        request.validate()?;
        if request != &self.native_field_request_from_result()? {
            return Err(invalid("Maxwell field request differs from original producer/source/sampling/native inventory"));
        }
        let definitions = maxwell_native_field_definitions(request)?;
        let fields = native_maxwell_fields(self)
            .into_iter()
            .zip(definitions)
            .map(|(field, definition)| {
                let layout = definition.layout();
                RecoveredField::new(
                    definition,
                    RecoveredFieldSample::Final,
                    AnalysisField::host_f64(
                        field.field_id,
                        layout,
                        field.shape,
                        field.values.values().collect(),
                    ),
                )
                .map_err(invalid)
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        request
            .plan
            .validate_definitions(fields.iter().map(|field| field.definition().clone()))?;
        RecoveredFieldArtifact::new(
            request.identity(),
            request.source,
            request.topology.discretization_digest(),
            native_maxwell_entity_counts(self),
            fields,
        )
        .map_err(invalid)
    }
}
pub(in crate::execution) fn native_maxwell_entity_counts(
    result: &MaxwellHarmonicSolverResult,
) -> RecoveredFieldEntityCounts {
    result.sampling_topology().entity_counts()
}
fn native_maxwell_field_plan(
    result: &MaxwellHarmonicSolverResult,
) -> AnalysisExecutionResult<FieldRecoveryPlan> {
    let requirements = maxwell_native_field_definition_inventory()?
        .into_iter()
        .map(|definition| {
            let count = match definition.entity_domain() {
                RecoveredFieldEntityDomain::OrientedEdges => result
                    .response()
                    .edge_circulations_v
                    .canonical_values()
                    .len(),
                RecoveredFieldEntityDomain::VolumeIntegrationPoints => {
                    result.response().samples.len()
                }
                RecoveredFieldEntityDomain::VolumeElements => result.response().samples.len() / 15,
                _ => return Err(invalid("non-native Maxwell entity domain")),
            };
            let count = u64::try_from(count).map_err(invalid)?;
            let components = definition
                .value_shape()
                .component_count()
                .ok_or_else(|| invalid("Maxwell component count"))?
                as u64;
            Ok(FieldRecoveryRequirement {
                support: Default::default(),
                quantity: definition.quantity(),
                source_location: definition.location(),
                output_location: definition.location(),
                recovery: RecoveryPolicy::Native,
                value_shape: definition.value_shape().clone(),
                entity_domain: definition.entity_domain(),
                cost: FieldRecoveryCost {
                    source_entity_count: count,
                    output_entity_count: count,
                    topology_incidence_count: 0,
                    arithmetic_operation_count: count
                        .checked_mul(components)
                        .and_then(|v| v.checked_mul(2))
                        .ok_or_else(|| invalid("Maxwell complex scalar count overflow"))?,
                },
            })
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let plan = FieldRecoveryPlan { requirements };
    plan.validate()?;
    Ok(plan)
}

/// Physics-native definitions only; no native arrays, solver invocation, or source substitution.
pub fn maxwell_native_field_definition_inventory(
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    [
        (
            ObservationQuantity::ElectricEdgeCirculation,
            "maxwell.electric_edge_circulation",
            FieldLocation::Edge,
            RecoveredFieldEntityDomain::OrientedEdges,
        ),
        (
            ObservationQuantity::ElectricField,
            "maxwell.electric_field",
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ),
        (
            ObservationQuantity::MagneticFluxDensity,
            "maxwell.flux_density",
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ),
        (
            ObservationQuantity::MagneticFieldIntensity,
            "maxwell.magnetic_field",
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ),
        (
            ObservationQuantity::ElectricDisplacement,
            "maxwell.electric_displacement",
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ),
        (
            ObservationQuantity::ElectricCurrentDensity,
            "maxwell.conduction_current_density",
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ),
        (
            ObservationQuantity::JoulePowerDensity,
            "maxwell.joule_power_density",
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
        ),
        (
            ObservationQuantity::ElectricField,
            "maxwell.element_electric_field",
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
        ),
        (
            ObservationQuantity::MagneticFluxDensity,
            "maxwell.element_flux_density",
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
        ),
        (
            ObservationQuantity::ElectricCurrentDensity,
            "maxwell.element_conduction_current_density",
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
        ),
        (
            ObservationQuantity::JoulePowerDensity,
            "maxwell.element_joule_power_density",
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
        ),
    ]
    .into_iter()
    .map(|(quantity, id, location, domain)| {
        RecoveredFieldDefinition::new(
            id,
            quantity,
            location,
            domain,
            quantity.canonical_value_shape(),
            if quantity == ObservationQuantity::JoulePowerDensity {
                RecoveredFieldNumericRepresentation::Real
            } else {
                RecoveredFieldNumericRepresentation::ComplexInterleaved
            },
        )
        .map_err(invalid)
    })
    .collect()
}

/// Strict metadata inventory validation. Authenticating an actual source additionally requires
/// request equality against the original retained result and original sampled topology.
pub fn maxwell_native_field_definitions(
    request: &FieldRecoveryRequest,
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    request.validate()?;
    if !matches!(
        request.topology,
        crate::execution::FieldTopologyIdentity::SampledVolumeMesh { .. }
    ) {
        return Err(invalid(
            "Maxwell native fields require sampled-volume topology",
        ));
    }
    let definitions = maxwell_native_field_definition_inventory()?;
    if request.plan.requirements.len() != MAXWELL_NATIVE_FIELD_COUNT {
        return Err(invalid(
            "Maxwell native inventory must retain all original sampled and element fields",
        ));
    }
    let samples = request.plan.requirements[1].cost.output_entity_count;
    let edge_upper = (samples / 15)
        .checked_mul(6)
        .ok_or_else(|| invalid("Maxwell edge count overflow"))?;
    if samples == 0
        || !samples.is_multiple_of(15)
        || request.plan.requirements[0].cost.output_entity_count == 0
        || request.plan.requirements[0].cost.output_entity_count > edge_upper
        || request.plan.requirements[1..].iter().any(|r| {
            r.cost.output_entity_count
                != if r.entity_domain == RecoveredFieldEntityDomain::VolumeElements {
                    samples / 15
                } else {
                    samples
                }
        })
    {
        return Err(invalid(
            "Maxwell native inventory requires complete Keast15 samples and original edges",
        ));
    }
    for (r, d) in request.plan.requirements.iter().zip(&definitions) {
        let components = d
            .value_shape()
            .component_count()
            .ok_or_else(|| invalid("Maxwell component count"))? as u64;
        let scalars = r
            .cost
            .output_entity_count
            .checked_mul(components)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("Maxwell complex scalar count overflow"))?;
        if r.quantity != d.quantity()
            || r.source_location != d.location()
            || r.output_location != d.location()
            || r.entity_domain != d.entity_domain()
            || r.recovery != RecoveryPolicy::Native
            || &r.value_shape != d.value_shape()
            || r.cost.source_entity_count != r.cost.output_entity_count
            || r.cost.topology_incidence_count != 0
            || r.cost.arithmetic_operation_count != scalars
        {
            return Err(invalid(
                "Maxwell native field inventory differs from exact complex native shape",
            ));
        }
    }
    Ok(definitions)
}

#[cfg(test)]
mod tests;
