//! Copy native retained SI arrays; no operator evaluation or weak-flux density relabeling.
use super::*;
use crate::physics::electrical::ElectricalConstitutiveResponse;
use crate::{
    execution::{
        FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoveryRequirement,
        FieldRecoverySource, FieldTopologyIdentity,
    },
    results::field::{
        RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition,
        RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
};
use runmat_analysis_core::{AnalysisField, FieldLocation, ObservationQuantity, RecoveryPolicy};

pub(in crate::execution) enum NativeElectricalValues<'a> {
    Scalar(&'a [f64]),
    Vector(&'a [[f64; 3]]),
}
impl Serialize for NativeElectricalValues<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Scalar(values) => serializer.collect_seq(values.iter()),
            Self::Vector(values) => serializer.collect_seq(values.iter().flatten()),
        }
    }
}
pub(in crate::execution) struct NativeElectricalField<'a> {
    pub field_id: &'static str,
    pub values: NativeElectricalValues<'a>,
    pub shape: Vec<usize>,
}
pub(in crate::execution) fn native_electrical_fields(
    result: &ElectricalStaticSolverResult,
) -> [NativeElectricalField<'_>; 4] {
    let r = result.response();
    let nodes = r.ordered_node_ids.len();
    let elements = r.ordered_element_ids.len();
    let (flux_id, flux, density_id, density) = match &r.constitutive {
        ElectricalConstitutiveResponse::Electrostatics {
            displacement_c_per_m2,
            energy_density_j_per_m3,
            ..
        } => (
            "electrical.electric_displacement",
            displacement_c_per_m2.canonical_values().as_slice(),
            "electrical.electric_energy_density",
            energy_density_j_per_m3.canonical_values().as_slice(),
        ),
        ElectricalConstitutiveResponse::SteadyConduction {
            current_density_a_per_m2,
            joule_density_w_per_m3,
            ..
        } => (
            "electrical.current_density",
            current_density_a_per_m2.canonical_values().as_slice(),
            "electrical.joule_power_density",
            joule_density_w_per_m3.canonical_values().as_slice(),
        ),
    };
    [
        NativeElectricalField {
            field_id: "electrical.potential",
            values: NativeElectricalValues::Scalar(r.potential_v.canonical_values()),
            shape: vec![nodes],
        },
        NativeElectricalField {
            field_id: "electrical.electric_field",
            values: NativeElectricalValues::Vector(
                r.element_electric_field_v_per_m.canonical_values(),
            ),
            shape: vec![elements, 3],
        },
        NativeElectricalField {
            field_id: flux_id,
            values: NativeElectricalValues::Vector(flux),
            shape: vec![elements, 3],
        },
        NativeElectricalField {
            field_id: density_id,
            values: NativeElectricalValues::Scalar(density),
            shape: vec![elements],
        },
    ]
}
impl ElectricalStaticSolverResult {
    pub fn native_field_request(
        &self,
        workload: &ElectricalStaticWorkload,
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
            self.producer(),
            FieldRecoverySource::SolverResult(self.source_result_identity()?),
            FieldTopologyIdentity::VolumeMesh {
                mesh_digest: self.mesh_digest(),
            },
            native_electrical_field_plan(
                self.response().equation,
                self.response().ordered_node_ids.len(),
                self.response().ordered_element_ids.len(),
            )?,
        )
    }
    pub fn recover_native_fields(
        &self,
        workload: &ElectricalStaticWorkload,
        mesh: &SolverMeshArtifact,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        self.validate_for(workload, mesh)?;
        self.recover_native_fields_from_result(request)
    }
    pub fn recover_native_fields_from_result(
        &self,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        request.validate()?;
        if request != &self.native_field_request_from_result()? {
            return Err(invalid("electrical field request differs from exact producer/result/topology/native inventory"));
        }
        let fields = native_electrical_fields(self)
            .into_iter()
            .zip(&request.plan.requirements)
            .map(|(field, requirement)| {
                let values = match field.values {
                    NativeElectricalValues::Scalar(values) => values.to_vec(),
                    NativeElectricalValues::Vector(values) => {
                        values.iter().flatten().copied().collect()
                    }
                };
                let definition = native_electrical_field_definition(field.field_id, requirement)?;
                let layout = definition.layout();
                RecoveredField::new(
                    definition,
                    RecoveredFieldSample::Final,
                    AnalysisField::host_f64(field.field_id, layout, field.shape, values),
                )
                .map_err(invalid)
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        request
            .plan
            .validate_definitions(fields.iter().map(|f| f.definition().clone()))?;
        RecoveredFieldArtifact::new(
            request.identity(),
            request.source,
            request.topology.discretization_digest(),
            native_electrical_entity_counts(self),
            fields,
        )
        .map_err(invalid)
    }
}
pub(in crate::execution) fn native_electrical_field_plan(
    equation: StaticElectricalEquation,
    nodes: usize,
    elements: usize,
) -> AnalysisExecutionResult<FieldRecoveryPlan> {
    let (flux, density) = match equation {
        StaticElectricalEquation::Electrostatics => (
            ObservationQuantity::ElectricDisplacement,
            ObservationQuantity::ElectricEnergyDensity,
        ),
        StaticElectricalEquation::SteadyConduction => (
            ObservationQuantity::ElectricCurrentDensity,
            ObservationQuantity::JoulePowerDensity,
        ),
    };
    let specs = [
        (
            ObservationQuantity::ElectricPotential,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            nodes,
        ),
        (
            ObservationQuantity::ElectricField,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            elements,
        ),
        (
            flux,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            elements,
        ),
        (
            density,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            elements,
        ),
    ];
    let requirements = specs
        .into_iter()
        .map(|(quantity, location, entity_domain, count)| {
            let count = u64::try_from(count).map_err(invalid)?;
            Ok(FieldRecoveryRequirement {
                support: Default::default(),
                quantity,
                source_location: location,
                output_location: location,
                recovery: RecoveryPolicy::Native,
                value_shape: quantity.canonical_value_shape(),
                entity_domain,
                cost: FieldRecoveryCost {
                    source_entity_count: count,
                    output_entity_count: count,
                    topology_incidence_count: 0,
                    arithmetic_operation_count: count,
                },
            })
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let plan = FieldRecoveryPlan { requirements };
    plan.validate()?;
    Ok(plan)
}
pub(in crate::execution) fn native_electrical_entity_counts(
    result: &ElectricalStaticSolverResult,
) -> RecoveredFieldEntityCounts {
    RecoveredFieldEntityCounts {
        oriented_faces: 0,
        nodes: result.response().ordered_node_ids.len(),
        volume_elements: result.response().ordered_element_ids.len(),
        volume_integration_points: 0,
        oriented_edges: 0,
        beam_elements: 0,
        shell_elements: 0,
        connectors: 0,
        constraint_groups: 0,
    }
}
pub(in crate::execution) fn native_electrical_field_definition(
    field_id: &str,
    requirement: &FieldRecoveryRequirement,
) -> AnalysisExecutionResult<RecoveredFieldDefinition> {
    RecoveredFieldDefinition::new(
        field_id,
        requirement.quantity,
        requirement.output_location,
        requirement.entity_domain,
        requirement.value_shape.clone(),
        RecoveredFieldNumericRepresentation::Real,
    )
    .map_err(invalid)
}

/// Exact native definition inventory of an electrical recovery request, without arrays.
pub fn electrical_native_field_definitions(
    request: &FieldRecoveryRequest,
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    request.validate()?;
    let FieldTopologyIdentity::VolumeMesh { .. } = request.topology else {
        return Err(invalid(
            "electrical native fields live on the canonical volume mesh",
        ));
    };
    let [potential, field, flux, density] = request.plan.requirements.as_slice() else {
        return Err(invalid("electrical native inventory has four fields"));
    };
    let equation = match (flux.quantity, density.quantity) {
        (ObservationQuantity::ElectricCurrentDensity, ObservationQuantity::JoulePowerDensity) => {
            StaticElectricalEquation::SteadyConduction
        }
        (ObservationQuantity::ElectricDisplacement, ObservationQuantity::ElectricEnergyDensity) => {
            StaticElectricalEquation::Electrostatics
        }
        _ => return Err(invalid("electrical native inventory quantities differ")),
    };
    let nodes = usize::try_from(potential.cost.output_entity_count).map_err(invalid)?;
    let elements = usize::try_from(field.cost.output_entity_count).map_err(invalid)?;
    if native_electrical_field_plan(equation, nodes, elements)? != request.plan {
        return Err(invalid(
            "electrical native inventory differs from canonical shape",
        ));
    }
    let ids = match equation {
        StaticElectricalEquation::SteadyConduction => [
            "electrical.potential",
            "electrical.electric_field",
            "electrical.current_density",
            "electrical.joule_power_density",
        ],
        StaticElectricalEquation::Electrostatics => [
            "electrical.potential",
            "electrical.electric_field",
            "electrical.electric_displacement",
            "electrical.electric_energy_density",
        ],
    };
    ids.into_iter()
        .zip(&request.plan.requirements)
        .map(|(id, requirement)| native_electrical_field_definition(id, requirement))
        .collect()
}
