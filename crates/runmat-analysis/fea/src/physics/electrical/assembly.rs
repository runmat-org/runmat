use super::{ElectricalStaticError, VACUUM_PERMITTIVITY_F_PER_M};
use crate::{
    algebra::{
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    properties::CompiledPropertyMap,
    topology::AnalysisProjection,
};
use runmat_analysis_core::StaticElectricalEquation;

pub(super) struct ElectricalAssembly {
    pub projection: AnalysisProjection,
    pub stiffness: RealSparseMatrix,
    pub local_energy: Option<crate::algebra::anchored_energy::AnchoredEnergySource>,
    pub coefficients: Vec<f64>,
    pub prescribed: Vec<Option<f64>>,
    pub terminal_nodes: Vec<(String, Vec<usize>)>,
    pub charge_sources: Option<super::ElectricalChargeSources>,
}

impl ElectricalAssembly {
    pub(super) fn assemble(
        projection: AnalysisProjection,
        properties: &CompiledPropertyMap,
        equation: StaticElectricalEquation,
    ) -> Result<Self, ElectricalStaticError> {
        let count = projection.nodes.len();
        let mut stiffness = RealSparseAssembler::square(
            count,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        stiffness.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Electromagnetic,
            contribution_id: ContributionId(
                match equation {
                    StaticElectricalEquation::Electrostatics => "dielectric_potential_gradient",
                    StaticElectricalEquation::SteadyConduction => "conductive_potential_gradient",
                }
                .into(),
            ),
        });
        let widths = projection
            .elements
            .iter()
            .map(|element| element.node_indices.len());
        let source_bytes =
            crate::algebra::anchored_energy::AnchoredEnergySource::resident_bytes_for_widths(
                widths.clone(),
            )
            .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
        let mut local_energy =
            crate::algebra::anchored_energy::AnchoredEnergySource::new(widths, source_bytes)
                .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
        let mut coefficients = Vec::with_capacity(projection.elements.len());
        for element in &projection.elements {
            let material = &properties
                .solid_material(element.stable_identity)
                .ok_or_else(|| {
                    ElectricalStaticError::Material("unassigned electrical element".into())
                })?
                .material;
            let electrical = material.electrical.as_ref().ok_or_else(|| {
                ElectricalStaticError::Material("missing electrical constitutive model".into())
            })?;
            if !electrical.conductivity_frequency_response.is_empty()
                || electrical.resistive_heating_coefficient != 0.0
            {
                return Err(ElectricalStaticError::Unsupported(
                    "dispersive or temperature-dependent electrical law".into(),
                ));
            }
            let coefficient = match equation {
                StaticElectricalEquation::Electrostatics => {
                    VACUUM_PERMITTIVITY_F_PER_M * electrical.relative_permittivity
                }
                StaticElectricalEquation::SteadyConduction => electrical.conductivity_s_per_m,
            };
            if !coefficient.is_finite() || coefficient <= 0.0 {
                return Err(ElectricalStaticError::Material(
                    "epsilon or sigma must be positive finite".into(),
                ));
            }
            coefficients.push(coefficient);
            let local_values = local_energy
                .capture_next(&element.node_indices)
                .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
            let width = element.node_indices.len();
            projection
                .visit_element_quadrature(element, |point| {
                    for (i, &row) in element.node_indices.iter().enumerate() {
                        for (j, &column) in element.node_indices.iter().enumerate() {
                            let dot = point.physical_shape_gradients()[i]
                                .iter()
                                .zip(point.physical_shape_gradients()[j])
                                .map(|(a, b)| a * b)
                                .sum::<f64>();
                            let contribution = coefficient * dot * point.volume_weight_m3;
                            stiffness.add(row, column, contribution);
                            local_values[i * width + j] += contribution;
                        }
                    }
                })
                .map_err(|error| ElectricalStaticError::Topology(error.to_string()))?;
        }
        Ok(Self {
            projection,
            stiffness: stiffness
                .finish()
                .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?,
            coefficients,
            local_energy: Some(local_energy),
            prescribed: vec![None; count],
            terminal_nodes: Vec::new(),
            charge_sources: None,
        })
    }
}
