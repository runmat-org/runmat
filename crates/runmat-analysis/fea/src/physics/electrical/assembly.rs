use super::{ElectricalStaticError, VACUUM_PERMITTIVITY_F_PER_M};
use crate::{
    algebra::{
        element_assembly::{
            assemble_elements, ElementAssemblyError, ElementOutputShape, ElementOutputs,
            ElementView,
        },
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    properties::CompiledPropertyMap,
    topology::{projection_element_map, AnalysisProjection},
};
use runmat_analysis_core::StaticElectricalEquation;

pub(super) struct ElectricalAssembly {
    pub projection: AnalysisProjection,
    pub stiffness: RealSparseMatrix,
    pub local_energy: Option<crate::algebra::anchored_energy::AnchoredEnergySource>,
    pub coefficients:
        runmat_types::physical::Quantity<runmat_types::physical::DynamicDimension, Vec<f64>>,
    pub prescribed: Vec<Option<f64>>,
    pub terminal_nodes: Vec<(String, Vec<usize>)>,
    pub charge_sources: Option<super::ElectricalChargeSources>,
    /// Periodic node relations (equal potentials across paired faces).
    pub relations: Vec<crate::physics::scalar_constraints::ScalarNodeRelation>,
}

impl ElectricalAssembly {
    pub(super) fn assemble(
        projection: AnalysisProjection,
        properties: &CompiledPropertyMap,
        equation: StaticElectricalEquation,
        conductivity_override: Option<&[f64]>,
    ) -> Result<Self, ElectricalStaticError> {
        if conductivity_override.is_some_and(|values| values.len() != projection.elements.len()) {
            return Err(ElectricalStaticError::Material(
                "one conductivity per element is required".into(),
            ));
        }
        let count = projection.nodes.len();
        let contribution = ContributionProvenance {
            domain: ContributionDomain::Electromagnetic,
            contribution_id: ContributionId(
                match equation {
                    StaticElectricalEquation::Electrostatics => "dielectric_potential_gradient",
                    StaticElectricalEquation::SteadyConduction => "conductive_potential_gradient",
                }
                .into(),
            ),
        };
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
        for (element_index, element) in projection.elements.iter().enumerate() {
            let material = &properties
                .solid_material(element.stable_identity)
                .ok_or_else(|| {
                    ElectricalStaticError::Material("unassigned electrical element".into())
                })?
                .material;
            let electrical = material.electrical.as_ref().ok_or_else(|| {
                ElectricalStaticError::Material("missing electrical constitutive model".into())
            })?;
            // A frequency response table describes the material in harmonic analyses; a
            // static (DC) solve uses the reference values.
            // A standalone static solve has no temperature field: conductivity is the
            // material's value at its reference temperature. Coupled electrothermal solves
            // pass the conductivity evaluated at each element's temperature.
            let coefficient = match equation {
                StaticElectricalEquation::Electrostatics => {
                    VACUUM_PERMITTIVITY_F_PER_M.canonical_value()
                        * electrical.relative_permittivity.canonical_value()
                }
                StaticElectricalEquation::SteadyConduction => conductivity_override.map_or(
                    electrical.conductivity_s_per_m.canonical_value(),
                    |values| values[element_index],
                ),
            };
            if !coefficient.is_finite() || coefficient <= 0.0 {
                return Err(ElectricalStaticError::Material(
                    "epsilon or sigma must be positive finite".into(),
                ));
            }
            coefficients.push(coefficient);
        }
        // Element gradient kernels run on the admitted CPU pool; the ordered visitor retains each
        // original local matrix, and the scatter is thread-count independent.
        let map = projection_element_map(&projection)
            .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
        let plan = crate::planning::resource::LocalExecutionPlan::serial();
        let pattern = map
            .symbolic_pattern(plan)
            .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
        let assembled = assemble_elements(
            &map,
            &pattern,
            ElementOutputShape {
                matrices: 1,
                ..Default::default()
            },
            plan,
            |index, _, out: &mut ElementOutputs<'_, f64>| {
                let element = &projection.elements[index];
                let coefficient = coefficients[index];
                let width = element.node_indices.len();
                // Fifteen quadrature points; each node pair has a three-term
                // gradient dot (5), coefficient/measure scales (2), and one
                // accumulation. Reserve the whole element before matrix writes.
                let pairs = u64::try_from(width)
                    .ok()
                    .and_then(|width| width.checked_mul(width))
                    .ok_or(ElectricalStaticError::Accounting(
                        runmat_simulation_usage::LedgerError::CountOverflow,
                    ))?;
                let per_point = pairs
                    .checked_mul(8)
                    .ok_or(ElectricalStaticError::Accounting(
                        runmat_simulation_usage::LedgerError::CountOverflow,
                    ))?;
                let maximum =
                    per_point
                        .checked_mul(15)
                        .ok_or(ElectricalStaticError::Accounting(
                            runmat_simulation_usage::LedgerError::CountOverflow,
                        ))?;
                let usage = runmat_simulation_usage::UsageContext::capture();
                let mut batch = usage
                    .as_ref()
                    .map(|usage| {
                        usage.ledger().reserve(
                            runmat_simulation_usage::UsageCategory::Assembly,
                            runmat_simulation_usage::OperationCount::from_operations(maximum),
                        )
                    })
                    .transpose()
                    .map_err(ElectricalStaticError::Accounting)?;
                let mut visited = 0_u64;
                let local = out.matrix(0);
                let quadrature = projection
                    .visit_element_quadrature(element, |point| {
                        let gradients = point.physical_shape_gradients();
                        for i in 0..width {
                            for j in 0..width {
                                let dot = gradients[i]
                                    .iter()
                                    .zip(gradients[j])
                                    .map(|(a, b)| a * b)
                                    .sum::<f64>();
                                local[i * width + j] += coefficient * dot * point.volume_weight_m3;
                            }
                        }
                        visited += 1;
                    })
                    .map_err(|error| match error {
                        crate::topology::AnalysisProjectionError::Accounting(error) => {
                            ElectricalStaticError::Accounting(error)
                        }
                        other => ElectricalStaticError::Topology(other.to_string()),
                    });
                if let Some(batch) = &mut batch {
                    batch
                        .complete(runmat_simulation_usage::OperationCount::from_operations(
                            visited * per_point,
                        ))
                        .map_err(ElectricalStaticError::Accounting)?;
                }
                quadrature
            },
            |view: &ElementView<'_, f64>| {
                let element = &projection.elements[view.element];
                local_energy
                    .capture_next(&element.node_indices)
                    .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?
                    .copy_from_slice(view.matrix(0));
                Ok(())
            },
        )
        .map_err(|error| match error {
            ElementAssemblyError::Element { error, .. } => error,
            ElementAssemblyError::Admission(error) => error,
            ElementAssemblyError::Accounting(error) => ElectricalStaticError::Accounting(error),
            other => ElectricalStaticError::Operator(other.to_string()),
        })?;
        let mut stiffness = RealSparseAssembler::from_planned_values(
            pattern,
            assembled.matrices.into_iter().next().unwrap_or_default(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?;
        stiffness.add_contribution(contribution);
        Ok(Self {
            projection,
            stiffness: stiffness
                .finish()
                .map_err(|error| ElectricalStaticError::Operator(error.to_string()))?,
            coefficients: runmat_types::physical::Quantity::try_from_descriptor(
                runmat_types::physical::DynamicDimension::quantity_type(
                    equation.constitutive_coefficient_kind(),
                )
                .with_semantic_kind(Some(equation.constitutive_coefficient_kind()))
                .expect("electrical coefficient"),
                coefficients,
            )
            .map_err(|error| ElectricalStaticError::Material(error.to_string()))?,
            local_energy: Some(local_energy),
            prescribed: vec![None; count],
            terminal_nodes: Vec::new(),
            relations: Vec::new(),
            charge_sources: None,
        })
    }
}
