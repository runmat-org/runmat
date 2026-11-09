//! Physical transfer powers, separate from the absolute-temperature algebraic equation.
use runmat_analysis_core::{AnalysisModel, BoundaryOperator};

use super::{
    apply_matrix, sample_ambient_temperature, ThermalAssemblyError, ThermalOperatorAssembly,
};
use crate::{
    contracts::{ThermalEnergyBalance, ThermalPowerScale, ThermalPrescribedHeatFlow},
    excitation::ExcitationSamplePoint,
};

impl ThermalOperatorAssembly {
    /// The immutable model and exact sample (including signal approach) must match assembly.
    /// Re-sampling avoids retaining an input-sized inventory of boundary quadrature descriptors.
    pub(crate) fn energy_balance(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        previous_temperature_k: &[f64],
        temperature_k: &[f64],
        time_step_s: f64,
        reference_temperature_k: f64,
    ) -> Result<ThermalEnergyBalance, ThermalAssemblyError> {
        let rate = previous_temperature_k
            .iter()
            .zip(temperature_k)
            .map(|(previous, current)| (current - previous) / time_step_s)
            .collect::<Vec<_>>();
        let storage_rate = apply_matrix(&self.capacity, &rate);
        let mut storage = PowerSum::default();
        let mut applied = PowerSum::default();
        let mut storage_absolute = PowerSum::default();
        let mut applied_absolute = PowerSum::default();
        let mut nodal_reaction = vec![0.0; storage_rate.len()];
        // Prescribed nodes carry the external reactions. Tie, periodic and jump relations add
        // equal and opposite internal reactions at their secondary and primary nodes (the
        // interpolation weights sum to one), so including those nodes leaves the total external
        // reaction unchanged while every nodal residual stays accounted for.
        let mut reacting = self
            .prescribed_temperature_k
            .canonical_values()
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        for relation in &self.relations {
            reacting[relation.secondary] = true;
            for &(node, _) in &relation.primaries {
                reacting[node] = true;
            }
        }
        for (node, &power) in storage_rate.iter().enumerate() {
            storage.add(power);
            storage_absolute.add(power.abs());
            applied.add(self.applied_source.canonical_values()[node]);
            applied_absolute.add(self.applied_source.canonical_values()[node].abs());
            if reacting[node] {
                nodal_reaction[node] = self.applied_source.canonical_values()[node] - power;
            }
        }

        // Internal conduction is equal/opposite exchange, never global external boundary loss.
        // Only pairs spanning the prescribed/free partition contribute to the total reaction.
        // Prescribed/prescribed pairs cancel globally but still contribute to individual
        // reactions and hence the physical through-flow scale, not an absolute-T floor.
        for (element_index, element) in self.projection.elements.iter().enumerate() {
            if element.node_indices.iter().all(|&node| !reacting[node]) {
                continue;
            }
            self.projection
                .visit_element_quadrature(element, |point| {
                    for (i, &left) in element.node_indices.iter().enumerate() {
                        for (j, &right) in element.node_indices.iter().enumerate().skip(i + 1) {
                            let left_prescribed = reacting[left];
                            let right_prescribed = reacting[right];
                            if !left_prescribed && !right_prescribed {
                                continue;
                            }
                            let gradient_product = point.physical_shape_gradients()[i]
                                .iter()
                                .zip(point.physical_shape_gradients()[j])
                                .map(|(a, b)| a * b)
                                .sum::<f64>();
                            let off_diagonal = self
                                .element_conductivity_w_per_mk
                                .canonical_values()[element_index]
                                * gradient_product
                                * point.volume_weight_m3;
                            let left_reaction =
                                off_diagonal * (temperature_k[left] - temperature_k[right]);
                            if left_prescribed {
                                nodal_reaction[left] += left_reaction;
                            }
                            if right_prescribed {
                                nodal_reaction[right] -= left_reaction;
                            }
                        }
                    }
                })
                .map_err(ThermalAssemblyError::from)?;
        }

        let mut boundary_loss = PowerSum::default();
        let mut boundary_absolute = PowerSum::default();
        for boundary in &model.boundaries {
            let BoundaryOperator::ThermalConvection {
                ambient_temperature,
                coefficient,
            } = &boundary.operator
            else {
                continue;
            };
            let ambient = sample_ambient_temperature(
                model,
                &boundary.boundary_id,
                ambient_temperature,
                sample,
            )?;
            for face in self.projection.boundary_faces_for_target(&boundary.target) {
                self.projection
                    .visit_boundary_quadrature(face, |point| {
                        let temperature_difference = face
                            .node_indices
                            .iter()
                            .zip(point.shape_values())
                            .map(|(&node, &shape)| {
                                shape * (temperature_k[node] - (ambient - reference_temperature_k))
                            })
                            .sum::<f64>();
                        let power = coefficient.canonical_value()
                            * temperature_difference
                            * point.area_weight_m2;
                        boundary_loss.add(power);
                        boundary_absolute.add(power.abs());
                        for (&node, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                            if reacting[node] {
                                nodal_reaction[node] -= shape * power;
                            }
                        }
                    })
                    .map_err(ThermalAssemblyError::from)?;
            }
        }
        if model
            .boundaries
            .iter()
            .any(|boundary| matches!(boundary.operator, BoundaryOperator::ThermalRadiation { .. }))
        {
            self.visit_radiation_scalar_reference(
                model,
                sample,
                reference_temperature_k,
                temperature_k,
                |face, form| {
                    boundary_loss.add(form.outward_power_w);
                    boundary_absolute.add(form.absolute_transfer_w);
                    for (local, &node) in face.node_indices.iter().enumerate() {
                        if reacting[node] {
                            nodal_reaction[node] -= form.outward_nodal_power_w[local];
                        }
                    }
                },
            )?;
        }
        // Both claims describe the same recovered nodal transfers. Summing the
        // unrounded quadrature terms separately can exceed the nodal L1 scale
        // by rounding alone when every reaction has the same sign.
        // A positive reaction is heat leaving the body. Gross flows count prescribed-temperature
        // nodes only: relation nodes exchange heat inside the model.
        let mut reaction = PowerSum::default();
        let mut reaction_absolute = PowerSum::default();
        let mut entering = PowerSum::default();
        let mut leaving = PowerSum::default();
        let prescribed = self.prescribed_temperature_k.canonical_values();
        for (node, value) in nodal_reaction.into_iter().enumerate() {
            reaction.add(value);
            reaction_absolute.add(value.abs());
            if prescribed[node].is_some() {
                if value < 0.0 {
                    entering.add(-value);
                } else {
                    leaving.add(value);
                }
            }
        }
        Ok(ThermalEnergyBalance::from_power_transfers(
            runmat_types::physical::Power::new(storage.total()),
            runmat_types::physical::Power::new(boundary_loss.total()),
            runmat_types::physical::Power::new(applied.total()),
            runmat_types::physical::Power::new(reaction.total()),
            ThermalPowerScale {
                storage_w: runmat_types::physical::Power::new(storage_absolute.total()),
                boundary_w: runmat_types::physical::Power::new(boundary_absolute.total()),
                applied_w: runmat_types::physical::Power::new(applied_absolute.total()),
                prescribed_w: runmat_types::physical::Power::new(reaction_absolute.total()),
            },
            ThermalPrescribedHeatFlow {
                entering_w: runmat_types::physical::Power::new(entering.total()),
                leaving_w: runmat_types::physical::Power::new(leaving.total()),
            },
        ))
    }
}

/// Neumaier accumulation reduces summation roundoff without discarding small powers.
#[derive(Default)]
struct PowerSum {
    sum: f64,
    correction: f64,
}

impl PowerSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}
