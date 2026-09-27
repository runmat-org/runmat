//! Physical transfer powers, separate from the absolute-temperature algebraic equation.
use runmat_analysis_core::{AnalysisModel, BoundaryOperator};

use super::{
    apply_matrix, sample_ambient_temperature, ThermalAssemblyError, ThermalOperatorAssembly,
};
use crate::{
    contracts::{ThermalEnergyBalance, ThermalPowerScale},
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
    ) -> Result<ThermalEnergyBalance, ThermalAssemblyError> {
        let rate = previous_temperature_k
            .iter()
            .zip(temperature_k)
            .map(|(previous, current)| (current - previous) / time_step_s)
            .collect::<Vec<_>>();
        let storage_rate = apply_matrix(&self.capacity, &rate);
        let mut storage = PowerSum::default();
        let mut applied = PowerSum::default();
        let mut reaction = PowerSum::default();
        let mut storage_absolute = PowerSum::default();
        let mut applied_absolute = PowerSum::default();
        let mut nodal_reaction = vec![0.0; storage_rate.len()];
        for (node, &power) in storage_rate.iter().enumerate() {
            storage.add(power);
            storage_absolute.add(power.abs());
            applied.add(self.applied_source[node]);
            applied_absolute.add(self.applied_source[node].abs());
            if self.prescribed_temperature_k[node].is_some() {
                reaction.add(-power);
                reaction.add(self.applied_source[node]);
                nodal_reaction[node] = self.applied_source[node] - power;
            }
        }

        // Internal conduction is equal/opposite exchange, never global external boundary loss.
        // Only pairs spanning the prescribed/free partition contribute to the total reaction.
        // Prescribed/prescribed pairs cancel globally but still contribute to individual
        // reactions and hence the physical through-flow scale, not an absolute-T floor.
        for (element_index, element) in self.projection.elements.iter().enumerate() {
            let first_prescribed = self.prescribed_temperature_k[element.node_indices[0]].is_some();
            if element
                .node_indices
                .iter()
                .all(|&node| self.prescribed_temperature_k[node].is_some() == first_prescribed)
            {
                continue;
            }
            self.projection
                .visit_element_quadrature(element, |point| {
                    for (i, &left) in element.node_indices.iter().enumerate() {
                        for (j, &right) in element.node_indices.iter().enumerate().skip(i + 1) {
                            let left_prescribed = self.prescribed_temperature_k[left].is_some();
                            let right_prescribed = self.prescribed_temperature_k[right].is_some();
                            if !left_prescribed && !right_prescribed {
                                continue;
                            }
                            let gradient_product = point.physical_shape_gradients()[i]
                                .iter()
                                .zip(point.physical_shape_gradients()[j])
                                .map(|(a, b)| a * b)
                                .sum::<f64>();
                            let off_diagonal = self.element_conductivity_w_per_mk[element_index]
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
                            if left_prescribed != right_prescribed {
                                reaction.add(if left_prescribed {
                                    left_reaction
                                } else {
                                    -left_reaction
                                });
                            }
                        }
                    }
                })
                .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
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
                            .map(|(&node, &shape)| shape * (temperature_k[node] - ambient))
                            .sum::<f64>();
                        let power = coefficient.watts_per_square_meter_kelvin
                            * temperature_difference
                            * point.area_weight_m2;
                        boundary_loss.add(power);
                        boundary_absolute.add(power.abs());
                        for (&node, &shape) in face.node_indices.iter().zip(point.shape_values()) {
                            if self.prescribed_temperature_k[node].is_some() {
                                reaction.add(-shape * power);
                                nodal_reaction[node] -= shape * power;
                            }
                        }
                    })
                    .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
            }
        }
        let mut reaction_absolute = PowerSum::default();
        for value in nodal_reaction {
            reaction_absolute.add(value.abs());
        }
        Ok(ThermalEnergyBalance::from_power_transfers(
            storage.total(),
            boundary_loss.total(),
            applied.total(),
            reaction.total(),
            ThermalPowerScale {
                storage_w: storage_absolute.total(),
                boundary_w: boundary_absolute.total(),
                applied_w: applied_absolute.total(),
                prescribed_w: reaction_absolute.total(),
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
