use super::{apply_matrix, ThermalAssemblyError, ThermalOperatorAssembly};

use crate::contracts::ThermalEnergyBalance;

impl ThermalOperatorAssembly {
    pub(crate) fn recover_element_temperature_gradients(
        &self,
        temperature_k: &[f64],
    ) -> Result<Vec<[f64; 3]>, ThermalAssemblyError> {
        self.projection
            .elements
            .iter()
            .map(|element| {
                let mut integrated = [0.0; 3];
                let mut volume = 0.0;
                self.projection
                    .visit_element_quadrature(element, |point| {
                        for (&node, gradient) in element
                            .node_indices
                            .iter()
                            .zip(point.physical_shape_gradients())
                        {
                            for axis in 0..3 {
                                integrated[axis] +=
                                    temperature_k[node] * gradient[axis] * point.volume_weight_m3;
                            }
                        }
                        volume += point.volume_weight_m3;
                    })
                    .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
                Ok(integrated.map(|component| component / volume))
            })
            .collect()
    }

    pub(crate) fn recover_element_heat_flux(
        &self,
        temperature_k: &[f64],
    ) -> Result<Vec<[f64; 3]>, ThermalAssemblyError> {
        Ok(self
            .recover_element_temperature_gradients(temperature_k)?
            .into_iter()
            .zip(&self.element_conductivity_w_per_mk)
            .map(|(gradient, conductivity)| gradient.map(|value| -conductivity * value))
            .collect())
    }

    pub(crate) fn recover_element_storage_rate(
        &self,
        previous_temperature_k: &[f64],
        temperature_k: &[f64],
        time_step_s: f64,
    ) -> Result<Vec<f64>, ThermalAssemblyError> {
        self.projection
            .elements
            .iter()
            .enumerate()
            .map(|(element_index, element)| {
                let mut integrated_rate = 0.0;
                let mut volume = 0.0;
                self.projection
                    .visit_element_quadrature(element, |point| {
                        let temperature_rate = element
                            .node_indices
                            .iter()
                            .zip(point.shape_values())
                            .map(|(&node, &shape)| {
                                shape * (temperature_k[node] - previous_temperature_k[node])
                                    / time_step_s
                            })
                            .sum::<f64>();
                        integrated_rate += self.element_volumetric_capacity_j_per_m3k
                            [element_index]
                            * temperature_rate
                            * point.volume_weight_m3;
                        volume += point.volume_weight_m3;
                    })
                    .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
                Ok(integrated_rate / volume)
            })
            .collect()
    }

    pub(crate) fn energy_balance(
        &self,
        previous_temperature_k: &[f64],
        temperature_k: &[f64],
        time_step_s: f64,
    ) -> ThermalEnergyBalance {
        let rate = previous_temperature_k
            .iter()
            .zip(temperature_k)
            .map(|(previous, current)| (current - previous) / time_step_s)
            .collect::<Vec<_>>();
        let storage_rate = apply_matrix(&self.capacity, &rate);
        let conduction = apply_matrix(&self.conductivity, temperature_k);
        let storage_rate_w = storage_rate.iter().sum::<f64>();
        let boundary_loss_w = conduction.iter().sum::<f64>();
        let applied_power_w = self.source.iter().sum::<f64>();
        let prescribed_boundary_power_w = self
            .prescribed_temperature_k
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_some())
            .map(|(node, _)| -(storage_rate[node] + conduction[node] - self.source[node]))
            .sum::<f64>();
        ThermalEnergyBalance::from_powers(
            storage_rate_w,
            boundary_loss_w,
            applied_power_w,
            prescribed_boundary_power_w,
        )
    }
}
