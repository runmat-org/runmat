use super::{ThermalAssemblyError, ThermalOperatorAssembly};

impl ThermalOperatorAssembly {
    pub(crate) fn recover_element_temperature_gradients(
        &self,
        temperature_k: &[f64],
    ) -> Result<Vec<[f64; 3]>, ThermalAssemblyError> {
        let elements = &self.projection.elements;
        crate::algebra::element_assembly::ordered_map_range(elements.len(), |index| {
            let element = &elements[index];
            let mut integrated = [0.0; 3];
            let mut volume = 0.0;
            // Partition of unity makes the reference temperature immaterial analytically.
            // Subtract before multiplication so a constant field has exactly zero gradient.
            let reference = temperature_k[element.node_indices[0]];
            self.projection
                .visit_element_quadrature(element, |point| {
                    for (&node, gradient) in element
                        .node_indices
                        .iter()
                        .zip(point.physical_shape_gradients())
                    {
                        for axis in 0..3 {
                            integrated[axis] += (temperature_k[node] - reference)
                                * gradient[axis]
                                * point.volume_weight_m3;
                        }
                    }
                    volume += point.volume_weight_m3;
                })
                .map_err(ThermalAssemblyError::from)?;
            Ok::<_, ThermalAssemblyError>(integrated.map(|component| component / volume))
        })
        .into_iter()
        .collect()
    }

    pub(crate) fn recover_element_heat_flux(
        &self,
        temperature_k: &[f64],
    ) -> Result<Vec<[f64; 3]>, ThermalAssemblyError> {
        Ok(self
            .recover_element_temperature_gradients(temperature_k)?
            .into_iter()
            .zip(self.element_conductivity_w_per_mk.canonical_values())
            .map(|(gradient, conductivity)| gradient.map(|value| -conductivity * value))
            .collect())
    }

    pub(crate) fn recover_element_storage_rate(
        &self,
        previous_temperature_k: &[f64],
        temperature_k: &[f64],
        time_step_s: f64,
    ) -> Result<Vec<f64>, ThermalAssemblyError> {
        let elements = &self.projection.elements;
        crate::algebra::element_assembly::ordered_map_range(elements.len(), |element_index| {
            let element = &elements[element_index];
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
                    integrated_rate += self
                        .element_volumetric_capacity_j_per_m3k
                        .canonical_values()[element_index]
                        * temperature_rate
                        * point.volume_weight_m3;
                    volume += point.volume_weight_m3;
                })
                .map_err(ThermalAssemblyError::from)?;
            Ok::<_, ThermalAssemblyError>(integrated_rate / volume)
        })
        .into_iter()
        .collect()
    }
}
