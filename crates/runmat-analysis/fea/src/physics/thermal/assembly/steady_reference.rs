//! One constant temperature reference per independently connected thermal body.
use runmat_analysis_core::{AnalysisModel, BoundaryOperator};

use super::{sample_ambient_temperature, ThermalAssemblyError, ThermalOperatorAssembly};
use crate::{algebra::connected_components, excitation::ExcitationSamplePoint};

impl ThermalOperatorAssembly {
    pub(crate) fn steady_reference_temperatures(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        initial_reference_k: f64,
    ) -> Result<Vec<f64>, ThermalAssemblyError> {
        // Use precisely the same connectivity definition as scalar nullspace admission.
        // Selecting a reference never supplies an anchor to an underconstrained component.
        let components = connected_components(&self.projection.node_neighbors)
            .map_err(|error| ThermalAssemblyError::SparseOperator(error.to_string()))?;
        let mut component_by_node = vec![0; self.projection.nodes.len()];
        let mut reference_by_component = Vec::with_capacity(components.len());
        for (component_index, nodes) in components.iter().enumerate() {
            reference_by_component.push(
                nodes
                    .iter()
                    .find_map(|&node| self.prescribed_temperature_k.canonical_values()[node]),
            );
            for &node in nodes {
                component_by_node[node] = component_index;
            }
        }
        drop(components);
        for boundary in &model.boundaries {
            let BoundaryOperator::ThermalConvection {
                ambient_temperature,
                coefficient,
            } = &boundary.operator
            else {
                continue;
            };
            if coefficient.canonical_value() <= 0.0 {
                continue;
            }
            let ambient = sample_ambient_temperature(
                model,
                &boundary.boundary_id,
                ambient_temperature,
                sample,
            )?;
            for face in self.projection.boundary_faces_for_target(&boundary.target) {
                let component = component_by_node[face.node_indices[0]];
                reference_by_component[component].get_or_insert(ambient);
            }
        }
        Ok(component_by_node
            .into_iter()
            .map(|component| reference_by_component[component].unwrap_or(initial_reference_k))
            .collect())
    }
}
