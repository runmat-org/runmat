use runmat_analysis_core::{FieldLocation, Signal, SignalShape};
use runmat_execution::Digest;
use serde::Serialize;

use super::{
    model::{ComplexComponent, ComplexScalar, ComplexVector3},
    realize::RealizationError,
    sample::{sample_signal, ExcitationSamplePoint, SampledValue},
};

/// Piecewise-constant scalar values in the sealed mesh's canonical volume-element row order.
/// The complete topology identity binds rows, including rows outside a selected load region.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ElementScalarField {
    mesh_digest: Digest,
    values: Vec<ComplexScalar>,
}

impl ElementScalarField {
    pub(crate) fn sample<Q>(
        signal: &Signal<Q>,
        sample: ExcitationSamplePoint<'_>,
        component: impl Fn(&Q) -> f64,
    ) -> Result<Self, RealizationError> {
        let SignalShape::TopologyField {
            topology_id,
            location: FieldLocation::Element,
            component_count: 1,
        } = &signal.shape
        else {
            return Err(RealizationError::new(
                "volume scalar fields require canonical element location and one component",
            ));
        };
        let mesh_digest = topology_id.parse::<Digest>().map_err(|_| {
            RealizationError::new("element scalar-field topology identity is not a digest")
        })?;
        let SampledValue::Tensor(tensor) = sample_signal(signal, sample, component)? else {
            return Err(RealizationError::new(
                "element scalar fields require an entity-by-component tensor",
            ));
        };
        let [rows, 1] = tensor.shape.as_slice() else {
            return Err(RealizationError::new(
                "element scalar-field tensor must have shape [positive elements, 1]",
            ));
        };
        if *rows == 0 {
            return Err(RealizationError::new("element scalar fields require rows"));
        }
        let values = tensor
            .real
            .into_iter()
            .zip(tensor.imaginary)
            .map(|(real, imaginary)| ComplexScalar { real, imaginary })
            .collect();
        Ok(Self {
            mesh_digest,
            values,
        })
    }

    pub(crate) fn validate_mesh(
        &self,
        mesh_digest: Digest,
        element_count: usize,
    ) -> Result<(), RealizationError> {
        if self.mesh_digest != mesh_digest || self.values.len() != element_count {
            return Err(RealizationError::new(
                "element scalar field differs from exact canonical mesh identity or element count",
            ));
        }
        Ok(())
    }

    pub(crate) fn is_real(&self) -> bool {
        self.values.iter().all(|value| value.imaginary == 0.0)
    }

    pub(crate) fn value_at(
        &self,
        canonical_element_row: usize,
    ) -> Result<ComplexScalar, RealizationError> {
        self.values
            .get(canonical_element_row)
            .copied()
            .ok_or_else(|| {
                RealizationError::new("element scalar-field row is outside canonical topology")
            })
    }

    pub(super) fn component(mut self, component: ComplexComponent) -> Self {
        for value in &mut self.values {
            *value = value.component(component);
        }
        self
    }
}

/// Frozen canonical-node values. Location and complete mesh identity bind row ordering;
/// these values are sampled before assembly, never by a callback inside quadrature.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct NodalVectorField {
    mesh_digest: Digest,
    values: Vec<ComplexVector3>,
}

impl NodalVectorField {
    pub(crate) fn sample<Q>(
        signal: &Signal<Q>,
        sample: ExcitationSamplePoint<'_>,
        component: impl Fn(&Q) -> f64,
    ) -> Result<Self, RealizationError> {
        let SignalShape::TopologyField {
            topology_id,
            location: FieldLocation::Node,
            component_count: 3,
        } = &signal.shape
        else {
            return Err(RealizationError::new(
                "volume vector fields require canonical node location and three components",
            ));
        };
        let mesh_digest = topology_id
            .parse::<Digest>()
            .map_err(|error| RealizationError::new(error.to_string()))?;
        let SampledValue::Tensor(tensor) = sample_signal(signal, sample, component)? else {
            return Err(RealizationError::new(
                "topology-field sample requires an entity-by-component tensor",
            ));
        };
        let [rows, 3] = tensor.shape.as_slice() else {
            return Err(RealizationError::new(
                "nodal vector-field tensor must have shape [positive nodes, 3]",
            ));
        };
        if *rows == 0 {
            return Err(RealizationError::new("nodal field has no entities"));
        }
        let values = tensor
            .real
            .chunks_exact(3)
            .zip(tensor.imaginary.chunks_exact(3))
            .map(|(real, imaginary)| ComplexVector3 {
                real: [real[0], real[1], real[2]],
                imaginary: [imaginary[0], imaginary[1], imaginary[2]],
            })
            .collect();
        Ok(Self {
            mesh_digest,
            values,
        })
    }

    pub(crate) fn validate_mesh(
        &self,
        mesh_digest: Digest,
        node_count: usize,
    ) -> Result<(), RealizationError> {
        if self.mesh_digest != mesh_digest || self.values.len() != node_count {
            return Err(RealizationError::new(
                "nodal field differs from the exact canonical mesh identity or node count",
            ));
        }
        Ok(())
    }

    pub(crate) fn real_at(
        &self,
        node_indices: &[usize],
        shapes: &[f64],
    ) -> Result<[f64; 3], RealizationError> {
        if node_indices.len() != shapes.len() || node_indices.is_empty() {
            return Err(RealizationError::new(
                "field interpolation requires complete element shape values",
            ));
        }
        let mut result = [0.0; 3];
        for (&node_index, &shape) in node_indices.iter().zip(shapes) {
            let value = self.values.get(node_index).ok_or_else(|| {
                RealizationError::new("field interpolation references an absent node")
            })?;
            for (component, result) in result.iter_mut().enumerate() {
                *result += shape * value.real[component];
            }
        }
        if result.iter().any(|value| !value.is_finite()) {
            return Err(RealizationError::new("field interpolation is not finite"));
        }
        Ok(result)
    }

    pub(super) fn component(mut self, component: ComplexComponent) -> Self {
        for value in &mut self.values {
            *value = value.component(component);
        }
        self
    }
}
