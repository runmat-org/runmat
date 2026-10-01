//! Lowest-order first-kind Nédélec space, independently of geometry order.
//!
//! Six corner-edge circulations per tetrahedron, including curved Tet10 geometry.
//! Covariant Piola values and contravariant Piola curls use the actual shared
//! isoparametric Jacobian. No edge-length/triangle-area surrogate enters a form.

use super::{
    element::{cross, dot},
    AnalysisProjection, AnalysisProjectionError, VolumeQuadraturePoint,
};
use runmat_execution::Digest;
use runmat_meshing_core::{ElementOrder, StableDigest};
use std::collections::{BTreeMap, BTreeSet};

mod boundary;

const EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/// Orientation is from the lesser persistent node identity to the greater one.
/// Neither node-array order nor the caller's numeric node labels define it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrientedEdge {
    pub node_identities: [StableDigest; 2],
    pub node_indices: [usize; 2],
}

#[derive(Debug, Clone)]
struct ElementEdges {
    identity: StableDigest,
    corners: [usize; 4],
    indices: [usize; 6],
    signs: [f64; 6],
}

#[derive(Debug, Clone)]
pub struct TetrahedralEdgeSpace {
    projection_digest: Digest,
    node_count: usize,
    edges: Vec<OrientedEdge>,
    elements: Vec<ElementEdges>,
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeQuadraturePoint {
    pub position_m: [f64; 3],
    pub volume_weight_m3: f64,
    /// Globally oriented basis, with units m⁻¹ for unit edge circulation.
    pub values: [[f64; 3]; 6],
    /// Physical curls, with units m⁻².
    pub curls: [[f64; 3]; 6],
    pub corner_gradients: [[f64; 3]; 4],
}

/// Unweighted physical element forms. Material coefficients belong to physics,
/// not topology. This is not a constrained operator or a solvability certificate.
#[derive(Debug, Clone)]
pub struct EdgeElementMoments {
    pub mass: [[f64; 6]; 6],
    pub curl_curl: [[f64; 6]; 6],
    /// ∫ Nᵢ dV, for constant physical vector sources.
    pub constant_vector_load: [[f64; 3]; 6],
    /// ∫ Nᵢ · grad λⱼ dV, for discrete gradient/gauge assembly.
    pub gradient_coupling: [[f64; 4]; 6],
}

impl TetrahedralEdgeSpace {
    pub fn derive(projection: &AnalysisProjection) -> Result<Self, AnalysisProjectionError> {
        let mut seen = BTreeSet::new();
        for node in &projection.nodes {
            if !seen.insert(node.stable_identity.0) {
                return Err(AnalysisProjectionError::InvalidEdgeSpace(
                    "duplicate node identity",
                ));
            }
        }
        let mut inventory = BTreeMap::new();
        let mut elements = Vec::with_capacity(projection.elements.len());
        for element in &projection.elements {
            let expected = match element.order {
                ElementOrder::Tet4 => 4,
                ElementOrder::Tet10 => 10,
            };
            if element.node_indices.len() != expected {
                return Err(AnalysisProjectionError::InvalidEdgeSpace(
                    "invalid tetrahedral arity",
                ));
            }
            let corners: [usize; 4] = std::array::from_fn(|i| element.node_indices[i]);
            let mut identities = [[0; 32]; 4];
            for (local, &index) in corners.iter().enumerate() {
                identities[local] = projection
                    .nodes
                    .get(index)
                    .ok_or(AnalysisProjectionError::InvalidEdgeSpace(
                        "unknown corner node",
                    ))?
                    .stable_identity
                    .0;
            }
            if corners.into_iter().collect::<BTreeSet<_>>().len() != 4 {
                return Err(AnalysisProjectionError::InvalidEdgeSpace(
                    "repeated corner node",
                ));
            }
            let mut signs = [0.0; 6];
            for (edge, [a, b]) in EDGES.into_iter().enumerate() {
                let (key, indices, sign) = if identities[a] < identities[b] {
                    (
                        [identities[a], identities[b]],
                        [corners[a], corners[b]],
                        1.0,
                    )
                } else {
                    (
                        [identities[b], identities[a]],
                        [corners[b], corners[a]],
                        -1.0,
                    )
                };
                inventory.insert(key, indices);
                signs[edge] = sign;
            }
            elements.push(ElementEdges {
                identity: element.stable_identity,
                corners,
                indices: [0; 6],
                signs,
            });
        }
        let keys: BTreeMap<_, _> = inventory
            .keys()
            .enumerate()
            .map(|(index, &key)| (key, index))
            .collect();
        for element in &mut elements {
            for (edge, [a, b]) in EDGES.into_iter().enumerate() {
                let mut key = [
                    projection.nodes[element.corners[a]].stable_identity.0,
                    projection.nodes[element.corners[b]].stable_identity.0,
                ];
                key.sort_unstable();
                element.indices[edge] = keys[&key];
            }
        }
        let edges = inventory
            .into_iter()
            .map(|(key, node_indices)| OrientedEdge {
                node_identities: key.map(StableDigest),
                node_indices,
            })
            .collect();
        Ok(Self {
            projection_digest: projection.projection_digest,
            node_count: projection.nodes.len(),
            edges,
            elements,
        })
    }

    pub fn edges(&self) -> &[OrientedEdge] {
        &self.edges
    }

    /// Canonical edge rows on the exact declared exterior tetrahedral faces.
    /// This imposes no connected-ball, gauge or physical boundary policy.
    pub fn exterior_edge_indices(
        &self,
        projection: &AnalysisProjection,
    ) -> Result<Vec<usize>, AnalysisProjectionError> {
        boundary::exterior_edge_indices(self, projection)
    }

    pub fn element_edge_indices(&self, element_index: usize) -> Option<[usize; 6]> {
        self.elements
            .get(element_index)
            .map(|element| element.indices)
    }

    /// Canonical discrete gradient: oriented edge coefficients are endpoint
    /// differences of the corner scalar potential. Tet10 geometry is not a
    /// claim of quadratic edge or quadratic scalar approximation here.
    pub fn gradient(&self, nodal_values: &[f64]) -> Result<Vec<f64>, AnalysisProjectionError> {
        if nodal_values.len() != self.node_count
            || nodal_values.iter().any(|value| !value.is_finite())
        {
            return Err(AnalysisProjectionError::InvalidEdgeSpace(
                "invalid scalar field cardinality or values",
            ));
        }
        self.edges
            .iter()
            .map(|edge| {
                let [a, b] = edge.node_indices;
                let left = nodal_values.get(a).filter(|v| v.is_finite()).ok_or(
                    AnalysisProjectionError::InvalidEdgeSpace("missing or nonfinite scalar value"),
                )?;
                let right = nodal_values.get(b).filter(|v| v.is_finite()).ok_or(
                    AnalysisProjectionError::InvalidEdgeSpace("missing or nonfinite scalar value"),
                )?;
                let difference = right - left;
                if !difference.is_finite() {
                    return Err(AnalysisProjectionError::InvalidEdgeSpace(
                        "nonfinite edge difference",
                    ));
                }
                Ok(difference)
            })
            .collect()
    }

    pub fn visit_element_quadrature(
        &self,
        projection: &AnalysisProjection,
        element_index: usize,
        mut visit: impl FnMut(EdgeQuadraturePoint),
    ) -> Result<(), AnalysisProjectionError> {
        let stored = self
            .elements
            .get(element_index)
            .ok_or(AnalysisProjectionError::InvalidEdgeSpace("unknown element"))?;
        let element = projection.elements.get(element_index).ok_or(
            AnalysisProjectionError::InvalidEdgeSpace("unknown projected element"),
        )?;
        if self.projection_digest != projection.projection_digest
            || stored.identity != element.stable_identity
            || element.node_indices.get(..4) != Some(stored.corners.as_slice())
        {
            return Err(AnalysisProjectionError::InvalidEdgeSpace(
                "projection identity mismatch",
            ));
        }
        projection.visit_element_quadrature(element, |point| visit(basis(point, stored.signs)))
    }

    /// Uses the shared 15-point volume rule. Exact for affine lowest-order mass
    /// and curl-curl forms; curved rational integrands require separate quadrature
    /// convergence evidence before formulation admission.
    pub fn element_moments(
        &self,
        projection: &AnalysisProjection,
        element_index: usize,
    ) -> Result<EdgeElementMoments, AnalysisProjectionError> {
        let mut result = EdgeElementMoments {
            mass: [[0.0; 6]; 6],
            curl_curl: [[0.0; 6]; 6],
            constant_vector_load: [[0.0; 3]; 6],
            gradient_coupling: [[0.0; 4]; 6],
        };
        self.visit_element_quadrature(projection, element_index, |point| {
            let w = point.volume_weight_m3;
            for i in 0..6 {
                for j in 0..6 {
                    result.mass[i][j] += w * dot(point.values[i], point.values[j]);
                    result.curl_curl[i][j] += w * dot(point.curls[i], point.curls[j]);
                }
                for axis in 0..3 {
                    result.constant_vector_load[i][axis] += w * point.values[i][axis];
                }
                for j in 0..4 {
                    result.gradient_coupling[i][j] +=
                        w * dot(point.values[i], point.corner_gradients[j]);
                }
            }
        })?;
        Ok(result)
    }
}

fn basis(point: VolumeQuadraturePoint, signs: [f64; 6]) -> EdgeQuadraturePoint {
    let gradients = point.physical_barycentric_gradients;
    let values = std::array::from_fn(|edge| {
        let [a, b] = EDGES[edge];
        std::array::from_fn(|axis| {
            signs[edge]
                * (point.barycentric[a] * gradients[b][axis]
                    - point.barycentric[b] * gradients[a][axis])
        })
    });
    let curls = std::array::from_fn(|edge| {
        let [a, b] = EDGES[edge];
        cross(gradients[a], gradients[b]).map(|value| 2.0 * signs[edge] * value)
    });
    EdgeQuadraturePoint {
        position_m: point.position_m,
        volume_weight_m3: point.volume_weight_m3,
        values,
        curls,
        corner_gradients: gradients,
    }
}

#[cfg(test)]
mod tests;
