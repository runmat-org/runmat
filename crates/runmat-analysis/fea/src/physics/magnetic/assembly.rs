use super::{boundary::MagneticBoundarySpace, MagneticVolumeData, MagnetostaticError};
use crate::{
    algebra::{
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler, RealSparseMatrix,
    },
    topology::{AnalysisProjection, TetrahedralEdgeSpace},
};
use runmat_meshing_core::ElementOrder;
use std::collections::BTreeMap;

pub(super) struct MagneticAssembly<'a> {
    pub projection: &'a AnalysisProjection,
    pub edge_space: TetrahedralEdgeSpace,
    pub boundary: MagneticBoundarySpace,
    pub volumes: Vec<MagneticVolumeData>,
    pub saddle: RealSparseMatrix,
    pub physical_stiffness: RealSparseMatrix,
    pub physical_load: Vec<f64>,
    pub scaled_rhs: Vec<f64>,
    pub permeability_scale: f64,
}

impl<'a> MagneticAssembly<'a> {
    pub fn assemble(
        projection: &'a AnalysisProjection,
        volumes: &[MagneticVolumeData],
    ) -> Result<Self, MagnetostaticError> {
        let edge_space = TetrahedralEdgeSpace::derive(projection)
            .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
        let boundary = MagneticBoundarySpace::derive(projection, &edge_space)?;
        let mut inventory = BTreeMap::new();
        for volume in volumes {
            if !volume.permeability_h_per_m.is_finite()
                || volume.permeability_h_per_m <= 0.0
                || volume
                    .current_density_a_per_m2
                    .iter()
                    .any(|v| !v.is_finite())
                || inventory.insert(volume.element_identity, *volume).is_some()
            {
                return Err(MagnetostaticError::InvalidInput(
                    "invalid or duplicate magnetic element law/source".into(),
                ));
            }
        }
        let volumes: Vec<_> = projection
            .elements
            .iter()
            .map(|element| {
                inventory.remove(&element.stable_identity).ok_or_else(|| {
                    MagnetostaticError::InvalidInput("unassigned magnetic volume".into())
                })
            })
            .collect::<Result<_, _>>()?;
        if !inventory.is_empty() {
            return Err(MagnetostaticError::InvalidInput(
                "foreign magnetic element identity".into(),
            ));
        }
        // Scalar rescaling of the entire physical equation is invertible and
        // changes neither source compatibility nor the solution/physical residual.
        let permeability_scale = volumes[0].permeability_h_per_m;
        let edge_rows: BTreeMap<_, _> = boundary
            .free_edges
            .iter()
            .enumerate()
            .map(|(row, &edge)| (edge, row))
            .collect();
        let gauge_rows: BTreeMap<_, _> = boundary
            .gauge_nodes
            .iter()
            .enumerate()
            .map(|(row, &node)| (node, row + boundary.free_edges.len()))
            .collect();
        let dimension = edge_rows
            .len()
            .checked_add(gauge_rows.len())
            .ok_or_else(|| {
                MagnetostaticError::InvalidInput("magnetic dimension overflow".into())
            })?;
        let mut saddle = RealSparseAssembler::square(
            dimension,
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_INDEFINITE,
        );
        saddle.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Electromagnetic,
            contribution_id: ContributionId(
                "magnetic_physical_curl_curl_and_weak_coulomb_gauge".into(),
            ),
        });
        let mut stiffness = RealSparseAssembler::square(
            edge_rows.len(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        let mut load = vec![0.0; edge_rows.len()];
        let length = projection.characteristic_length_m;
        if !length.is_finite() || length <= 0.0 {
            return Err(MagnetostaticError::Topology("invalid length scale".into()));
        }
        let gauge_scale = 1.0 / (length * length);
        validate_affine_geometry(projection)?;
        for (index, (element, data)) in projection.elements.iter().zip(&volumes).enumerate() {
            let moments = edge_space
                .element_moments(projection, index)
                .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
            let indices = edge_space.element_edge_indices(index).unwrap();
            for (i, &edge) in indices.iter().enumerate() {
                let Some(&row) = edge_rows.get(&edge) else {
                    continue;
                };
                load[row] += moments.constant_vector_load[i]
                    .iter()
                    .zip(data.current_density_a_per_m2)
                    .map(|(n, j)| n * j)
                    .sum::<f64>();
                for (j, &other) in indices.iter().enumerate() {
                    if let Some(&column) = edge_rows.get(&other) {
                        let value = moments.curl_curl[i][j] / data.permeability_h_per_m;
                        stiffness.add(row, column, value);
                        saddle.add(row, column, value * permeability_scale);
                    }
                }
                for (j, &node) in element.node_indices[..4].iter().enumerate() {
                    if let Some(&gauge) = gauge_rows.get(&node) {
                        let value = moments.gradient_coupling[i][j] * gauge_scale;
                        saddle.add(row, gauge, value);
                        saddle.add(gauge, row, value);
                    }
                }
            }
        }
        // No divergence cleaning or source projection is hidden here. Compatible
        // volume current must annihilate each zero-boundary scalar gradient.
        for &node in &boundary.gauge_nodes {
            let mut defect = 0.0;
            let mut scale = 0.0;
            for (&edge, &row) in &edge_rows {
                let [a, b] = edge_space.edges()[edge].node_indices;
                let sign = if b == node {
                    1.0
                } else if a == node {
                    -1.0
                } else {
                    0.0
                };
                defect += sign * load[row];
                scale += (sign * load[row]).abs();
            }
            if defect.abs() > 1e-11 * scale {
                return Err(MagnetostaticError::IncompatibleCurrent(if scale > 0.0 {
                    defect.abs() / scale
                } else {
                    defect.abs()
                }));
            }
        }
        let mut scaled_rhs = vec![0.0; dimension];
        for (rhs, value) in scaled_rhs.iter_mut().zip(&load) {
            *rhs = permeability_scale * value;
        }
        Ok(Self {
            projection,
            edge_space,
            boundary,
            volumes,
            saddle: saddle
                .finish()
                .map_err(|e| MagnetostaticError::Operator(e.to_string()))?,
            physical_stiffness: stiffness
                .finish()
                .map_err(|e| MagnetostaticError::Operator(e.to_string()))?,
            physical_load: load,
            scaled_rhs,
            permeability_scale,
        })
    }
}

/// Eligibility check shared by numerical execution and cold validation; no
/// physical form integration, operator assembly or field reconstruction.
pub(super) fn validate_affine_geometry(
    projection: &AnalysisProjection,
) -> Result<(), MagnetostaticError> {
    let length = projection.characteristic_length_m;
    if !length.is_finite() || length <= 0.0 {
        return Err(MagnetostaticError::Topology("invalid length scale".into()));
    }
    for element in &projection.elements {
        if element.order != ElementOrder::Tet10 {
            continue;
        }
        for (midside, [a, b]) in runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS
            .into_iter()
            .enumerate()
        {
            let midpoint = projection.nodes[element.node_indices[4 + midside]].coordinates_m;
            let left = projection.nodes[element.node_indices[a]].coordinates_m;
            let right = projection.nodes[element.node_indices[b]].coordinates_m;
            if (0..3).any(|axis| {
                (midpoint[axis] - (left[axis] + right[axis]) * 0.5).abs()
                    > 64.0 * f64::EPSILON * length
            }) {
                return Err(MagnetostaticError::Unsupported(
                    "general curved magnetic form quadrature".into(),
                ));
            }
        }
    }
    Ok(())
}
