use super::{
    boundary::MagneticBoundarySpace, MagneticSystemPlan, MagneticVolumeData, MagnetostaticError,
};
use crate::{
    algebra::{
        element_assembly::{
            assemble_elements, no_visit, AssembledElements, ElementAssemblyError, ElementDofMap,
            ElementOutputShape, ElementOutputs,
        },
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
        grounded_faces: &[runmat_meshing_core::StableDigest],
        volumes: &[MagneticVolumeData],
    ) -> Result<Self, MagnetostaticError> {
        let plan = MagneticSystemPlan::derive(projection, grounded_faces)?;
        Self::assemble_with_plan(projection, volumes, &plan)
    }
    pub fn assemble_with_plan(
        projection: &'a AnalysisProjection,
        volumes: &[MagneticVolumeData],
        plan: &MagneticSystemPlan,
    ) -> Result<Self, MagnetostaticError> {
        plan.validate_for(projection)?;
        let edge_space = TetrahedralEdgeSpace::derive(projection)
            .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
        let boundary =
            MagneticBoundarySpace::derive(projection, &edge_space, plan.grounded_faces())?;
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
        if dimension != plan.saddle_pattern().rows() {
            return Err(MagnetostaticError::InvalidInput(
                "planned magnetic dimension mismatch".into(),
            ));
        }
        let length = projection.characteristic_length_m;
        if !length.is_finite() || length <= 0.0 {
            return Err(MagnetostaticError::Topology("invalid length scale".into()));
        }
        let gauge_scale = 1.0 / (length * length);
        validate_affine_geometry(projection)?;
        // Dense row lookups (edge -> free row, node -> gauge row) for the element maps.
        let mut edge_row = vec![None; edge_space.edges().len()];
        for (&edge, &row) in &edge_rows {
            edge_row[edge] = Some(row);
        }
        let mut gauge_row = vec![None; projection.nodes.len()];
        for (&node, &row) in &gauge_rows {
            gauge_row[node] = Some(row);
        }
        let element_count = projection.elements.len();
        let execution = crate::planning::resource::LocalExecutionPlan::serial();
        let operator_error = |e: ElementAssemblyError<MagnetostaticError>| match e {
            ElementAssemblyError::Element { error, .. } => error,
            other => MagnetostaticError::Operator(other.to_string()),
        };
        let edge_indices = |index: usize| {
            edge_space.element_edge_indices(index).ok_or_else(|| {
                MagnetostaticError::Topology("magnetic element has no edge indices".into())
            })
        };
        // Physical curl-curl operator and current load on free edges.
        let edge_map =
            ElementDofMap::from_elements(edge_rows.len(), element_count, execution, |index, out| {
                out.extend(edge_indices(index)?.iter().map(|&edge| edge_row[edge]));
                Ok(())
            })
            .map_err(operator_error)?;
        let physical = assemble_elements(
            &edge_map,
            plan.physical_pattern(),
            ElementOutputShape {
                matrices: 1,
                vectors: 1,
                scalars: 0,
            },
            execution,
            |index, _, out: &mut ElementOutputs<'_, f64>| {
                let moments = edge_space
                    .element_moments(projection, index)
                    .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
                let data = &volumes[index];
                let local = out.matrix(0);
                for i in 0..6 {
                    for j in 0..6 {
                        local[i * 6 + j] = moments.curl_curl[i][j] / data.permeability_h_per_m;
                    }
                }
                for (i, value) in out.vector(0).iter_mut().enumerate() {
                    *value = moments.constant_vector_load[i]
                        .iter()
                        .zip(data.current_density_a_per_m2)
                        .map(|(n, j)| n * j)
                        .sum::<f64>();
                }
                Ok(())
            },
            no_visit,
        )
        .map_err(operator_error)?;
        drop(edge_map);
        // Scaled curl-curl plus weak Coulomb gauge coupling (edges then corner gauge nodes).
        let saddle_map =
            ElementDofMap::from_elements(dimension, element_count, execution, |index, out| {
                out.extend(edge_indices(index)?.iter().map(|&edge| edge_row[edge]));
                out.extend(
                    projection.elements[index].node_indices[..4]
                        .iter()
                        .map(|&node| gauge_row[node]),
                );
                Ok(())
            })
            .map_err(operator_error)?;
        let saddle_values = assemble_elements(
            &saddle_map,
            plan.saddle_pattern(),
            ElementOutputShape {
                matrices: 1,
                ..Default::default()
            },
            execution,
            |index, _, out: &mut ElementOutputs<'_, f64>| {
                let moments = edge_space
                    .element_moments(projection, index)
                    .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
                let data = &volumes[index];
                let local = out.matrix(0);
                for i in 0..6 {
                    for j in 0..6 {
                        local[i * 10 + j] = moments.curl_curl[i][j] / data.permeability_h_per_m
                            * permeability_scale;
                    }
                    for j in 0..4 {
                        let value = moments.gradient_coupling[i][j] * gauge_scale;
                        local[i * 10 + 6 + j] = value;
                        local[(6 + j) * 10 + i] = value;
                    }
                }
                Ok(())
            },
            no_visit,
        )
        .map_err(operator_error)?;
        drop(saddle_map);
        let mut saddle = RealSparseAssembler::from_planned_values(
            plan.saddle_pattern().clone(),
            saddle_values.matrices.into_iter().next().unwrap_or_default(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_INDEFINITE,
        )
        .map_err(|e| MagnetostaticError::Operator(e.to_string()))?;
        saddle.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Electromagnetic,
            contribution_id: ContributionId(
                "magnetic_physical_curl_curl_and_weak_coulomb_gauge".into(),
            ),
        });
        let AssembledElements {
            matrices: physical_matrices,
            vectors: physical_vectors,
            ..
        } = physical;
        let stiffness = RealSparseAssembler::from_planned_values(
            plan.physical_pattern().clone(),
            physical_matrices.into_iter().next().unwrap_or_default(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .map_err(|e| MagnetostaticError::Operator(e.to_string()))?;
        let load = physical_vectors.into_iter().next().unwrap_or_default();
        // No divergence cleaning or source projection is hidden here. Compatible
        // volume current must annihilate each zero-boundary scalar gradient. One pass over
        // the free edges (canonical edge order) accumulates every node's defect and scale.
        let mut defect = vec![0.0_f64; projection.nodes.len()];
        let mut scale = vec![0.0_f64; projection.nodes.len()];
        for (&edge, &row) in &edge_rows {
            let [a, b] = edge_space.edges()[edge].node_indices;
            defect[b] += load[row];
            scale[b] += load[row].abs();
            defect[a] += -load[row];
            scale[a] += (-load[row]).abs();
        }
        for &node in &boundary.gauge_nodes {
            let (defect, scale) = (defect[node], scale[node]);
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
