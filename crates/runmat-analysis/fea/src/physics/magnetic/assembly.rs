use super::quantity_layout::{
    MagneticPhysicalOperator, MagneticSaddleOperator, MagneticUnknownLayout,
};
use super::{
    boundary::MagneticBoundarySpace, plan::DerivedMagneticSystem, MagneticSystemPlan,
    MagneticVolumeData, MagnetostaticError,
};
use crate::{
    algebra::{
        element_assembly::{
            assemble_elements, no_visit, AssembledElements, ElementAssemblyError, ElementDofMap,
            ElementOutputShape, ElementOutputs,
        },
        ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties, OperatorRole,
        RealSparseAssembler,
    },
    topology::{AnalysisProjection, TetrahedralEdgeSpace},
};
use runmat_meshing_core::ElementOrder;
use runmat_types::physical::{
    kind, Length, MagneticPermeability, Quantity, QuantityLinearOperator, UniformQuantityLayout,
};
use std::collections::BTreeMap;

pub(super) struct MagneticAssembly<'a> {
    pub projection: &'a AnalysisProjection,
    pub edge_space: TetrahedralEdgeSpace,
    pub boundary: MagneticBoundarySpace,
    pub volumes: Vec<MagneticVolumeData>,
    pub saddle: MagneticSaddleOperator,
    pub physical_stiffness: MagneticPhysicalOperator,
    pub physical_load: Quantity<kind::ElectricCurrent, Vec<f64>>,
    pub scaled_rhs: Quantity<kind::MagneticVectorPotential, Vec<f64>>,
    pub permeability_scale: MagneticPermeability,
    /// Prescribed circulation on every grounded edge (zero where A is grounded at zero).
    pub grounded_circulation: Quantity<kind::MagneticFlux, Vec<f64>>,
}

impl<'a> MagneticAssembly<'a> {
    pub fn assemble(
        projection: &'a AnalysisProjection,
        grounded_faces: &[runmat_meshing_core::StableDigest],
        prescribed_potential: &[super::PrescribedVectorPotential],
        volumes: &[MagneticVolumeData],
    ) -> Result<Self, MagnetostaticError> {
        let derived = MagneticSystemPlan::derive_with_spaces(projection, grounded_faces)?;
        Self::assemble_derived(projection, volumes, prescribed_potential, derived)
    }
    pub fn assemble_with_plan(
        projection: &'a AnalysisProjection,
        volumes: &[MagneticVolumeData],
        prescribed_potential: &[super::PrescribedVectorPotential],
        plan: &MagneticSystemPlan,
    ) -> Result<Self, MagnetostaticError> {
        // One derivation both proves the supplied plan exact and yields its spaces.
        let derived = MagneticSystemPlan::derive_with_spaces(projection, plan.grounded_faces())?;
        plan.validate()?;
        if &derived.plan != plan {
            return Err(MagnetostaticError::InvalidInput(
                "magnetic system plan: plan is not exact canonical magnetic support".into(),
            ));
        }
        Self::assemble_derived(projection, volumes, prescribed_potential, derived)
    }
    /// Assemble on a plan derived in this process from this projection: no rederivation.
    pub(crate) fn assemble_derived(
        projection: &'a AnalysisProjection,
        volumes: &[MagneticVolumeData],
        prescribed_potential: &[super::PrescribedVectorPotential],
        derived: DerivedMagneticSystem,
    ) -> Result<Self, MagnetostaticError> {
        if derived.plan.projection_digest() != projection.projection_digest {
            return Err(MagnetostaticError::InvalidInput(
                "magnetic system plan derives from another projection".into(),
            ));
        }
        let DerivedMagneticSystem {
            plan,
            edges: edge_space,
            boundary,
        } = derived;
        let mut inventory = BTreeMap::new();
        for volume in volumes {
            if !volume.permeability_h_per_m.canonical_value().is_finite()
                || volume.permeability_h_per_m.canonical_value() <= 0.0
                || volume
                    .current_density_a_per_m2
                    .canonical_value()
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
        let permeability_scale = volumes[0].permeability_h_per_m.canonical_value();
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
        let grounded_circulation =
            grounded_circulation(projection, &edge_space, &edge_row, prescribed_potential)?;
        let lifted = grounded_circulation.iter().any(|value| *value != 0.0);
        let element_count = projection.elements.len();
        let execution = crate::planning::resource::LocalExecutionPlan::serial();
        let operator_error = |e: ElementAssemblyError<MagnetostaticError>| match e {
            ElementAssemblyError::Element { error, .. } => error,
            ElementAssemblyError::Admission(error) => error,
            ElementAssemblyError::Accounting(error) => MagnetostaticError::Accounting(error),
            other => MagnetostaticError::Operator(other.to_string()),
        };
        let edge_indices = |index: usize| {
            edge_space.element_edge_indices(index).ok_or_else(|| {
                MagnetostaticError::Topology("magnetic element has no edge indices".into())
            })
        };
        // Physical curl-curl operator and current load on free edges.
        let edge_map = ElementDofMap::from_elements(
            edge_rows.len(),
            element_count,
            execution,
            |index, out| {
                out.extend(edge_indices(index)?.iter().map(|&edge| edge_row[edge]));
                Ok(())
            },
        )
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
                        local[i * 6 + j] =
                            moments.curl_curl[i][j] / data.permeability_h_per_m.canonical_value();
                    }
                }
                let edges = edge_indices(index)?;
                for (i, value) in out.vector(0).iter_mut().enumerate() {
                    *value = moments.constant_vector_load[i]
                        .iter()
                        .zip(data.current_density_a_per_m2.canonical_value())
                        .map(|(n, j)| n * j)
                        .sum::<f64>();
                    // Lift the prescribed tangential potential of grounded edges.
                    if lifted {
                        for j in 0..6 {
                            if edge_row[edges[j]].is_none() {
                                *value -= moments.curl_curl[i][j]
                                    / data.permeability_h_per_m.canonical_value()
                                    * grounded_circulation[edges[j]];
                            }
                        }
                    }
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
                        local[i * 10 + j] = moments.curl_curl[i][j]
                            / data.permeability_h_per_m.canonical_value()
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
            saddle_values
                .matrices
                .into_iter()
                .next()
                .unwrap_or_default(),
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
        // The weak gauge holds the whole field divergence-free: grounded edges with a
        // prescribed circulation contribute to every gauge row they touch.
        if lifted {
            for (index, element) in projection.elements.iter().enumerate() {
                let edges = edge_indices(index)?;
                if edges.iter().all(|&edge| grounded_circulation[edge] == 0.0) {
                    continue;
                }
                let moments = edge_space
                    .element_moments(projection, index)
                    .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
                for i in 0..6 {
                    if edge_row[edges[i]].is_some() || grounded_circulation[edges[i]] == 0.0 {
                        continue;
                    }
                    for j in 0..4 {
                        if let Some(row) = gauge_row[element.node_indices[j]] {
                            scaled_rhs[row] -= moments.gradient_coupling[i][j]
                                * gauge_scale
                                * grounded_circulation[edges[i]];
                        }
                    }
                }
            }
        }
        let unknown_layout = MagneticUnknownLayout::try_new(
            edge_rows.len(),
            gauge_rows.len(),
            MagneticPermeability::new(permeability_scale),
            Length::new(length),
        )
        .map_err(|e| MagnetostaticError::Operator(e.into()))?;
        let saddle = QuantityLinearOperator::try_new(
            unknown_layout,
            UniformQuantityLayout::new(dimension),
            saddle
                .finish()
                .map_err(|e| MagnetostaticError::Operator(e.to_string()))?,
        )
        .map_err(|e| MagnetostaticError::Operator(e.to_string()))?;
        let physical_stiffness = QuantityLinearOperator::try_new(
            UniformQuantityLayout::new(edge_rows.len()),
            UniformQuantityLayout::new(edge_rows.len()),
            stiffness
                .finish()
                .map_err(|e| MagnetostaticError::Operator(e.to_string()))?,
        )
        .map_err(|e| MagnetostaticError::Operator(e.to_string()))?;
        Ok(Self {
            projection,
            edge_space,
            boundary,
            volumes,
            saddle,
            physical_stiffness,
            physical_load: Quantity::try_new(load)
                .map_err(|e| MagnetostaticError::Operator(e.to_string()))?,
            scaled_rhs: Quantity::try_new(scaled_rhs)
                .map_err(|e| MagnetostaticError::Operator(e.to_string()))?,
            permeability_scale: MagneticPermeability::new(permeability_scale),
            grounded_circulation: Quantity::try_new(grounded_circulation)
                .map_err(|e| MagnetostaticError::Operator(e.to_string()))?,
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
            let midpoint = projection.nodes[element.node_indices[4 + midside]]
                .coordinates_m
                .canonical_value();
            let left = projection.nodes[element.node_indices[a]]
                .coordinates_m
                .canonical_value();
            let right = projection.nodes[element.node_indices[b]]
                .coordinates_m
                .canonical_value();
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

/// Circulation of the prescribed tangential potential along every grounded edge, in the
/// edge's canonical orientation; edges shared by faces with different values are rejected.
fn grounded_circulation(
    projection: &AnalysisProjection,
    edge_space: &TetrahedralEdgeSpace,
    edge_row: &[Option<usize>],
    prescribed: &[super::PrescribedVectorPotential],
) -> Result<Vec<f64>, MagnetostaticError> {
    let mut circulation = vec![0.0; edge_space.edges().len()];
    if prescribed.is_empty() {
        return Ok(circulation);
    }
    let index_of: BTreeMap<[usize; 2], usize> = edge_space
        .edges()
        .iter()
        .enumerate()
        .map(|(index, edge)| {
            let [a, b] = edge.node_indices;
            ([a.min(b), a.max(b)], index)
        })
        .collect();
    let values: BTreeMap<_, _> = prescribed
        .iter()
        .map(|(face, potential)| (*face, potential.canonical_value()))
        .collect();
    let mut assigned = vec![false; circulation.len()];
    for face in &projection.boundary_faces {
        let Some(potential) = values.get(&face.stable_identity) else {
            continue;
        };
        let corners = &face.node_indices[..3];
        for [a, b] in [[0, 1], [0, 2], [1, 2]] {
            let (u, v) = (corners[a].min(corners[b]), corners[a].max(corners[b]));
            let edge = *index_of.get(&[u, v]).ok_or_else(|| {
                MagnetostaticError::Topology("prescribed face edge absent from edge space".into())
            })?;
            if edge_row[edge].is_some() {
                return Err(MagnetostaticError::Topology(
                    "a prescribed vector potential edge is not grounded".into(),
                ));
            }
            let [start, end] = edge_space.edges()[edge].node_indices;
            let (p, q) = (
                projection.nodes[start].coordinates_m.canonical_value(),
                projection.nodes[end].coordinates_m.canonical_value(),
            );
            let value: f64 = (0..3)
                .map(|axis| potential[axis] * (q[axis] - p[axis]))
                .sum();
            if assigned[edge] && (circulation[edge] - value).abs() > 1e-12 * value.abs().max(1e-30)
            {
                return Err(MagnetostaticError::InvalidInput(
                    "faces sharing an edge prescribe different vector potentials".into(),
                ));
            }
            circulation[edge] = value;
            assigned[edge] = true;
        }
    }
    Ok(circulation)
}
