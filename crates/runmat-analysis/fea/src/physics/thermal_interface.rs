//! Declared thermal interfaces between bodies that share a conformal interface: the interface
//! law, the side-owned temperature degrees of freedom of a finite conductance and its operator.
//!
//! A perfect interface shares the conformal interface nodes. A finite conductance (or contact
//! resistance R = 1/h) gives every interface node a second, side-owned degree of freedom used by
//! the elements of the interface's second side, joined by q = h (T_first - T_second). The
//! consistent face mass makes the coupling symmetric positive semidefinite, and the heat it
//! removes from one side is exactly the heat it adds to the other. Heat conduction and
//! conjugate heat transfer both build their interface operator here.
use std::collections::BTreeSet;

use runmat_analysis_core::{AnalysisInterface, AnalysisInterfaceKind, ThermalInterfaceTransfer};
use runmat_meshing_core::BoundaryFaceRole;

use crate::topology::{AnalysisProjection, ProjectedBoundaryFace};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum InterfaceLaw {
    Perfect,
    /// Interface conductance h = 1 / R, W/m^2/K.
    Conductance(f64),
}

impl InterfaceLaw {
    pub(crate) fn of(transfer: ThermalInterfaceTransfer) -> Result<Self, String> {
        let law = match transfer {
            ThermalInterfaceTransfer::Perfect => Self::Perfect,
            ThermalInterfaceTransfer::Conductance(h) => Self::Conductance(h.canonical_value()),
            ThermalInterfaceTransfer::Resistance(r) => Self::Conductance(1.0 / r.canonical_value()),
        };
        if let Self::Conductance(h) = law {
            if !h.is_finite() || h <= 0.0 {
                return Err("interface conductance must be positive and finite".into());
            }
        }
        Ok(law)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InterfacePlan {
    pub interface_id: String,
    pub law: InterfaceLaw,
    /// (projection boundary face, first-side element, second-side element).
    pub faces: Vec<(usize, usize, usize)>,
}

/// Plans every declared thermal interface. `sides` orders the two elements of a face as
/// (first, second). Returns the plans and, per boundary face, whether an interface claimed it.
pub(crate) fn plan<'a>(
    projection: &AnalysisProjection,
    interfaces: impl IntoIterator<Item = &'a AnalysisInterface>,
    mut sides: impl FnMut(usize, usize) -> Result<(usize, usize), String>,
) -> Result<(Vec<InterfacePlan>, Vec<bool>), String> {
    let mut claimed = vec![false; projection.boundary_faces.len()];
    let mut plans = Vec::new();
    for interface in interfaces {
        let AnalysisInterfaceKind::ConjugateHeatTransfer(model) = &interface.kind else {
            return Err(format!(
                "interface `{}` is not a thermal interface",
                interface.interface_id
            ));
        };
        let law = InterfaceLaw::of(model.transfer)?;
        let mut faces = Vec::new();
        for (index, face) in projection.boundary_faces.iter().enumerate() {
            let selected = face.provenance.iter().any(|entity| {
                interface.primary.matches_persistent_entity(entity)
                    || interface.secondary.matches_persistent_entity(entity)
            });
            if !selected {
                continue;
            }
            let [a, b] = face.adjacent_element_indices[..] else {
                return Err(format!(
                    "interface `{}` selects a face that is not a conformal interface between two bodies",
                    interface.interface_id
                ));
            };
            if face.role != BoundaryFaceRole::ConformalInterface {
                return Err(format!(
                    "interface `{}` selects a face whose role is not a conformal interface",
                    interface.interface_id
                ));
            }
            if std::mem::replace(&mut claimed[index], true) {
                return Err(format!(
                    "interface `{}` overlaps another declared interface",
                    interface.interface_id
                ));
            }
            let (first, second) = sides(a, b)?;
            faces.push((index, first, second));
        }
        if faces.is_empty() {
            return Err(format!(
                "interface `{}` selects no conformal interface face",
                interface.interface_id
            ));
        }
        plans.push(InterfacePlan {
            interface_id: interface.interface_id.clone(),
            law,
            faces,
        });
    }
    Ok((plans, claimed))
}

/// Between two solids the second side is the greater region (a deterministic, authored-order
/// independent choice).
pub(crate) fn solid_sides(projection: &AnalysisProjection, a: usize, b: usize) -> (usize, usize) {
    if projection.elements[a].region_id <= projection.elements[b].region_id {
        (a, b)
    } else {
        (b, a)
    }
}

/// The interfaces of a heat-conduction study (every body a solid) and their dofs.
pub(crate) fn solid_conduction(
    projection: &AnalysisProjection,
    interfaces: &[AnalysisInterface],
) -> Result<(Vec<InterfacePlan>, InterfaceDofs), String> {
    if interfaces.is_empty() {
        return Ok((Vec::new(), InterfaceDofs::nodal(projection)));
    }
    let (plans, _) = plan(projection, interfaces, |a, b| {
        Ok(solid_sides(projection, a, b))
    })?;
    let dofs = InterfaceDofs::derive(projection, &plans)?;
    Ok((plans, dofs))
}

/// Temperature degrees of freedom: one per node, plus a second, side-owned one at every node of
/// a finite-conductance interface, used by the elements of the interface's second side.
#[derive(Clone, Debug, PartialEq)]
pub struct InterfaceDofs {
    nodes: usize,
    /// (projection node, second-side dof), sorted by node.
    second: Vec<(usize, usize)>,
    /// Per element: whether it takes the second-side dof of the interface nodes it touches.
    second_side: Vec<bool>,
}

impl InterfaceDofs {
    /// Node dofs only (no finite-conductance interface).
    pub(crate) fn nodal(projection: &AnalysisProjection) -> Self {
        Self {
            nodes: projection.nodes.len(),
            second: Vec::new(),
            second_side: vec![false; projection.elements.len()],
        }
    }

    /// Second-side ownership is by region: every element of a region on the second side of a
    /// conductance interface takes the second-side dof of the interface nodes it touches,
    /// including elements meeting the interface only along an edge or at a vertex. Face
    /// adjacency alone would let those elements short-circuit the jump through the first side.
    pub(crate) fn derive(
        projection: &AnalysisProjection,
        plans: &[InterfacePlan],
    ) -> Result<Self, String> {
        let mut jump_nodes = BTreeSet::new();
        let mut first_regions = BTreeSet::new();
        let mut second_regions = BTreeSet::new();
        for plan in plans {
            if plan.law == InterfaceLaw::Perfect {
                continue;
            }
            for &(face, first, second) in &plan.faces {
                first_regions.insert(&projection.elements[first].region_id);
                second_regions.insert(&projection.elements[second].region_id);
                jump_nodes.extend(projection.boundary_faces[face].node_indices.iter().copied());
            }
        }
        // A region on both sides of conductance interfaces would need two side-owned dofs at
        // one node.
        if first_regions.iter().any(|r| second_regions.contains(r)) {
            return Err(
                "a region lies on both sides of conductance interfaces; split it into one region per side".into(),
            );
        }
        let second_side = projection
            .elements
            .iter()
            .map(|element| {
                second_regions.contains(&element.region_id)
                    && element.node_indices.iter().any(|n| jump_nodes.contains(n))
            })
            .collect();
        let nodes = projection.nodes.len();
        Ok(Self {
            nodes,
            second: jump_nodes
                .into_iter()
                .enumerate()
                .map(|(i, n)| (n, nodes + i))
                .collect(),
            second_side,
        })
    }

    pub(crate) fn count(&self) -> usize {
        self.nodes + self.second.len()
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nodes
    }

    pub(crate) fn is_nodal(&self) -> bool {
        self.second.is_empty()
    }

    /// (projection node, second-side dof) pairs, sorted by node.
    pub(crate) fn second(&self) -> &[(usize, usize)] {
        &self.second
    }

    fn second_of(&self, node: usize) -> Option<usize> {
        self.second
            .binary_search_by_key(&node, |(n, _)| *n)
            .ok()
            .map(|i| self.second[i].1)
    }

    /// The dof an element uses at one of its nodes.
    pub(crate) fn dof(&self, element: usize, node: usize) -> usize {
        if self.second_side[element] {
            if let Some(second) = self.second_of(node) {
                return second;
            }
        }
        node
    }

    pub(crate) fn element_dofs(
        &self,
        projection: &AnalysisProjection,
        element: usize,
    ) -> Vec<usize> {
        projection.elements[element]
            .node_indices
            .iter()
            .map(|&node| self.dof(element, node))
            .collect()
    }

    /// The dofs of a face as seen from one of its adjacent elements.
    pub(crate) fn face_dofs(&self, face: &ProjectedBoundaryFace, element: usize) -> Vec<usize> {
        face.node_indices
            .iter()
            .map(|&node| self.dof(element, node))
            .collect()
    }

    /// Node values and (node, second-side value) pairs.
    pub(crate) fn split(&self, values: &[f64]) -> Result<(Vec<f64>, Vec<(usize, f64)>), String> {
        if values.len() != self.count() {
            return Err("temperature dof inventory".into());
        }
        Ok((
            values[..self.nodes].to_vec(),
            self.second.iter().map(|&(n, d)| (n, values[d])).collect(),
        ))
    }

    pub(crate) fn join(&self, nodes: &[f64], second: &[(usize, f64)]) -> Result<Vec<f64>, String> {
        if nodes.len() != self.nodes
            || second.len() != self.second.len()
            || second
                .iter()
                .zip(&self.second)
                .any(|((a, _), (b, _))| a != b)
            || nodes
                .iter()
                .chain(second.iter().map(|(_, v)| v))
                .any(|v| !v.is_finite())
        {
            return Err("retained temperature inventory".into());
        }
        let mut values = nodes.to_vec();
        values.extend(second.iter().map(|(_, v)| *v));
        Ok(values)
    }

    /// The dof cliques of every conductance coupling: for each finite-conductance interface
    /// face, its first-side and second-side dofs together.
    pub(crate) fn support_cliques(
        &self,
        projection: &AnalysisProjection,
        plans: &[InterfacePlan],
    ) -> Vec<Vec<usize>> {
        let coupled = || plans.iter().filter(|plan| plan.law != InterfaceLaw::Perfect);
        let mut cliques = Vec::with_capacity(coupled().map(|plan| plan.faces.len()).sum());
        for plan in coupled() {
            for &(face, first, second) in &plan.faces {
                let face = &projection.boundary_faces[face];
                let mut dofs = Vec::with_capacity(2 * face.node_indices.len());
                dofs.extend(face.node_indices.iter().map(|&node| self.dof(first, node)));
                dofs.extend(face.node_indices.iter().map(|&node| self.dof(second, node)));
                cliques.push(dofs);
            }
        }
        cliques
    }

    /// Adds the conductance couplings of every interface to the rows of a sparse support.
    pub(crate) fn extend_support(
        &self,
        projection: &AnalysisProjection,
        plans: &[InterfacePlan],
        rows: &mut [BTreeSet<usize>],
    ) {
        for plan in plans {
            if plan.law == InterfaceLaw::Perfect {
                continue;
            }
            for &(face, first, second) in &plan.faces {
                let face = &projection.boundary_faces[face];
                let mut dofs = self.face_dofs(face, first);
                dofs.extend(self.face_dofs(face, second));
                for &row in &dofs {
                    rows[row].extend(dofs.iter().copied());
                }
            }
        }
    }
}

impl InterfaceDofs {
    /// The projection whose nodes are these dofs: every second-side dof becomes a node at its
    /// interface node's position, the second side's elements and exterior faces use it, and the
    /// conductance couplings join the two sides in the node adjacency. Heat conduction runs on
    /// this projection unchanged; the conformal interface faces keep their first-side nodes.
    pub(crate) fn split_projection(
        &self,
        projection: &AnalysisProjection,
        plans: &[InterfacePlan],
    ) -> AnalysisProjection {
        let mut split = projection.clone();
        if self.is_nodal() {
            return split;
        }
        use sha2::{Digest as _, Sha256};
        for &(node, _) in &self.second {
            let original = &projection.nodes[node];
            let mut hasher = Sha256::new();
            hasher.update(b"runmat.thermal.interface.second_side_node.v1");
            hasher.update(original.stable_identity.bytes());
            split.nodes.push(crate::topology::ProjectedNode {
                node_id: original.node_id,
                stable_identity: runmat_meshing_core::StableDigest::from_bytes(
                    hasher.finalize().into(),
                ),
                coordinates_m: original.coordinates_m,
            });
        }
        for (index, element) in split.elements.iter_mut().enumerate() {
            if self.second_side[index] {
                for node in &mut element.node_indices {
                    *node = self.dof(index, *node);
                }
            }
        }
        for face in &mut split.boundary_faces {
            if let [element] = face.adjacent_element_indices[..] {
                for node in &mut face.node_indices {
                    *node = self.dof(element, *node);
                }
            }
        }
        let mut pairs = Vec::new();
        for element in &split.elements {
            for &a in &element.node_indices {
                for &b in &element.node_indices {
                    if a != b {
                        pairs.push((a, b));
                    }
                }
            }
        }
        for plan in plans {
            if plan.law == InterfaceLaw::Perfect {
                continue;
            }
            for &(face, first, second) in &plan.faces {
                let face = &projection.boundary_faces[face];
                let a = self.face_dofs(face, first);
                let b = self.face_dofs(face, second);
                for &x in &a {
                    for &y in &b {
                        pairs.push((x, y));
                        pairs.push((y, x));
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut neighbors = vec![Vec::new(); split.nodes.len()];
        for (a, b) in pairs {
            neighbors[a].push(b);
        }
        split.node_neighbors = neighbors;
        split
    }
}

/// Consistent face mass, integral of N_i N_j over the face.
pub(crate) fn interface_mass(
    projection: &AnalysisProjection,
    face: &ProjectedBoundaryFace,
) -> Result<Vec<f64>, String> {
    let n = face.node_indices.len();
    let mut mass = vec![0.0; n * n];
    projection
        .visit_boundary_quadrature(face, |point| {
            let shapes = point.shape_values();
            for i in 0..n {
                for j in 0..n {
                    mass[i * n + j] += shapes[i] * shapes[j] * point.area_weight_m2;
                }
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(mass)
}

/// The conductance operator h M [[1, -1], [-1, 1]] of every finite-conductance face.
pub(crate) fn add_conductance(
    projection: &AnalysisProjection,
    plans: &[InterfacePlan],
    dofs: &InterfaceDofs,
    mut add: impl FnMut(usize, usize, f64),
) -> Result<(), String> {
    for plan in plans {
        let InterfaceLaw::Conductance(h) = plan.law else {
            continue;
        };
        for &(face, first, second) in &plan.faces {
            let f = &projection.boundary_faces[face];
            let mass = interface_mass(projection, f)?;
            let n = f.node_indices.len();
            let a = dofs.face_dofs(f, first);
            let b = dofs.face_dofs(f, second);
            for i in 0..n {
                for j in 0..n {
                    let m = h * mass[i * n + j];
                    add(a[i], a[j], m);
                    add(a[i], b[j], -m);
                    add(b[i], b[j], m);
                    add(b[i], a[j], -m);
                }
            }
        }
    }
    Ok(())
}
