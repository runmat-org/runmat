//! A kernel proof for an assembled linear-elastic operator, with no global factorization.
//!
//! `K = Σₑ Kₑ`, where every element block `Kₑ` acts on the element's `3n` nodal translations.
//! The proof has three parts, each checked on data the assembly actually used:
//!
//! 1. **Elements.** Every block annihilates the element's six rigid motions to a scale-relative
//!    tolerance, and is positive definite on their orthogonal complement with a roundoff-relative
//!    margin: the projected `d × d` block `A` minus `c·(d+1)·ε·tr(A)` (Rump's floating-point
//!    Cholesky criterion, `c` = [`ELASTIC_ELEMENT_MARGIN_ROUNDING_FACTOR`]) admits a Cholesky
//!    factorization, so `A` is positive definite despite the rounding of its formation and
//!    factorization. Element shape quality is the mesher's concern, not this proof's. So `Kₑ` is positive semidefinite with kernel exactly the rigid
//!    motions of that element.
//! 2. **Gluing.** Elements that share a face (three non-collinear corner nodes) share one rigid
//!    motion, since three non-collinear points fix a rigid motion. Face adjacency partitions the
//!    elements into glued components; inside a component every kernel vector is one rigid motion.
//! 3. **Anchors.** The constrained translations of every component determine its rigid motion:
//!    the 6×6 Gram matrix of the rigid basis rows at the constrained DOFs, in component-scaled
//!    coordinates, is positive definite with margin [`ANCHOR_MINIMUM_RANK_RATIO`] of its trace.
//!
//! Then `uᵀKu = 0` with `u = 0` on the constrained DOFs forces `u = 0` on every node, so the
//! operator restricted to the free DOFs is positive definite in exact arithmetic. The proof is
//! qualitative: it does not bound the smallest eigenvalue, so solution accuracy is accepted by the
//! solver's true-residual (backward-error) certificate, never by an eigenvalue bound.
use std::collections::HashMap;

/// The element margin factor and rigid-mode tolerance are the shared element rigidity rule's
/// (`runmat_meshing_core::element_rigidity`), from which the mesher also derives its volume
/// shape floor.
pub use runmat_meshing_core::{
    ELEMENT_RIGIDITY_MARGIN_ROUNDING_FACTOR as ELASTIC_ELEMENT_MARGIN_ROUNDING_FACTOR,
    RIGID_MODE_RESIDUAL_RATIO,
};
/// Smallest admitted ratio of the anchor Gram matrix's smallest eigenvalue to its trace.
pub const ANCHOR_MINIMUM_RANK_RATIO: f64 = 1.0e-8;
/// Smallest admitted area of a shared face relative to its longest edge squared.
const FACE_MINIMUM_AREA_RATIO: f64 = 1.0e-8;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ElasticRigidityError {
    #[error("element {element}: elastic rigidity needs a tetrahedron with at least four nodes, found {nodes}")]
    UnsupportedElement { element: u64, nodes: usize },
    #[error("element {element}: its nodes do not span the six rigid motions")]
    DegenerateElementGeometry { element: u64 },
    #[error("element {element}: stiffness does not annihilate rigid motions (‖K U‖/‖K‖ = {}, tolerance {RIGID_MODE_RESIDUAL_RATIO})", f64::from_bits(*.residual_ratio_bits))]
    RigidMotionEnergy {
        element: u64,
        residual_ratio_bits: u64,
    },
    #[error("element {element}: stiffness is not positive definite on the rigid-motion complement beyond rounding (smallest energy ≈ {} ‖K‖, required margin {} ‖K‖)", f64::from_bits(*.smallest_ratio_bits), f64::from_bits(*.margin_ratio_bits))]
    ElementEnergyMargin {
        element: u64,
        /// The required roundoff-relative margin over `‖Kₑ‖_F`.
        margin_ratio_bits: u64,
        /// Bisection estimate of the projected block's smallest eigenvalue over `‖Kₑ‖_F`.
        smallest_ratio_bits: u64,
    },
    #[error("node {node} has a free translation but belongs to no element")]
    UncoveredNode { node: u64 },
    #[error("glued component {component} ({elements} elements, including element {representative_element}) is a mechanism: its {constrained_dofs} constrained DOFs do not fix all six rigid motions")]
    UnanchoredComponent {
        component: usize,
        elements: usize,
        representative_element: u64,
        constrained_dofs: usize,
    },
    #[error("elastic rigidity capture does not match the assembled mesh")]
    Shape,
    #[error("elastic rigidity proof cancelled")]
    Cancelled,
}

/// Completed proof facts. Built only by [`ElasticRigidityCapture::finish`] in this process.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ElasticRigidityProof {
    fixed: Vec<bool>,
    elements: u64,
    components: u64,
    scalar_operations: u64,
    resident_bytes: u64,
}

impl ElasticRigidityProof {
    pub fn fixed(&self) -> &[bool] {
        &self.fixed
    }
    pub fn elements(&self) -> u64 {
        self.elements
    }
    pub fn components(&self) -> u64 {
        self.components
    }
    pub fn scalar_operations(&self) -> u64 {
        self.scalar_operations
    }
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// Streams element stiffness blocks in canonical element order and checks each one at once, so
/// no element matrix is retained.
pub(crate) struct ElasticRigidityCapture {
    coordinates: Vec<[f64; 3]>,
    node_ids: Vec<u64>,
    fixed: Vec<bool>,
    element_ids: Vec<u64>,
    element_nodes: Vec<u32>,
    element_offsets: Vec<usize>,
    captured: usize,
    operations: u64,
    failure: Option<ElasticRigidityError>,
}

impl ElasticRigidityCapture {
    /// Bytes retained by the capture and the finishing pass for `nodes` nodes and elements of the
    /// given widths.
    pub fn resident_bytes_for(nodes: u64, widths: impl Iterator<Item = usize>) -> u64 {
        let (elements, entries) = widths.fold((0_u64, 0_u64), |(e, n), w| (e + 1, n + w as u64));
        // Coordinates, ids and fixed flags per node; ids, offsets, union-find and component
        // facts per element; node lists; the face map holds at most four faces per element.
        nodes
            .saturating_mul(24 + 8 + 3 + 16)
            .saturating_add(elements.saturating_mul(8 + 8 + 8 + 4 * 24 + 96))
            .saturating_add(entries.saturating_mul(4))
            .saturating_add(1 << 16)
    }

    pub fn new(
        nodes: impl Iterator<Item = (u64, [f64; 3])>,
        elements: impl Iterator<Item = (u64, Vec<u32>)>,
        fixed: Vec<bool>,
    ) -> Result<Self, ElasticRigidityError> {
        let (node_ids, coordinates): (Vec<_>, Vec<_>) = nodes.unzip();
        if fixed.len() != coordinates.len() * 3 {
            return Err(ElasticRigidityError::Shape);
        }
        let mut element_ids = Vec::new();
        let mut element_nodes = Vec::new();
        let mut element_offsets = vec![0];
        for (id, nodes) in elements {
            if nodes.len() < 4 {
                return Err(ElasticRigidityError::UnsupportedElement {
                    element: id,
                    nodes: nodes.len(),
                });
            }
            if nodes.iter().any(|&node| node as usize >= coordinates.len()) {
                return Err(ElasticRigidityError::Shape);
            }
            element_ids.push(id);
            element_nodes.extend(nodes);
            element_offsets.push(element_nodes.len());
        }
        Ok(Self {
            coordinates,
            node_ids,
            fixed,
            element_ids,
            element_nodes,
            element_offsets,
            captured: 0,
            operations: 0,
            failure: None,
        })
    }

    fn nodes_of(&self, element: usize) -> &[u32] {
        &self.element_nodes[self.element_offsets[element]..self.element_offsets[element + 1]]
    }

    /// Checks the next element block: `dofs` are three per node, node-major; `stiffness` is the
    /// row-major block over them.
    pub fn capture(&mut self, dofs: &[u32], stiffness: &[f64]) -> Result<(), ElasticRigidityError> {
        let element = self.captured;
        self.captured += 1;
        if self.failure.is_some() {
            return Ok(());
        }
        if element >= self.element_ids.len() {
            return Err(ElasticRigidityError::Shape);
        }
        let nodes = self.nodes_of(element);
        if dofs.len() != nodes.len() * 3
            || stiffness.len() != dofs.len() * dofs.len()
            || dofs
                .iter()
                .enumerate()
                .any(|(i, &dof)| dof as usize != nodes[i / 3] as usize * 3 + i % 3)
        {
            return Err(ElasticRigidityError::Shape);
        }
        let points = nodes
            .iter()
            .map(|&node| self.coordinates[node as usize])
            .collect::<Vec<_>>();
        let id = self.element_ids[element];
        let d = dofs.len();
        self.operations = self.operations.saturating_add((d * d * 40) as u64);
        if let Err(failure) = check_element(id, &points, stiffness) {
            self.failure = Some(failure);
        }
        Ok(())
    }

    pub fn finish(self) -> Result<ElasticRigidityProof, ElasticRigidityError> {
        if let Some(failure) = self.failure {
            return Err(failure);
        }
        let elements = self.element_ids.len();
        if self.captured != elements || elements == 0 {
            return Err(ElasticRigidityError::Shape);
        }
        // Face-adjacency gluing over the four corner faces of every tetrahedron.
        let mut parent = (0..elements as u32).collect::<Vec<_>>();
        let mut faces: HashMap<[u32; 3], u32> = HashMap::with_capacity(elements * 2);
        for element in 0..elements {
            if element % 4096 == 0 && crate::progress::is_cancelled() {
                return Err(ElasticRigidityError::Cancelled);
            }
            let c = &self.nodes_of(element)[..4];
            for skip in 0..4 {
                let mut face = [0_u32; 3];
                let mut k = 0;
                for (i, &node) in c.iter().enumerate() {
                    if i != skip {
                        face[k] = node;
                        k += 1;
                    }
                }
                face.sort_unstable();
                if !self.non_collinear(face) {
                    continue;
                }
                match faces.entry(face) {
                    std::collections::hash_map::Entry::Occupied(other) => {
                        union(&mut parent, element as u32, *other.get());
                    }
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(element as u32);
                    }
                }
            }
        }
        drop(faces);
        // Every free translation lies in some element.
        let mut covered = vec![false; self.coordinates.len()];
        for &node in &self.element_nodes {
            covered[node as usize] = true;
        }
        for (node, covered) in covered.iter().enumerate() {
            if !covered
                && self.fixed[3 * node..3 * node + 3]
                    .iter()
                    .any(|fixed| !fixed)
            {
                return Err(ElasticRigidityError::UncoveredNode {
                    node: self.node_ids[node],
                });
            }
        }
        drop(covered);
        // Components with their bounding boxes (the scale of the anchor test).
        let mut component_of = vec![0_u32; elements];
        let mut index: HashMap<u32, usize> = HashMap::new();
        let mut boxes: Vec<([f64; 3], [f64; 3], usize, u64)> = Vec::new();
        for element in 0..elements {
            let root = find(&mut parent, element as u32);
            let next = index.len();
            let component = *index.entry(root).or_insert(next);
            if component == boxes.len() {
                boxes.push((
                    [f64::INFINITY; 3],
                    [f64::NEG_INFINITY; 3],
                    0,
                    self.element_ids[element],
                ));
            }
            component_of[element] = component as u32;
            let entry = &mut boxes[component];
            entry.2 += 1;
            for &node in self.nodes_of(element) {
                let p = self.coordinates[node as usize];
                for axis in 0..3 {
                    entry.0[axis] = entry.0[axis].min(p[axis]);
                    entry.1[axis] = entry.1[axis].max(p[axis]);
                }
            }
        }
        drop(parent);
        drop(index);
        // Anchor Gram matrices from each component's distinct constrained nodes.
        let mut gram = vec![[0.0_f64; 36]; boxes.len()];
        let mut constrained = vec![0_usize; boxes.len()];
        let mut seen = std::collections::HashSet::new();
        for element in 0..elements {
            let component = component_of[element] as usize;
            let (low, high, _, _) = boxes[component];
            let center = [0, 1, 2].map(|a| 0.5 * (low[a] + high[a]));
            let scale = (0..3)
                .map(|a| high[a] - low[a])
                .fold(0.0_f64, f64::max)
                .max(f64::MIN_POSITIVE);
            for &node in self.nodes_of(element) {
                let n = node as usize;
                if !self.fixed[3 * n..3 * n + 3].iter().any(|&fixed| fixed)
                    || !seen.insert((component as u32, node))
                {
                    continue;
                }
                let p = self.coordinates[n];
                let q = [0, 1, 2].map(|a| (p[a] - center[a]) / scale);
                for axis in 0..3 {
                    if !self.fixed[3 * n + axis] {
                        continue;
                    }
                    constrained[component] += 1;
                    let row = rigid_row(q, axis);
                    for i in 0..6 {
                        for j in 0..6 {
                            gram[component][i * 6 + j] += row[i] * row[j];
                        }
                    }
                }
            }
        }
        for (component, matrix) in gram.iter().enumerate() {
            let trace = (0..6).map(|i| matrix[i * 6 + i]).sum::<f64>();
            let mut shifted = *matrix;
            for i in 0..6 {
                shifted[i * 6 + i] -= ANCHOR_MINIMUM_RANK_RATIO * trace;
            }
            if !(trace > 0.0) || !cholesky_in_place(&mut shifted, 6) {
                return Err(ElasticRigidityError::UnanchoredComponent {
                    component,
                    elements: boxes[component].2,
                    representative_element: boxes[component].3,
                    constrained_dofs: constrained[component],
                });
            }
        }
        let resident_bytes = Self::resident_bytes_for(
            self.coordinates.len() as u64,
            (0..elements)
                .map(|element| self.element_offsets[element + 1] - self.element_offsets[element]),
        );
        Ok(ElasticRigidityProof {
            elements: elements as u64,
            components: boxes.len() as u64,
            scalar_operations: self
                .operations
                .saturating_add((self.element_nodes.len() as u64).saturating_mul(64)),
            resident_bytes,
            fixed: self.fixed,
        })
    }

    fn non_collinear(&self, face: [u32; 3]) -> bool {
        let [a, b, c] = face.map(|node| self.coordinates[node as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let w = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let area = norm(&cross);
        let longest = [norm(&u), norm(&v), norm(&w)]
            .into_iter()
            .fold(0.0, f64::max);
        area > FACE_MINIMUM_AREA_RATIO * longest * longest
    }
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra.max(rb) as usize] = ra.min(rb);
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Translation `axis` component of the six rigid motions at scaled point `q`: three translations,
/// then rotations about x, y and z.
fn rigid_row(q: [f64; 3], axis: usize) -> [f64; 6] {
    let mut row = [0.0; 6];
    row[axis] = 1.0;
    let rotations = [[0.0, -q[2], q[1]], [q[2], 0.0, -q[0]], [-q[1], q[0], 0.0]];
    for (r, rotation) in rotations.iter().enumerate() {
        row[3 + r] = rotation[axis];
    }
    row
}

/// Lower Cholesky of a row-major symmetric `n × n` matrix; false when a pivot is not positive.
fn cholesky_in_place(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut pivot = a[j * n + j];
        for k in 0..j {
            pivot -= a[j * n + k] * a[j * n + k];
        }
        if !(pivot > 0.0) || !pivot.is_finite() {
            return false;
        }
        let pivot = pivot.sqrt();
        a[j * n + j] = pivot;
        for i in j + 1..n {
            let mut value = a[i * n + j];
            for k in 0..j {
                value -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = value / pivot;
        }
    }
    true
}

/// Element test: the shared element rigidity rule, reported against this element.
fn check_element(element: u64, points: &[[f64; 3]], k: &[f64]) -> Result<(), ElasticRigidityError> {
    use runmat_meshing_core::ElementRigidityShortfall as Shortfall;
    runmat_meshing_core::element_rigidity(points, k).map_err(|shortfall| match shortfall {
        Shortfall::DegenerateGeometry => {
            ElasticRigidityError::DegenerateElementGeometry { element }
        }
        Shortfall::RigidMotionEnergy { residual_ratio } => {
            ElasticRigidityError::RigidMotionEnergy {
                element,
                residual_ratio_bits: residual_ratio.to_bits(),
            }
        }
        Shortfall::EnergyMargin {
            margin_ratio,
            smallest_ratio,
        } => ElasticRigidityError::ElementEnergyMargin {
            element,
            margin_ratio_bits: margin_ratio.to_bits(),
            smallest_ratio_bits: smallest_ratio.to_bits(),
        },
    })
}

#[cfg(test)]
mod tests;
