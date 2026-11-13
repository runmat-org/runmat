//! Global problem description held by the coordinator before partitioning.

use crate::error::{DistributedError, DistributedResult};

/// Unknown field carried by every node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldKind {
    /// One unknown per node; the coarse space holds one constant per subdomain.
    Scalar,
    /// Three displacement components per node; the coarse space holds the six rigid-body
    /// modes of each subdomain.
    Elastic,
}

impl FieldKind {
    pub const fn dofs_per_node(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Elastic => 3,
        }
    }

    pub const fn coarse_modes(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Elastic => 6,
        }
    }

    pub(crate) const fn code(self) -> u32 {
        match self {
            Self::Scalar => 1,
            Self::Elastic => 3,
        }
    }

    pub(crate) fn from_code(code: u32) -> DistributedResult<Self> {
        match code {
            1 => Ok(Self::Scalar),
            3 => Ok(Self::Elastic),
            _ => Err(DistributedError::InvalidProblem(format!(
                "unknown field code {code}"
            ))),
        }
    }
}

/// Solver mesh with one uniform element type. The first four nodes of every element are its
/// tetrahedral corners; higher-order nodes follow.
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionMesh {
    pub coordinates: Vec<[f64; 3]>,
    pub nodes_per_element: usize,
    pub element_nodes: Vec<u32>,
}

impl PartitionMesh {
    pub fn node_count(&self) -> usize {
        self.coordinates.len()
    }

    pub fn element_count(&self) -> usize {
        if self.nodes_per_element == 0 {
            0
        } else {
            self.element_nodes.len() / self.nodes_per_element
        }
    }

    pub fn element(&self, element: usize) -> &[u32] {
        let start = element * self.nodes_per_element;
        &self.element_nodes[start..start + self.nodes_per_element]
    }

    pub(crate) fn validate(&self) -> DistributedResult<()> {
        if self.nodes_per_element < 4
            || !self
                .element_nodes
                .len()
                .is_multiple_of(self.nodes_per_element)
            || self.element_count() == 0
        {
            return Err(DistributedError::InvalidProblem(
                "mesh needs a uniform tetrahedral element connectivity".into(),
            ));
        }
        let nodes = self.node_count();
        if self
            .element_nodes
            .iter()
            .any(|&node| node as usize >= nodes)
        {
            return Err(DistributedError::InvalidProblem(
                "element references a node outside the mesh".into(),
            ));
        }
        if self
            .coordinates
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(DistributedError::InvalidProblem(
                "mesh coordinates must be finite".into(),
            ));
        }
        Ok(())
    }
}

/// Global degree of freedom held at a prescribed value (`node * dofs_per_node + component`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrescribedDof {
    pub dof: u64,
    pub value: f64,
}

/// Assembled external nodal load on one global degree of freedom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLoad {
    pub dof: u64,
    pub value: f64,
}

/// Element operator of one physics. Implementations come from the physics owners; this crate
/// only places and combines their element contributions.
pub trait ElementKernel: Send + Sync {
    fn field(&self) -> FieldKind;
    fn nodes_per_element(&self) -> usize;
    /// Write the dense row-major element operator, `(n * d)` square, ordered node-major.
    fn element_matrix(
        &self,
        material: u32,
        coordinates: &[[f64; 3]],
        matrix: &mut [f64],
    ) -> Result<(), crate::ElementKernelError>;
    /// Width of one element's recovered result (stress, heat flux, ...).
    fn result_width(&self) -> usize;
    /// Recover the element result from its nodal solution, ordered node-major.
    fn element_result(
        &self,
        material: u32,
        coordinates: &[[f64; 3]],
        solution: &[f64],
        result: &mut [f64],
    ) -> Result<(), crate::ElementKernelError>;
}

/// Complete global problem before partitioning.
#[derive(Clone, Debug)]
pub struct PartitionedProblem {
    pub mesh: PartitionMesh,
    pub element_materials: Vec<u32>,
    pub field: FieldKind,
    pub prescribed: Vec<PrescribedDof>,
    pub loads: Vec<PointLoad>,
}

impl PartitionedProblem {
    pub fn dof_count(&self) -> usize {
        self.mesh.node_count() * self.field.dofs_per_node()
    }

    pub fn validate(&self) -> DistributedResult<()> {
        self.mesh.validate()?;
        if self.element_materials.len() != self.mesh.element_count() {
            return Err(DistributedError::InvalidProblem(
                "every element needs exactly one material index".into(),
            ));
        }
        let dofs = self.dof_count() as u64;
        if self
            .prescribed
            .iter()
            .any(|entry| entry.dof >= dofs || !entry.value.is_finite())
            || self
                .loads
                .iter()
                .any(|entry| entry.dof >= dofs || !entry.value.is_finite())
        {
            return Err(DistributedError::InvalidProblem(
                "prescribed values and loads need finite values on existing dofs".into(),
            ));
        }
        Ok(())
    }
}
