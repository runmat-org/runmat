//! The static solid plan's memory from mesh counts alone, for a check-time estimate before any
//! mesh exists.
//!
//! The terms are the elastic rigidity multigrid plan's (the path a constrained budget selects):
//! the topology, the stiffness, mass and damping patterns, the assembly contributions, the
//! multigrid hierarchy, the solve vectors, the rigidity proof inventory and the recovered
//! fields, each by the formula the mesh-based planner uses. Only the sparse pattern comes from
//! counts: the node adjacency of a conforming tetrahedral mesh follows from its vertices,
//! tetrahedra and boundary facets (Euler's relation). The factorization's fill is not
//! estimated here; whether the proving factorization fits stays an execution-time decision.
use runmat_meshing_core::SolidMeshCounts;

use super::{
    ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT, FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
    STRUCTURAL_ARTIFACT_OVERHEAD_BYTES, STRUCTURAL_FIXED_WORKER_MEMORY_BYTES,
    STRUCTURAL_FIXED_WORKER_SCRATCH_BYTES, STRUCTURAL_RECOVERY_PEAK_VALUES_PER_ELEMENT,
    STRUCTURAL_RECOVERY_VALUES_PER_NODE, STRUCTURAL_SOLVE_VECTOR_COUNT,
};
use crate::planning::resource::ResourcePlanningError;

/// Translational DOFs per solid node.
const DOFS_PER_NODE: u64 = 3;
/// Static plans reserve the stiffness, mass and damping patterns of one node graph. Stiffness
/// and damping couple every DOF pair of adjacent nodes; mass couples each axis to itself only.
const OPERATOR_PATTERNS: u64 = 3;
/// Bytes of one assembly contribution (row, column, value).
const CONTRIBUTION_BYTES: u64 = 24;
/// Resident bytes of the prepared solver mesh the run holds: per node its coordinates, stable
/// identity and provenance; per element its node list and identity; per boundary facet its
/// nodes, identity and role. Measured against encoded production meshes.
const MESH_BYTES_PER_NODE: u64 = 96;
const MESH_BYTES_PER_ELEMENT_BASE: u64 = 64;
const MESH_BYTES_PER_BOUNDARY_FACET: u64 = 96;

/// A count-based static solid estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CountedStaticSolidEstimate {
    pub nodes: u64,
    pub elements: u64,
    /// Nonzeros of the full (both triangles) stiffness pattern.
    pub stiffness_nonzeros: u64,
    /// Resident bytes of the prepared mesh the solve reads.
    pub mesh_bytes: u64,
    /// Worker resident memory of the multigrid plan, with the prepared mesh.
    pub memory_bytes: u64,
    /// Worker scratch of the multigrid plan.
    pub scratch_bytes: u64,
}

/// Bounds a linear static solid's multigrid plan from mesh counts, serial execution.
pub fn estimate_static_solid_from_counts(
    counts: SolidMeshCounts,
) -> Result<CountedStaticSolidEstimate, ResourcePlanningError> {
    let overflow = ResourcePlanningError::Overflow;
    let mul = |a: u64, b: u64, label| a.checked_mul(b).ok_or(overflow(label));
    let add = |a: u64, b: u64, label| a.checked_add(b).ok_or(overflow(label));
    let nodes = counts.nodes();
    let elements = counts.tetrahedra;
    let width = counts.nodes_per_element();
    let dofs = mul(nodes, DOFS_PER_NODE, "counted DOFs")?;
    let pairs = counts.node_pairs();
    let node_nonzeros = add(
        nodes,
        mul(pairs, 2, "counted node pairs")?,
        "counted node graph",
    )?;
    let stiffness_nonzeros = mul(
        node_nonzeros,
        DOFS_PER_NODE * DOFS_PER_NODE,
        "counted nonzeros",
    )?;
    let as_usize = |value: u64| usize::try_from(value).map_err(|_| overflow("counted size"));
    let boundary_nodes = match counts.order {
        runmat_meshing_core::ElementOrder::Tet4 => 3,
        runmat_meshing_core::ElementOrder::Tet10 => 6,
    };
    let topology = crate::planning::resource::facts::estimate_topology_resident_bytes(
        as_usize(nodes)?,
        as_usize(elements)?,
        as_usize(counts.boundary_facets)?,
        as_usize(mul(elements, width, "counted element nodes")?)?,
        as_usize(mul(
            counts.boundary_facets,
            boundary_nodes,
            "counted boundary nodes",
        )?)?,
        as_usize(mul(pairs, 2, "counted neighbor references")?)?,
    )?
    .0;
    let mass_nonzeros = mul(node_nonzeros, DOFS_PER_NODE, "counted mass nonzeros")?;
    let operator_nonzeros = add(
        mul(stiffness_nonzeros, 2, "counted stiffness and damping")?,
        mass_nonzeros,
        "counted operator nonzeros",
    )?;
    let csr = add(
        mul(
            add(dofs, 1, "counted rows")?,
            8 * OPERATOR_PATTERNS,
            "counted row offsets",
        )?,
        mul(operator_nonzeros, 16, "counted CSR entries")?,
        "counted CSR bytes",
    )?;
    let persistent = add(topology, csr, "counted persistent memory")?;
    let assembly = mul(
        stiffness_nonzeros,
        CONTRIBUTION_BYTES,
        "counted contributions",
    )?;
    let vectors = mul(
        dofs,
        STRUCTURAL_SOLVE_VECTOR_COUNT * 8,
        "counted solve vectors",
    )?;
    let multigrid = runmat_sparse_solvers::estimate::amg_bytes(
        as_usize(dofs)?,
        as_usize(stiffness_nonzeros)?,
        3,
    );
    let proof = crate::algebra::elastic_rigidity::ElasticRigidityCapture::resident_bytes_for(
        nodes,
        std::iter::repeat_n(width as usize, as_usize(elements)?),
    );
    let solve = add(
        add(vectors, multigrid, "counted solve")?,
        proof,
        "counted solve",
    )?;
    let recovery_values = add(
        mul(
            nodes,
            STRUCTURAL_RECOVERY_VALUES_PER_NODE as u64,
            "counted nodal recovery",
        )?,
        mul(
            elements,
            STRUCTURAL_RECOVERY_PEAK_VALUES_PER_ELEMENT as u64,
            "counted recovery",
        )?,
        "counted recovery values",
    )?;
    let recovery = add(
        mul(recovery_values, 8, "counted recovery bytes")?,
        STRUCTURAL_ARTIFACT_OVERHEAD_BYTES,
        "counted recovery resident",
    )?;
    let mesh_bytes = [
        mul(nodes, MESH_BYTES_PER_NODE, "counted mesh nodes")?,
        mul(
            elements,
            MESH_BYTES_PER_ELEMENT_BASE + 8 * width,
            "counted mesh elements",
        )?,
        mul(
            counts.boundary_facets,
            MESH_BYTES_PER_BOUNDARY_FACET,
            "counted mesh facets",
        )?,
    ]
    .into_iter()
    .try_fold(0, |sum, bytes| add(sum, bytes, "counted mesh bytes"))?;
    // The resource plan itself is an input of the solve: its decoded form is resident and
    // its transport closure is retained, as for every operator plan.
    let plan_input = crate::planning::resource::estimate::implicit_resource_plan_input_facts()?;
    let numerical = crate::planning::resource::estimate::operator_peak_resident_bytes(
        persistent,
        mesh_bytes,
        plan_input.decoded_resident_bytes_admission_budget.0,
        assembly,
        solve,
        recovery,
    )?;
    let memory_bytes = [
        numerical,
        plan_input.transport_closure_bytes.0,
        STRUCTURAL_FIXED_WORKER_MEMORY_BYTES,
    ]
    .into_iter()
    .try_fold(0, |sum, bytes| add(sum, bytes, "counted memory"))?;
    let assembly_scratch = add(
        FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER,
        mul(
            elements,
            ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT,
            "counted assembly scratch",
        )?,
        "counted assembly scratch",
    )?;
    let scratch_bytes = add(
        assembly_scratch.max(recovery),
        STRUCTURAL_FIXED_WORKER_SCRATCH_BYTES,
        "counted scratch",
    )?;
    Ok(CountedStaticSolidEstimate {
        nodes,
        elements,
        stiffness_nonzeros,
        mesh_bytes,
        memory_bytes,
        scratch_bytes,
    })
}
