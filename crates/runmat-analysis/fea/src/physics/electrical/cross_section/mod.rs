//! Transmission-line parameters of a board cross-section (quasi-static TEM).
//!
//! A printed line whose cross-section is small against the wavelength carries a quasi-TEM
//! wave. Its per-length capacitance and inductance follow from two 2D electrostatic solves
//! of div(ε grad φ) = 0 on the cross-section:
//!
//! - with the real dielectrics, the Maxwell capacitance matrix C (F/m);
//! - with every dielectric replaced by vacuum, C0. Copper is non-magnetic, so the inductance
//!   matrix is L = μ0 ε0 C0⁻¹ (H/m).
//!
//! From C and L: the characteristic impedance matrix Zc, and for one signal conductor
//! Z0 = sqrt(L/C), delay = sqrt(L C) per metre and εeff = C/C0. Two signal conductors form a
//! differential pair: odd- and even-mode impedances, Zdiff, Zcommon, modal εeff and delay,
//! and near- and far-end coupling coefficients.
//!
//! The cross-section comes from a stackup (layer thicknesses, Dk, copper, solder mask and
//! reference planes) plus the traces on one signal layer (width, etch angle, position). The
//! mesher builds a graded, interface-conforming triangle mesh; the operator is the same
//! dielectric potential-gradient operator as the 3D electrostatic solve, on linear triangles,
//! factored once per dielectric set by the shared sparse Cholesky on the admitted CPU plan.
//! The first version is lossless: conductor and dielectric loss (R and G per length) are a
//! separate capability.
mod artifact;
mod authored;
mod geometry;
mod lines;
mod mesh;
mod solve;
#[cfg(test)]
mod tests;

pub use artifact::{
    cross_section_digest, cross_section_geometry, cross_section_geometry_revision,
};
pub use lines::{DifferentialPairParameters, LineParameters};
pub use mesh::CrossSectionMeshSummary;
use thiserror::Error;

/// A board cross-section: its stackup, the conductors on one signal layer, the mesh controls
/// and the extent of the open (air) region.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSection {
    /// Layers from the top of the board to the bottom, as the board stackup lists them.
    pub layers: Vec<StackLayer>,
    /// Traces on one copper layer of role [`CopperRole::Signal`].
    pub conductors: Vec<Conductor>,
    pub mesh: CrossSectionMeshControls,
    /// Open-region extent; `None` sizes it from the stackup (see [`Enclosure::default_for`]).
    pub enclosure: Option<Enclosure>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StackLayer {
    pub name: String,
    /// Layer thickness in metres. Copper of a signal layer may be zero thickness (an
    /// infinitely thin strip); every other layer is positive.
    pub thickness_m: f64,
    pub kind: StackLayerKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StackLayerKind {
    /// Core or prepreg, with its relative permittivity (Dk) at the frequency of interest.
    Dielectric { relative_permittivity: f64 },
    /// Solder mask: conformal over the outer copper it covers, `thickness_m` deep over the
    /// laminate between traces and over each trace's top and sidewalls.
    SolderMask { relative_permittivity: f64 },
    Copper {
        role: CopperRole,
        /// Relative permittivity of what fills a signal layer between traces. `None` takes
        /// the solder mask (or air) on an outer layer and the dielectric directly above on an
        /// inner layer.
        fill_relative_permittivity: Option<f64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopperRole {
    /// A full-width plane held at zero potential: the return conductor.
    ReferencePlane,
    /// A routed layer carrying the conductors.
    Signal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConductorRole {
    /// A line whose parameters are reported.
    Signal,
    /// Coplanar ground copper on the signal layer, held at zero potential with the planes.
    Ground,
}

/// Which face of a trace is the etched base (the wider face, bonded to the laminate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceBase {
    Bottom,
    Top,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Conductor {
    pub name: String,
    /// Copper layer the trace is etched on.
    pub layer: String,
    pub role: ConductorRole,
    /// Lateral position of the trace centre, metres.
    pub center_m: f64,
    /// Width of the base face, metres.
    pub width_m: f64,
    /// Sidewall angle from the base plane, radians; π/2 is a rectangular trace. The far face
    /// is narrower by 2 t / tan(angle).
    pub etch_angle_rad: f64,
    /// `None` takes the laminate side: the bottom on a top outer layer, the top on a bottom
    /// outer layer and the bottom on an inner layer.
    pub base: Option<TraceBase>,
}

/// Mesh controls. Sizes are lengths in metres; `None` derives them from the smallest
/// feature (trace thickness, width, gap and dielectric height).
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSectionMeshControls {
    /// Element size at trace edges and corners, where the field is singular.
    pub edge_size_m: Option<f64>,
    /// Size ratio between neighbouring elements, in (1, 2].
    pub growth: f64,
    /// Largest element size.
    pub max_size_m: Option<f64>,
}

impl Default for CrossSectionMeshControls {
    fn default() -> Self {
        Self {
            edge_size_m: None,
            growth: DEFAULT_GROWTH,
            max_size_m: None,
        }
    }
}

/// Default element growth ratio away from trace edges.
pub const DEFAULT_GROWTH: f64 = 1.1;

/// The open region around the traces: insulating side walls `half_width_m` from the centre
/// of the traces, and an insulating lid `air_m` above (or below) an outer layer that has no
/// reference plane on that side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Enclosure {
    pub half_width_m: f64,
    pub air_m: f64,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CrossSectionError {
    #[error("invalid cross-section: {0}")]
    Invalid(String),
    #[error("cross-section operator: {0}")]
    Operator(String),
    #[error("cross-section numerical accounting failed: {0}")]
    Accounting(runmat_simulation_usage::LedgerError),
    #[error(
        "cross-section factorization needs {required_bytes} bytes, above the admitted {budget_bytes}"
    )]
    Budget {
        required_bytes: u64,
        budget_bytes: u64,
    },
    #[error("cross-section solve was cancelled")]
    Cancelled,
}

/// Solve the cross-section and return its per-length line parameters.
pub fn solve_cross_section(section: &CrossSection) -> Result<LineParameters, CrossSectionError> {
    let geometry = geometry::Geometry::build(section)?;
    let mesh = mesh::CrossSectionMesh::build(&geometry, &section.mesh)?;
    let capacitance = solve::capacitance_matrices(&geometry, &mesh)?;
    lines::LineParameters::from_capacitance(
        geometry.signal_names(),
        capacitance.dielectric,
        capacitance.vacuum,
        mesh.summary(),
        capacitance.relative_residual,
    )
}

/// A solved section and the canonical artifact of the mesh its potential solves ran on.
#[derive(Debug, Clone)]
pub struct SolvedCrossSection {
    pub parameters: LineParameters,
    pub mesh: runmat_meshing_core::SolverMeshArtifact,
}

/// Solve the section and publish its mesh as one canonical solver mesh. `section_digest`
/// (see [`cross_section_digest`]) is the geometry identity; `request` is the study's resolved
/// meshing request, whose size limits come from the admitted run budget.
pub fn solve_cross_section_with_mesh(
    section: &CrossSection,
    section_digest: runmat_meshing_core::StableDigest,
    request: runmat_meshing_core::MeshingRequest,
) -> Result<SolvedCrossSection, CrossSectionError> {
    let geometry = geometry::Geometry::build(section)?;
    let mesh = mesh::CrossSectionMesh::build(&geometry, &section.mesh)?;
    let artifact = artifact::solver_mesh_artifact(&geometry, &mesh, section_digest, request)?;
    let capacitance = solve::capacitance_matrices(&geometry, &mesh)?;
    let parameters = lines::LineParameters::from_capacitance(
        geometry.signal_names(),
        capacitance.dielectric,
        capacitance.vacuum,
        mesh.summary(),
        capacitance.relative_residual,
    )?;
    Ok(SolvedCrossSection {
        parameters,
        mesh: artifact,
    })
}
