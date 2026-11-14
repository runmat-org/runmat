//! Check-time estimate of a tetrahedral solver mesh's size, from the preview measure of its
//! geometry and the mesh settings, before any meshing.
//!
//! The local element size starts at each surface patch as the smallest of the target size, the
//! chordal size its curvature allows (`sqrt(8 R d)`, the sagitta bound, for radius `R` and deviation `d`) and its
//! wall thickness or facing gap divided by the layers proximity sizing resolves across it.
//! Inside the material it grows with the grading ratio up to that interior size. Integrating
//! one element per `(c s)^3 / (6 sqrt 2)` over the volume behind each patch gives the
//! tetrahedra, and one triangle per `sqrt(3)/4 (c s)^2` over the surface gives the boundary
//! facets; the vertices follow from both.
//!
//! The constants are fitted to meshes the production mesher made of two fixtures: a bracket of
//! 3 mm plates with 1.1 mm bores (seven sizings from 10 mm to 0.7 mm, deviations 0.1 to
//! 0.01 mm) and a 100 x 20 x 10 mm bar (3 mm and 1.5 mm). There the estimate is within -8% and
//! +11% of the tetrahedra and within -11% and +12% of the quadratic nodes (see the tests). A
//! coarse mesh of a chunky part is overestimated: the bar at 10 mm is 3.5 times its 1,138
//! tetrahedra. Size controls on named regions are not modelled.
use runmat_geometry_core::{PreviewMeasure, PreviewSurfacePatch};

use crate::{ElementOrder, MeshingRequestSettings};

/// The mesher's mean tetrahedron edge as a fraction of the local size bound.
const VOLUME_EDGE_FRACTION: f64 = 0.64;
/// The mesher's mean boundary-triangle edge as a fraction of the local size bound.
const SURFACE_EDGE_FRACTION: f64 = 0.71;
/// The chordal size on a face of radius `R` at deviation `d` is `sqrt(CHORDAL * R * d)`.
const CHORDAL: f64 = 8.0;
/// Size growth per unit distance per unit of grading above one.
const GRADING_GROWTH: f64 = 0.25;
/// Size bound across a wall or gap: its width over this many times the requested layers.
const LAYER_SPAN: f64 = 1.25;
/// Layers across a wall or gap when none are requested (two across gaps between edges).
const DEFAULT_LAYERS: u32 = 2;
/// Tetrahedra per interior vertex and vertices per boundary facet (fitted, within 0.3%).
const TETRAHEDRA_PER_INTERIOR_VERTEX: f64 = 6.4;
const VERTICES_PER_BOUNDARY_FACET: f64 = 0.26;

/// Counts of a tetrahedral solid mesh. With the element order they fix the node count and the
/// node adjacency through Euler's relation for one solid ball (V - E + F - T = 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolidMeshCounts {
    pub vertices: u64,
    pub tetrahedra: u64,
    pub boundary_facets: u64,
    pub order: ElementOrder,
}

impl SolidMeshCounts {
    /// Unique tetrahedron edges.
    pub fn edges(&self) -> u64 {
        (self.vertices + self.tetrahedra + self.boundary_facets / 2).saturating_sub(1)
    }

    /// Unique triangular faces, interior and boundary.
    pub fn faces(&self) -> u64 {
        2 * self.tetrahedra + self.boundary_facets / 2
    }

    /// Mesh nodes: the vertices, plus one midside node per edge for quadratic tetrahedra.
    pub fn nodes(&self) -> u64 {
        match self.order {
            ElementOrder::Tet4 => self.vertices,
            ElementOrder::Tet10 => self.vertices + self.edges(),
        }
    }

    /// Nodes of one element.
    pub fn nodes_per_element(&self) -> u64 {
        match self.order {
            ElementOrder::Tet4 => 4,
            ElementOrder::Tet10 => 10,
        }
    }

    /// Unordered pairs of distinct nodes that share an element: the off-diagonal blocks of one
    /// triangle of the stiffness matrix. A quadratic tetrahedron's 45 pairs are 3 per edge (its
    /// corners and midside), 6 per face beyond its edges, and 3 between opposite midsides.
    pub fn node_pairs(&self) -> u64 {
        match self.order {
            ElementOrder::Tet4 => self.edges(),
            ElementOrder::Tet10 => 3 * self.edges() + 6 * self.faces() + 3 * self.tetrahedra,
        }
    }
}

/// The estimated solid mesh of a geometry under one set of mesh settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshSizeEstimate {
    pub counts: SolidMeshCounts,
}

/// Estimates the solid mesh the settings make of the measured geometry. `None` when the
/// settings carry no positive target size or deviation.
pub fn estimate_solid_mesh_size(
    measure: &PreviewMeasure,
    settings: &MeshingRequestSettings,
) -> Option<MeshSizeEstimate> {
    let target = settings.target_edge_length_m.canonical_value();
    let deviation = settings.maximum_chordal_deviation_m.canonical_value();
    if !(target.is_finite() && target > 0.0 && deviation.is_finite() && deviation > 0.0) {
        return None;
    }
    let layers = LAYER_SPAN * f64::from(settings.cells_across_gap.unwrap_or(DEFAULT_LAYERS).max(1));
    let growth = GRADING_GROWTH * (settings.maximum_grading_ratio - 1.0).max(0.05);
    // Each patch owns the material behind it to half its wall; scale the columns so they
    // fill exactly the enclosed volume (corners and open rays make the raw sum differ).
    let fallback_depth = measure.volume_m3 / measure.area_m2;
    let depth = |patch: &PreviewSurfacePatch| patch.thickness_m.map_or(fallback_depth, |t| t / 2.0);
    let raw: f64 = measure.patches.iter().map(|p| p.area_m2 * depth(p)).sum();
    if !(raw.is_finite() && raw > 0.0) {
        return None;
    }
    let fill = measure.volume_m3 / raw;
    let tetrahedron = VOLUME_EDGE_FRACTION.powi(3) / (6.0 * std::f64::consts::SQRT_2);
    let triangle = 3.0_f64.sqrt() / 4.0 * SURFACE_EDGE_FRACTION.powi(2);
    let (mut tetrahedra, mut facets) = (0.0, 0.0);
    for patch in &measure.patches {
        let wall = [patch.thickness_m, patch.gap_m]
            .into_iter()
            .flatten()
            .fold(f64::INFINITY, f64::min);
        let interior = target.min(wall / layers);
        let surface = patch.radius_m.map_or(interior, |radius| {
            interior.min((CHORDAL * radius * deviation).sqrt())
        });
        let column = depth(patch) * fill;
        tetrahedra += patch.area_m2 * graded_inverse_cube(surface, interior, growth, column);
        facets += patch.area_m2 / surface.powi(2);
    }
    let tetrahedra = (tetrahedra / tetrahedron).ceil();
    let facets = (facets / triangle).ceil();
    let vertices =
        (tetrahedra / TETRAHEDRA_PER_INTERIOR_VERTEX + facets * VERTICES_PER_BOUNDARY_FACET).ceil();
    let count = |value: f64| (value.is_finite() && value < 9.0e15).then_some(value as u64);
    Some(MeshSizeEstimate {
        counts: SolidMeshCounts {
            vertices: count(vertices)?.max(4),
            tetrahedra: count(tetrahedra)?.max(1),
            boundary_facets: count(facets)?.max(4),
            order: settings.element_order,
        },
    })
}

/// The integral of `s(x)^-3` over `0..depth`, where the size `s` grows from `surface` by
/// `growth` per unit distance until it reaches `interior`.
fn graded_inverse_cube(surface: f64, interior: f64, growth: f64, depth: f64) -> f64 {
    let graded = ((interior - surface) / growth).max(0.0);
    let reach = graded.min(depth);
    let end = surface + growth * reach;
    let ramp = (surface.powi(-2) - end.powi(-2)) / (2.0 * growth);
    ramp + (depth - reach).max(0.0) / interior.powi(3)
}

#[cfg(test)]
#[path = "estimate_tests.rs"]
mod tests;
