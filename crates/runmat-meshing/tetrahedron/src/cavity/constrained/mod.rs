#![cfg_attr(test, allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{
        point_in_closed_triangle_surface, tetrahedron_centroid, Point3, PointInClosedSurface,
    },
    quality::tolerance::MeshingTolerance,
};

mod boundary_completion;
mod boundary_nodes;
mod boundary_operations;
mod boundary_splits;
mod cap_connectors;
mod caps;
mod component_steiner;
mod connectivity;
#[cfg(test)]
mod diagnostic_metrics;
#[cfg(test)]
mod diagnostics;
mod exact_cover;
mod geometry;
mod missing_faces;
mod pressure;
mod refill_candidates;
mod refill_evaluation;
mod refill_faces;
mod refill_tetrahedra;
mod retriangulate;
mod selection;
mod solid_empty;
mod topology;
mod types;
mod validation;

#[cfg(test)]
use boundary_completion::*;
#[cfg(test)]
use boundary_nodes::candidate_respects_protected_boundary_distance;
use boundary_nodes::{
    boundary_node_coordinates, cavity_boundary_node_centroid, cavity_boundary_node_ids,
    cavity_boundary_triangles, next_cavity_node_id,
};
use boundary_operations::*;
pub use boundary_operations::{
    recover_constrained_cavity_source_edge_by_split_refill, split_constrained_cavity_boundary_edge,
    split_constrained_cavity_boundary_edge_patch_at_centroid,
    split_constrained_cavity_boundary_face, split_constrained_cavity_boundary_face_at_barycentric,
    split_constrained_cavity_boundary_face_at_centroid, split_constrained_cavity_boundary_faces,
    split_constrained_cavity_boundary_faces_at_centroids,
    split_constrained_cavity_boundary_patch_at_centroids, split_constrained_cavity_source_edge,
};
use boundary_splits::*;
#[cfg(test)]
use cap_connectors::*;
#[cfg(test)]
use caps::*;
pub use caps::{
    generate_constrained_cavity_boundary_cap_nodes, generate_constrained_cavity_patch_steiner_nodes,
};
pub use component_steiner::generate_constrained_cavity_component_steiner_nodes;
use connectivity::*;
#[cfg(test)]
use diagnostic_metrics::*;
#[cfg(test)]
use diagnostics::*;
use exact_cover::*;
pub use exact_cover::{
    selected_exact_cover_face_count_blockers, selected_exact_cover_saturated_component,
};
#[cfg(test)]
use geometry::*;
#[cfg(test)]
use missing_faces::*;
pub use pressure::constrained_cavity_refill_pressure_boundary_faces;
#[cfg(test)]
use refill_candidates::{
    boundary_node_refill_rejection_reason, boundary_node_refill_validation_reason,
    centroid_interior_refill_candidate, multi_interior_exact_cover_failure_reason,
    multi_interior_node_refill_candidate, two_interior_node_refill_candidate,
};
pub use refill_evaluation::{
    evaluate_constrained_cavity_refill_candidates, generate_constrained_cavity_refill_candidates,
};
#[cfg(test)]
use refill_faces::*;
use refill_tetrahedra::{
    boundary_faces_from_refill_tetrahedra, raw_refill_tetrahedron_with_rejection_reason,
    refill_from_tetrahedra,
};
pub use refill_tetrahedra::{
    flip_refill_tetrahedra_across_shared_face, flip_refill_tetrahedra_around_shared_edge,
    split_refill_tetrahedra_across_shared_face_at_barycentric,
};
#[cfg(test)]
use refill_tetrahedra::{
    improve_refill_with_local_flips_with_diagnostics, refill_is_better,
    star_refill_candidate_with_rejection_reason,
};
pub use retriangulate::retriangulate_constrained_cavity_from_nodes;
use selection::*;
pub use selection::{
    constrained_cavity_expanded_across_boundary_edge_star_excluding_nodes,
    constrained_cavity_expanded_across_boundary_face,
    constrained_cavity_expanded_across_boundary_faces,
    constrained_cavity_expanded_across_boundary_faces_or_recovered_edge_star,
    constrained_cavity_expanded_across_first_valid_boundary_face,
    constrained_cavity_from_refill_tetrahedron_component,
    constrained_cavity_from_selected_tetrahedra,
    constrained_cavity_from_selected_tetrahedra_with_anchor_trim,
    constrained_cavity_recovered_boundary_edge_star_excluding_nodes,
    constrained_cavity_recovered_boundary_edge_star_queue_excluding_nodes,
};
pub use solid_empty::{
    constrained_cavity_classified_solid_empty_boundary_faces,
    constrained_cavity_solid_empty_boundary_faces,
    recover_constrained_cavity_solid_empty_boundaries,
};
use topology::*;
pub use types::*;
use validation::*;
pub use validation::{
    validate_constrained_cavity, validate_constrained_cavity_boundary_preserved,
    validate_constrained_cavity_refill_volume,
};

#[cfg(test)]
mod tests;
