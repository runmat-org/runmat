mod adaptive_metric;
mod adaptive_planar;
mod adaptive_spatial;
mod circumsphere;
mod coordinate_bounds;
mod distance;
mod exact_dyadic;
mod lifting;
mod radial;
mod ray;
mod simplex;
mod solve;
mod surface;
mod types;
mod vector;

pub use adaptive_metric::{diametral_sphere_position, diametral_sphere_position_constructed};
pub use adaptive_planar::{
    incircle2d, incircle2d_symbolic, orient2d, orient2d_symbolic, PlanarPredicateError,
    PlanarPredicatePoint, PredicateSign,
};
pub use adaptive_spatial::{
    compose_dyadic_point, insphere3d, insphere3d_constructed, insphere3d_symbolic, orient3d,
    orient3d_constructed, orient3d_symbolic, points_collinear_constructed,
    projected_orient2d_constructed, triangle_normal_alignment_constructed,
    triangle_projection_axes, triangle_projection_axes_constructed, Point3Construction,
    SpatialPredicateError, SpatialPredicatePoint,
};
pub use circumsphere::{tetrahedron_circumsphere, tetrahedron_circumsphere_contains_point};
pub use coordinate_bounds::constructed_point_bounds;
pub use distance::{closest_point_on_triangle, point_triangle_distance};
pub use exact_dyadic::ExactDyadic;
pub use lifting::tetrahedral_lifting_change;
pub use radial::compare_radial_distances_constructed;
pub use ray::ray_triangle_intersection;
pub use simplex::{
    orient_tetrahedron_node_ids, tetrahedron_centroid, tetrahedron_edge_aspect_ratio,
    tetrahedron_scaled_jacobian, tetrahedron_scaled_jacobian_with_owner, tetrahedron_signed_volume,
    tetrahedron_volume, triangle_area, triangle_centroid,
};
pub use solve::solve_3x3;
pub use surface::{point_in_closed_triangle_surface, ray_has_odd_surface_intersections};
pub use types::{Point3, PointInClosedSurface, RayTriangleHit, Tetrahedron3, Triangle3};
pub use vector::{add, cross, distance, distance_squared, dot, norm, norm_squared, scale, sub};

#[cfg(test)]
mod tests;
