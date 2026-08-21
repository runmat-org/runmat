mod carve;
mod types;
mod validate;

pub use carve::carve_exact_face_domain;
pub(crate) use carve::carve_validated_face_domain;
pub use types::ExactFaceTrimmedDelaunay;
pub use validate::validate_exact_face_trimmed_delaunay;
pub(crate) use validate::validate_face_trimmed_topology;

#[cfg(test)]
mod tests;
