use serde::{Deserialize, Serialize};

pub type Point3 = [f64; 3];
pub type Triangle3 = [Point3; 3];
pub type Tetrahedron3 = [Point3; 4];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RayTriangleHit {
    pub distance: f64,
    pub barycentric_u: f64,
    pub barycentric_v: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PointInClosedSurface {
    Inside,
    Outside,
    OnBoundary,
}
