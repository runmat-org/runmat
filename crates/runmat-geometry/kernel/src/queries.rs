//! Exact queries a kernel answers besides operations: hidden-line projection for drawings,
//! proximity for interference and clearance checks, and exchange export (AP242).
//!
//! Like [`crate::op::Kernel`], nothing here names a kernel type. Bodies are handles the kernel
//! made, entities are positions in their topology summaries, and every value is SI.

use crate::frame::Frame;
use crate::history::EntityIndex;
use crate::op::{BodyHandle, KernelError, Transform};
use serde::{Deserialize, Serialize};

/// A body placed in the query's world.
#[derive(Clone, Debug)]
pub struct Placed<'a> {
    pub body: &'a BodyHandle,
    pub transform: Transform,
}

/// What a hidden-line projection is asked for.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionRequest {
    /// The view frame in world coordinates: x to the right, y up, z towards the viewer.
    /// Projected points are `(p − origin)·x, (p − origin)·y`.
    pub frame: Frame,
    /// Also report hidden edges (flagged not visible).
    pub hidden: bool,
    /// Report smooth (tangent-continuous) edges between faces.
    pub smooth: bool,
    /// Largest chord deviation of a polyline from its curve (m).
    pub tolerance: f64,
}

/// What a projected curve draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum ProjectedSource {
    /// A model edge: the placed body's index in the request, then the edge's index in that
    /// body's topology summary.
    Edge { body: u32, edge: EntityIndex },
    /// A silhouette (outline) of a curved face.
    Silhouette { body: u32, face: EntityIndex },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectedCurve {
    pub source: ProjectedSource,
    pub visible: bool,
    pub polyline: Vec<[f64; 2]>,
}

/// How two placed bodies relate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proximity {
    /// Minimum distance between the bodies; zero when they touch or overlap.
    pub distance: f64,
    /// Volume of their common material; zero when they only touch.
    pub common_volume: f64,
}

/// An exchange format the kernel writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExchangeFormat {
    /// STEP AP242 (managed model-based 3D engineering), with product names, assembly
    /// structure, colours and face names.
    Ap242,
    /// STEP AP214, for receivers that do not read AP242.
    Ap214,
}

/// A named face of an exported body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExchangeFaceName {
    /// The body's position in its part.
    pub body: u32,
    pub face: EntityIndex,
    pub name: String,
}

/// A coloured face of an exported body (linear RGB in 0..1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExchangeFaceColor {
    pub body: u32,
    pub face: EntityIndex,
    pub color: [f64; 3],
}

/// One product: its bodies, name, colour, material and face names.
#[derive(Clone, Debug)]
pub struct ExchangePart<'a> {
    pub name: String,
    pub bodies: Vec<&'a BodyHandle>,
    pub color: Option<[f64; 3]>,
    pub material: Option<String>,
    pub face_names: Vec<ExchangeFaceName>,
    pub face_colors: Vec<ExchangeFaceColor>,
}

/// One node of the assembly structure. `part: None` is a sub-assembly: a product named
/// `definition` that holds the occurrences naming this one as their parent, or, with `same_as`,
/// another use of the sub-assembly an earlier occurrence defined (it holds nothing itself).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExchangeOccurrence {
    pub name: String,
    pub part: Option<u32>,
    /// A sub-assembly's product name (its definition), when `part` is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<String>,
    /// A sub-assembly placed again: the earlier occurrence whose product this one uses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub same_as: Option<u32>,
    /// The occurrence this one is placed in; `None`: the top assembly.
    pub parent: Option<u32>,
    /// Placement in the parent.
    pub transform: Transform,
}

/// A document to export. With no occurrences, every part is written as a top-level product.
#[derive(Clone, Debug)]
pub struct ExchangeDocument<'a> {
    pub name: String,
    pub parts: Vec<ExchangePart<'a>>,
    pub occurrences: Vec<ExchangeOccurrence>,
}

/// The exact-query seam. A kernel that cannot answer one returns
/// [`KernelError::NotSupported`]; it never approximates without saying so.
pub trait ExactQueries {
    fn project(
        &self,
        bodies: &[Placed<'_>],
        request: &ProjectionRequest,
    ) -> Result<Vec<ProjectedCurve>, KernelError>;
    fn proximity(&self, a: &Placed<'_>, b: &Placed<'_>) -> Result<Proximity, KernelError>;
    fn export(
        &self,
        document: &ExchangeDocument<'_>,
        format: ExchangeFormat,
    ) -> Result<Vec<u8>, KernelError>;
}
