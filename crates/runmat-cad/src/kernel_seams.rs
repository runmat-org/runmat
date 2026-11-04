//! The drawing and assembly seams on an exact kernel.
//!
//! [`HiddenLineRemover`] and [`InterferenceKernel`] are this crate's seams; an exact kernel
//! answers them through the kernel-neutral
//! [`ExactQueries`](runmat_geometry_kernel::queries::ExactQueries) (K1 implements it on OCCT).
//! What a view or an occurrence shows comes from the caller: [`ViewModels`] places a model's
//! bodies and names their entities, [`PlacedBodies`] gives an occurrence's body. Nothing here
//! names a kernel type.

use crate::assembly::interference::{
    BoxKernel, CheckMode, InterferenceKernel, PairResult, PlacedShape,
};
use crate::drawing::{Display, HiddenLineRemover, ModelRef, ProjectedEdge, View, ViewGeometry};
use crate::model::datum::Frame;
use crate::refs::name::EntityName;
use crate::regen::kernel::{BodyHandle, KernelError, Transform};
use runmat_geometry_kernel::queries::{ExactQueries, Placed, ProjectedSource, ProjectionRequest};

/// One body a view shows, placed in model coordinates, with its entities' names.
#[derive(Clone, Debug)]
pub struct ViewBody {
    pub handle: BodyHandle,
    pub transform: Transform,
    /// Per summary edge: the names of its two faces.
    pub edge_names: Vec<Vec<EntityName>>,
    /// Per summary face: its name.
    pub face_names: Vec<EntityName>,
}

/// What the views of a drawing show.
pub trait ViewModels {
    /// The view's projection frame in model coordinates: X right, Y up, Z towards the viewer.
    fn view_frame(&self, view: &View) -> Result<Frame, String>;
    /// The bodies of the model a view shows (occurrences placed, excluded ones left out).
    fn bodies(&self, view: &View, model: &ModelRef) -> Result<Vec<ViewBody>, String>;
}

/// Hidden-line removal through an exact kernel.
pub struct KernelHiddenLines<'a> {
    pub queries: &'a dyn ExactQueries,
    pub models: &'a dyn ViewModels,
    /// Largest chord deviation of a projected polyline (m).
    pub tolerance: f64,
}

impl HiddenLineRemover for KernelHiddenLines<'_> {
    fn project(&self, view: &View, model_key: &str) -> Result<ViewGeometry, KernelError> {
        let failed = |message: String| KernelError::Failed { message };
        let frame = self.models.view_frame(view).map_err(failed)?;
        let bodies = self.models.bodies(view, &view.model).map_err(failed)?;
        let placed: Vec<Placed<'_>> = bodies
            .iter()
            .map(|b| Placed {
                body: &b.handle,
                transform: b.transform,
            })
            .collect();
        let request = ProjectionRequest {
            frame,
            hidden: view.display == Display::HiddenLines,
            smooth: false,
            tolerance: self.tolerance,
        };
        let curves = self.queries.project(&placed, &request)?;
        let mut edges = Vec::with_capacity(curves.len());
        for c in curves {
            let entity = match c.source {
                ProjectedSource::Edge { body, edge } => bodies
                    .get(body as usize)
                    .and_then(|b| b.edge_names.get(edge as usize))
                    .cloned(),
                ProjectedSource::Silhouette { body, face } => bodies
                    .get(body as usize)
                    .and_then(|b| b.face_names.get(face as usize))
                    .map(|n| vec![n.clone()]),
            }
            .ok_or_else(|| failed("a projected curve names no model entity".into()))?;
            edges.push(ProjectedEdge {
                entity,
                visible: c.visible,
                polyline: c.polyline,
            });
        }
        Ok(ViewGeometry {
            edges,
            key: model_key.to_string(),
        })
    }
}

/// The exact body of a placed shape, when the caller has one.
pub trait PlacedBodies {
    fn body(&self, shape: &PlacedShape) -> Option<BodyHandle>;
}

/// Interference and clearance through an exact kernel's proximity query. A pair without exact
/// bodies is answered from boxes and reported as not exact.
pub struct KernelInterference<'a> {
    pub queries: &'a dyn ExactQueries,
    pub bodies: &'a dyn PlacedBodies,
}

impl InterferenceKernel for KernelInterference<'_> {
    fn check(
        &self,
        a: &PlacedShape,
        b: &PlacedShape,
        mode: CheckMode,
    ) -> Result<(PairResult, bool), KernelError> {
        let (Some(ha), Some(hb)) = (self.bodies.body(a), self.bodies.body(b)) else {
            return BoxKernel.check(a, b, mode);
        };
        let p = self.queries.proximity(
            &Placed {
                body: &ha,
                transform: a.world,
            },
            &Placed {
                body: &hb,
                transform: b.world,
            },
        )?;
        let result = if p.common_volume > 0.0 {
            PairResult::Interfering {
                volume: p.common_volume,
            }
        } else if p.distance <= 0.0 {
            PairResult::Touching
        } else {
            PairResult::Clear {
                distance: p.distance,
            }
        };
        Ok((result, true))
    }
}
