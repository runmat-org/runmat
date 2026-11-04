//! The drawing and assembly seams on an exact kernel.
//!
//! [`HiddenLineRemover`] and [`InterferenceKernel`] are this crate's seams; an exact kernel
//! answers them through the kernel-neutral
//! [`ExactQueries`](runmat_geometry_kernel::queries::ExactQueries) (K1 implements it on OCCT).
//! What a view or an occurrence shows comes from the caller: [`ViewModels`] places a model's
//! bodies and names their entities, [`PlacedBodies`] gives an occurrence's body. Hole tables
//! ([`HoleGeometry`]) read the regenerated model the caller gives through
//! [`RegeneratedModels`]. Nothing here names a kernel type.

use crate::assembly::interference::{
    BoxKernel, CheckMode, InterferenceKernel, PairResult, PlacedShape,
};
use crate::drawing::hole_table::{HoleGeometry, HoleInstance};
use crate::drawing::{Display, HiddenLineRemover, ModelRef, ProjectedEdge, View, ViewGeometry};
use crate::model::datum::Frame;
use crate::model::items::Cardinality;
use crate::model::PartDocument;
use crate::refs::name::{EntityName, TopoKind};
use crate::refs::reference::Reference;
use crate::refs::resolve::{ResolutionPolicy, ResolveOutcome, Resolver};
use crate::regen::engine::RegenResult;
use crate::regen::kernel::{BodyHandle, KernelError, Transform};
use crate::regen::records::FeatureRecord;
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

/// A part as regenerated for a drawing: its document and the regeneration of the
/// configuration and variant the model reference names.
pub struct RegeneratedModel<'a> {
    pub doc: &'a PartDocument,
    pub result: &'a RegenResult,
}

/// The regenerated parts a drawing documents.
pub trait RegeneratedModels {
    fn model(&self, model: &ModelRef) -> Result<RegeneratedModel<'_>, String>;
}

/// Hole tables from a regenerated part: hole instances from its hole features' records (in
/// feature order), the view frame from [`ViewModels`], and datum origins from the part's
/// topology: a vertex, or a hole's centre named by its wall or rim.
pub struct RegenHoleGeometry<'a> {
    pub models: &'a dyn RegeneratedModels,
    pub views: &'a dyn ViewModels,
}

/// Distance tolerance for matching a hole's wall or rim to its instance (m).
const HOLE_MATCH: f64 = 1e-7;

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}
fn unit3(a: [f64; 3]) -> [f64; 3] {
    let l = len3(a);
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        a
    }
}

fn holes_of(m: &RegeneratedModel<'_>) -> Vec<HoleInstance> {
    let mut out = Vec::new();
    for f in &m.doc.features {
        if let Some(FeatureRecord::Holes { holes }) = m.result.state.records.get(&f.id) {
            out.extend(holes.iter().cloned());
        }
    }
    out
}

/// The hole whose axis is the line through `point` along `direction`.
fn hole_on_axis<'h>(
    holes: &'h [HoleInstance],
    point: [f64; 3],
    direction: [f64; 3],
) -> Option<&'h HoleInstance> {
    let d = unit3(direction);
    holes.iter().find(|h| {
        let a = unit3(h.axis);
        let parallel = len3([
            a[1] * d[2] - a[2] * d[1],
            a[2] * d[0] - a[0] * d[2],
            a[0] * d[1] - a[1] * d[0],
        ]) < 1e-9;
        let v = sub3(point, h.position);
        let off = sub3(v, [a[0] * dot3(v, a), a[1] * dot3(v, a), a[2] * dot3(v, a)]);
        parallel && len3(off) < HOLE_MATCH
    })
}

impl HoleGeometry for RegenHoleGeometry<'_> {
    fn holes(&self, model: &ModelRef) -> Result<Vec<HoleInstance>, String> {
        Ok(holes_of(&self.models.model(model)?))
    }

    fn view_frame(&self, view: &View) -> Result<Frame, String> {
        self.views.view_frame(view)
    }

    fn point(&self, model: &ModelRef, reference: &Reference) -> Result<[f64; 3], String> {
        let m = self.models.model(model)?;
        let st = &m.result.state;
        let resolver = Resolver {
            file: &m.doc.path,
            bodies: st.views(),
            features: &st.table,
            consumer: None,
            burials: &st.burials,
            publishes: &m.doc.publishes,
            lock: None,
            catalog: None,
            policy: ResolutionPolicy::default(),
            names: Some(&st.names),
        };
        let entities = match resolver.resolve(reference, None, Cardinality::ExactlyOne) {
            ResolveOutcome::Resolved(r) => r.entities(),
            ResolveOutcome::Waiting(w) => return Err(format!("is not resolved yet ({w:?})")),
            ResolveOutcome::Failed(p) => return Err(format!("does not resolve ({p:?})")),
        };
        let Some(h) = entities.iter().next() else {
            return Err("names no entity of the model".into());
        };
        let body = st
            .bodies
            .get(h.body as usize)
            .ok_or_else(|| "names a body the model does not have".to_string())?;
        let t = &body.topology;
        let holes = holes_of(&m);
        match h.kind {
            TopoKind::Vertex => t
                .vertices
                .get(h.index as usize)
                .map(|v| v.point)
                .ok_or_else(|| "names a vertex the model does not have".into()),
            TopoKind::Face => {
                let f = t
                    .faces
                    .get(h.index as usize)
                    .ok_or_else(|| "names a face the model does not have".to_string())?;
                match f.geom.surface {
                    runmat_geometry_kernel::history::SurfaceKind::Cylinder {
                        axis_point,
                        axis_direction,
                        ..
                    } => hole_on_axis(&holes, axis_point, axis_direction)
                        .map(|h| h.position)
                        .ok_or_else(|| "is a cylindrical face of no hole".into()),
                    _ => {
                        Err("is a face, not a point of the model (name a vertex or a hole)".into())
                    }
                }
            }
            TopoKind::Edge => {
                let e = t
                    .edges
                    .get(h.index as usize)
                    .ok_or_else(|| "names an edge the model does not have".to_string())?;
                // A hole's rim: a closed edge on a circle about the hole's axis.
                if len3(sub3(e.geom.start, e.geom.end)) > HOLE_MATCH {
                    return Err("is an open edge, not a point of the model".into());
                }
                holes
                    .iter()
                    .find(|hole| {
                        let a = unit3(hole.axis);
                        let v = sub3(e.geom.start, hole.position);
                        let along = dot3(v, a);
                        let radial = sub3(v, [a[0] * along, a[1] * along, a[2] * along]);
                        let r = len3(radial);
                        let circumference = std::f64::consts::PI * 2.0 * r;
                        r > HOLE_MATCH && (circumference - e.geom.length).abs() < 1e-6
                    })
                    .map(|hole| hole.position)
                    .ok_or_else(|| "is a closed edge of no hole".into())
            }
            TopoKind::Body => Err("is a body, not a point of the model".into()),
        }
    }
}
