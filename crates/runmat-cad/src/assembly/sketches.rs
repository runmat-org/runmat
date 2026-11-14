//! Sketches in an assembly: `sketch s1 on origin.xy { … }` or `sketch s2 on plane.<connector>`,
//! held in the assembly's `features` in the part's feature form and solved by the part's sketch
//! solver.
//!
//! A sketch sits on an origin plane of the assembly or on the XY plane of one of its mate
//! connectors. A connector is fixed in its component, and anchored to what that component's
//! part publishes, so a sketch on a component's face follows the component when mates, a drag
//! or a placement move it, and follows the face when the part changes.
//!
//! Regeneration order: the occurrences are placed and the mates solved first
//! ([`super::resolve`]), then each sketch's frame is read from its plane and the sketch is
//! solved in the assembly's value environment.

use crate::assembly::model::{AnchorRef, AssemblyDocument};
use crate::assembly::resolve::Resolver;
use crate::diagnostics::{DiagCode, Severity, Site};
use crate::ids::{FeatureId, SketchEntityId};
use crate::model::datum::{DatumRef, Frame, PlaneRef};
use crate::model::sketch::{
    PointRole, Sketch, SketchGeometry, SketchSolveStatus, SolvedSketch,
};
use crate::regen::kernel::Transform;
use crate::model::{Feature, FeatureKind};
use crate::sketch_solver::{last_solved, ConstraintSketchSolver, SketchSolver};
use crate::value::ValueEnv;

/// A solved assembly sketch on its plane, in the assembly's coordinates.
#[derive(Clone, Debug)]
pub struct ResolvedSketch {
    pub id: FeatureId,
    pub sketch: Sketch,
    pub solved: SolvedSketch,
    pub frame: Frame,
}

impl ResolvedSketch {
    /// The sketch's closed profile loops, the largest first, in its plane's coordinates (place
    /// them with [`ResolvedSketch::frame`], which is in assembly coordinates). This is what an
    /// assembly feature's `profile = sketch.<name>` cuts, holes or extrudes through the parts.
    pub fn profile(
        &self,
    ) -> Result<Vec<Vec<runmat_geometry_kernel::op::ProfileSegment>>, String> {
        crate::regen::engine::profile_loops(&self.sketch, &self.solved)
    }
}

impl crate::assembly::resolve::ResolvedAssembly {
    /// The top assembly's solved sketch `name` (`profile = sketch.<name>`), if it solved.
    pub fn sketch(&self, name: &str) -> Option<&ResolvedSketch> {
        self.sketches.iter().find(|s| s.id.0 == name)
    }
}

/// The frame a connector on sketch point or line `entity.role` has, in the coordinates of the
/// sketch's `frame`'s space: at a point (Z the sketch's normal), or along a line from its start
/// (Z along the line, X the sketch's normal). With no role, a point entity's position, a
/// circle's, arc's or ellipse's center, or a line.
pub fn sketch_point_frame(
    frame: &Frame,
    sketch: &Sketch,
    solved: &SolvedSketch,
    entity: &SketchEntityId,
    role: Option<PointRole>,
) -> Result<Frame, String> {
    let geometry = &sketch
        .entities
        .iter()
        .find(|e| &e.id == entity)
        .ok_or_else(|| format!("the sketch has no `{entity}`"))?
        .geometry;
    let at = |p: [f64; 2]| Frame {
        origin: frame.to_world(p),
        ..*frame
    };
    let point = |role: PointRole| {
        solved
            .point(entity, role)
            .ok_or_else(|| format!("`{entity}` has no solved point `{}`", role.word()))
    };
    match (role, geometry) {
        (Some(role), _) => Ok(at(point(role)?)),
        (None, SketchGeometry::Point { .. }) => Ok(at(point(PointRole::At)?)),
        (
            None,
            SketchGeometry::Circle { .. } | SketchGeometry::Arc { .. } | SketchGeometry::Ellipse { .. },
        ) => Ok(at(point(PointRole::Center)?)),
        (None, SketchGeometry::Line { .. }) => {
            let (a, b) = (
                frame.to_world(point(PointRole::Start)?),
                frame.to_world(point(PointRole::End)?),
            );
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if l <= 0.0 {
                return Err(format!("`{entity}` has no length"));
            }
            let z = d.map(|c| c / l);
            let x = frame.z;
            let y = [
                z[1] * x[2] - z[2] * x[1],
                z[2] * x[0] - z[0] * x[2],
                z[0] * x[1] - z[1] * x[0],
            ];
            Ok(Frame { origin: a, x, y, z })
        }
        (None, _) => Err(format!(
            "pick a point of `{entity}`, or a sketch point, line, circle or arc"
        )),
    }
}

/// Why the assembly's `features` can't hold `feature`, if it can't: only sketches today.
pub fn unsupported(feature: &Feature) -> Option<String> {
    match &feature.kind {
        FeatureKind::Sketch(_) => None,
        _ => Some(format!(
            "`{}`: an assembly holds sketches; cuts, holes and weld beads across parts come with assembly features",
            feature.id
        )),
    }
}

impl<'a, 'b> Resolver<'a, 'b> {
    /// The frame of an assembly sketch's plane, in the coordinates of assembly node `owner`.
    fn sketch_frame(&mut self, owner: u32, plane: &PlaneRef) -> Result<Frame, String> {
        match plane {
            PlaneRef::Datum {
                datum: DatumRef::OriginPlane { plane },
            } => Ok(plane.frame()),
            PlaneRef::Datum {
                datum: DatumRef::OriginFrame,
            } => Ok(Frame::WORLD),
            PlaneRef::Datum {
                datum: DatumRef::Feature { feature },
            } => {
                let (host, f) = self.connector(owner, &feature.0).map_err(|e| {
                    format!("the sketch plane `plane.{feature}` is the assembly's connector `{feature}`: {e}")
                })?;
                Ok(self.chain(owner, host).compose(&f).to_frame())
            }
            PlaneRef::Datum { datum } => Err(format!("`{datum}` isn't a plane")),
            PlaneRef::Face { .. } => Err(
                "an assembly sketch sits on an origin plane or on a connector (`plane.<connector>`); a component face is picked through a connector"
                    .into(),
            ),
        }
    }

    /// The frame of an assembly connector anchored to sketch `sketch` (layout geometry for
    /// mates), in the coordinates of node `owner`. The sketch is solved here, before the mates
    /// that read it, so it must not move with them: it sits on an origin plane or on a fixed
    /// component. A sketch on a component that mates move would place that component's own
    /// mates, a cycle, which is refused.
    pub(in crate::assembly) fn sketch_anchor(
        &mut self,
        owner: u32,
        doc: &AssemblyDocument,
        env: &ValueEnv,
        sketch: &FeatureId,
        entity: &SketchEntityId,
        role: Option<PointRole>,
    ) -> Result<Transform, String> {
        let feature = doc
            .features
            .iter()
            .find(|f| &f.id == sketch)
            .ok_or_else(|| format!("the assembly has no sketch `{sketch}`"))?;
        let FeatureKind::Sketch(s) = &feature.kind else {
            return Err(format!("`{sketch}` isn't a sketch"));
        };
        if feature.suppressed {
            return Err(format!("the sketch `{sketch}` is suppressed"));
        }
        if let Some(problem) = moves_with_mates(doc, &s.plane) {
            return Err(format!(
                "the sketch `{sketch}` {problem}, so it can't place mates: the component would follow its own sketch"
            ));
        }
        let frame = self.sketch_frame(owner, &s.plane)?;
        let s = &self.with_projections(owner, &frame, s)?;
        let ev = self.ev();
        let solved = ConstraintSketchSolver
            .solve(s, &ev, env)
            .map_err(|f| format!("the sketch `{sketch}` doesn't solve: {}", f.message))?;
        let f = sketch_point_frame(&frame, s, &solved, entity, role)?;
        Ok(Transform::from_frame(&f))
    }

    /// Solves the sketches of the assembly at node `owner` (the top assembly), after its
    /// occurrences are placed and its mates solved.
    pub(in crate::assembly) fn solve_sketches(
        &mut self,
        owner: u32,
        features: &[Feature],
        env: &ValueEnv,
    ) {
        for feature in features {
            if let Some(message) = unsupported(feature) {
                self.diag(
                    DiagCode::RefKindMismatch,
                    Severity::Error,
                    Site::feature(&feature.id, None),
                    message,
                    Vec::new(),
                );
                continue;
            }
            let FeatureKind::Sketch(sketch) = &feature.kind else {
                continue;
            };
            if feature.suppressed {
                continue;
            }
            let frame = match self.sketch_frame(owner, &sketch.plane) {
                Ok(frame) => frame,
                Err(message) => {
                    self.diag(
                        DiagCode::AsmUnknownConnector,
                        Severity::Error,
                        Site::feature(&feature.id, Some("on")),
                        message,
                        Vec::new(),
                    );
                    continue;
                }
            };
            let sketch = &match self.with_projections(owner, &frame, sketch) {
                Ok(sketch) => sketch,
                Err(message) => {
                    self.diag(
                        DiagCode::RefKindMismatch,
                        Severity::Error,
                        Site::feature(&feature.id, None),
                        message,
                        Vec::new(),
                    );
                    continue;
                }
            };
            let ev = self.ev();
            let solved = match ConstraintSketchSolver.solve(sketch, &ev, env) {
                Ok(solved) => solved,
                Err(failure) => {
                    self.diag(
                        DiagCode::SketchSolveFailed,
                        Severity::Error,
                        Site::feature(&feature.id, Some("constraints")),
                        failure.message.clone(),
                        Vec::new(),
                    );
                    // Shown as it last solved, the conflict marked, as a part shows it.
                    let Ok(mut shown) = last_solved(sketch, &ev, env) else {
                        continue;
                    };
                    shown.status = SketchSolveStatus::OverDefined;
                    shown.conflicting = failure.constraints;
                    shown
                }
            };
            self.sketches.push(ResolvedSketch {
                id: feature.id.clone(),
                sketch: sketch.clone(),
                solved,
                frame,
            });
        }
    }
}

/// Why a sketch on `plane` moves when the mates solve, if it does: it sits on a connector of a
/// floating component, or on a connector that is itself on a sketch.
pub fn moves_with_mates(doc: &AssemblyDocument, plane: &PlaneRef) -> Option<String> {
    let PlaneRef::Datum {
        datum: DatumRef::Feature { feature },
    } = plane
    else {
        return None;
    };
    let c = doc.connector(&feature.0)?;
    if matches!(c.anchor, AnchorRef::Sketch { .. }) {
        return Some(format!("sits on `{}`, a connector on a sketch", c.name));
    }
    let top = c.occurrence.0.first()?;
    let floating = doc
        .occurrences
        .iter()
        .any(|o| o.name == top.name && !o.fixed);
    floating.then(|| format!("sits on `{}`, which mates move", top.name))
}
