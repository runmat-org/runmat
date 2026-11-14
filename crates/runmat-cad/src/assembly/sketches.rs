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

use crate::assembly::resolve::Resolver;
use crate::diagnostics::{DiagCode, Severity, Site};
use crate::ids::FeatureId;
use crate::model::datum::{DatumRef, Frame, PlaneRef};
use crate::model::sketch::{Sketch, SketchSolveStatus, SolvedSketch};
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
