//! Direct edits, recorded as features: move, offset, rotate, replace and delete faces; move
//! and rotate bodies; Instant3D dimension drags recorded as feature edits; and live relations
//! (keep parallel, concentric, tangent, coplanar, symmetric, fixed) written as typed relations
//! the edit must keep.
//!
//! A direct edit is history like any other feature: its faces continue the faces it moved, so
//! references downstream keep resolving, and an edit that would break a live relation fails
//! with a diagnostic naming the relation instead of quietly breaking it.

use crate::ids::FeatureId;
use crate::model::datum::{AxisRef, PlaneRef};
use crate::model::visit::{Params, Visit};
use crate::model::PartDocument;
use crate::refs::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};

/// A relation a direct edit keeps between the faces it moves and the faces around them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "keep")]
pub enum LiveRelation {
    Parallel {
        a: Reference,
        b: Reference,
    },
    Perpendicular {
        a: Reference,
        b: Reference,
    },
    Coplanar {
        a: Reference,
        b: Reference,
    },
    Concentric {
        a: Reference,
        b: Reference,
    },
    Tangent {
        a: Reference,
        b: Reference,
    },
    /// Symmetric about a plane.
    Symmetric {
        a: Reference,
        b: Reference,
        plane: PlaneRef,
    },
    /// The face does not move.
    Fixed {
        face: Reference,
    },
}

impl LiveRelation {
    fn visit<'a>(&'a self, v: &mut Visit<'a>, i: usize) {
        let p = |s: &str| format!("keep[{}].{}", i, s);
        match self {
            LiveRelation::Parallel { a, b }
            | LiveRelation::Perpendicular { a, b }
            | LiveRelation::Coplanar { a, b }
            | LiveRelation::Concentric { a, b }
            | LiveRelation::Tangent { a, b } => {
                v.reference(p("a"), a);
                v.reference(p("b"), b);
            }
            LiveRelation::Symmetric { a, b, plane } => {
                v.reference(p("a"), a);
                v.reference(p("b"), b);
                v.plane(&p("plane"), plane);
            }
            LiveRelation::Fixed { face } => v.reference(p("face"), face),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "motion")]
pub enum DirectMotion {
    /// Along each face's normal (offset face, push-pull).
    Offset {
        distance: ValueExpr,
    },
    Translate {
        dx: ValueExpr,
        dy: ValueExpr,
        dz: ValueExpr,
    },
    Rotate {
        axis: AxisRef,
        angle: ValueExpr,
    },
}

impl DirectMotion {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        match self {
            DirectMotion::Offset { distance } => v.value(distance),
            DirectMotion::Translate { dx, dy, dz } => {
                v.value(dx);
                v.value(dy);
                v.value(dz);
            }
            DirectMotion::Rotate { axis, angle } => {
                v.axis("axis", axis);
                v.value(angle);
            }
        }
    }
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "direct")]
pub enum DirectEdit {
    /// Move faces (move face, offset face, push-pull); neighbours follow.
    MoveFaces {
        faces: Vec<Reference>,
        motion: DirectMotion,
        #[serde(default)]
        keep: Vec<LiveRelation>,
    },
    /// Replace faces by a face of a surface body.
    ReplaceFaces {
        faces: Vec<Reference>,
        with: Reference,
    },
    /// Delete faces; `heal` extends the neighbours to close the gap, otherwise the body
    /// becomes a surface body open where the faces were.
    DeleteFaces {
        faces: Vec<Reference>,
        #[serde(default = "yes")]
        heal: bool,
    },
    /// Move or rotate whole bodies (move/copy body).
    MoveBodies {
        bodies: Vec<Reference>,
        motion: DirectMotion,
        /// Keep the originals and move copies.
        #[serde(default)]
        copy: bool,
    },
}

fn yes() -> bool {
    true
}

impl Params for DirectEdit {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        match self {
            DirectEdit::MoveFaces {
                faces,
                motion,
                keep,
            } => {
                v.references("faces", faces);
                motion.visit(v);
                for (i, k) in keep.iter().enumerate() {
                    k.visit(v, i);
                }
            }
            DirectEdit::ReplaceFaces { faces, with } => {
                v.references("faces", faces);
                v.reference("with", with);
            }
            DirectEdit::DeleteFaces { faces, .. } => v.references("faces", faces),
            DirectEdit::MoveBodies { bodies, motion, .. } => {
                v.references("bodies", bodies);
                motion.visit(v);
            }
        }
    }
    fn kind_name(&self) -> &'static str {
        match self {
            DirectEdit::MoveFaces { .. } => "direct.move_faces",
            DirectEdit::ReplaceFaces { .. } => "direct.replace_faces",
            DirectEdit::DeleteFaces { .. } => "direct.delete_faces",
            DirectEdit::MoveBodies { .. } => "direct.move_bodies",
        }
    }
}

// ------------------------------------------------------------------ Instant3D edits

/// What an Instant3D drag (or any dimension edit in the viewport) changes: a value in the
/// document, located by where it lives. The drag never writes geometry; it writes this value,
/// and regeneration does the rest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum DimensionTarget {
    /// An input's default (`input width = 80 mm`).
    Input { input: crate::ids::InputName },
    /// A feature parameter by its parameter path (`extent.distance`, `radius`).
    Feature { feature: FeatureId, param: String },
    /// A sketch dimension constraint, by index.
    SketchConstraint { sketch: FeatureId, index: usize },
}

/// A recorded dimension edit: what changed, from what, to what.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DimensionEdit {
    pub target: DimensionTarget,
    pub from: ValueExpr,
    pub to: ValueExpr,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error")]
pub enum DimensionEditError {
    #[error("no input `{input}`")]
    NoInput { input: String },
    #[error("no feature `{feature}`")]
    NoFeature { feature: String },
    #[error("`{feature}` has no editable value at `{param}`")]
    NoParam { feature: String, param: String },
    #[error("`{sketch}` constraint {index} is not a dimension")]
    NotADimension { sketch: String, index: usize },
}

/// Apply a dimension edit to the document (the Instant3D write). Returns the edit with the
/// value it replaced, so it can be undone or recorded.
pub fn apply_dimension_edit(
    doc: &mut PartDocument,
    target: DimensionTarget,
    to: ValueExpr,
) -> Result<DimensionEdit, DimensionEditError> {
    let slot: &mut ValueExpr = match &target {
        DimensionTarget::Input { input } => {
            &mut doc
                .inputs
                .iter_mut()
                .find(|i| &i.name == input)
                .ok_or_else(|| DimensionEditError::NoInput {
                    input: input.0.clone(),
                })?
                .default
        }
        DimensionTarget::Feature { feature, param } => {
            let f = doc
                .features
                .iter_mut()
                .find(|f| &f.id == feature)
                .ok_or_else(|| DimensionEditError::NoFeature {
                    feature: feature.0.clone(),
                })?;
            feature_value_mut(&mut f.kind, param).ok_or_else(|| DimensionEditError::NoParam {
                feature: feature.0.clone(),
                param: param.clone(),
            })?
        }
        DimensionTarget::SketchConstraint { sketch, index } => {
            use crate::model::sketch::SketchConstraintKind as K;
            let f = doc
                .features
                .iter_mut()
                .find(|f| &f.id == sketch)
                .ok_or_else(|| DimensionEditError::NoFeature {
                    feature: sketch.0.clone(),
                })?;
            let not_dim = || DimensionEditError::NotADimension {
                sketch: sketch.0.clone(),
                index: *index,
            };
            let crate::model::FeatureKind::Sketch(s) = &mut f.kind else {
                return Err(not_dim());
            };
            match s.constraints.get_mut(*index).map(|c| &mut c.kind) {
                Some(
                    K::Distance { value, .. }
                    | K::Length { value, .. }
                    | K::Radius { value, .. }
                    | K::Diameter { value, .. }
                    | K::Angle { value, .. },
                ) => value,
                _ => return Err(not_dim()),
            }
        }
    };
    let from = std::mem::replace(slot, to.clone());
    Ok(DimensionEdit { target, from, to })
}

/// The value at a parameter path of a feature, for edits.
fn feature_value_mut<'a>(
    kind: &'a mut crate::model::FeatureKind,
    param: &str,
) -> Option<&'a mut ValueExpr> {
    use crate::model::feature::*;
    use crate::model::FeatureKind as F;
    Some(match (kind, param) {
        (F::Extrude(e), "extent.distance") => match &mut e.extent {
            Extent::Blind { distance } | Extent::Symmetric { distance } => distance,
            _ => return None,
        },
        (F::Revolve(r), "angle") => &mut r.angle,
        (F::Extrude(e), "second.distance") => match e.second.as_mut()? {
            Extent::Blind { distance } => distance,
            _ => return None,
        },
        (F::Extrude(e), "draft.angle") => &mut e.draft.as_mut()?.angle,
        (F::Extrude(e), "thin.thickness") => &mut e.thin.as_mut()?.thickness,
        (F::Hole(h), "diameter") => h.diameter.as_mut()?,
        (F::Hole(h), "depth") => match &mut h.end {
            HoleEnd::Blind { depth } => depth,
            _ => return None,
        },
        (F::Fillet(f), "radius") => &mut f.radius,
        (F::FaceFillet(f), "radius") => &mut f.radius,
        (F::Chamfer(c), "distance") => match &mut c.form {
            ChamferForm::Equal { distance }
            | ChamferForm::Distances { distance, .. }
            | ChamferForm::DistanceAngle { distance, .. } => distance,
            ChamferForm::Vertex { distances } => &mut distances[0],
        },
        (F::Chamfer(c), "distance2") => match &mut c.form {
            ChamferForm::Distances { distance2, .. } => distance2,
            _ => return None,
        },
        (F::Chamfer(c), "angle") => match &mut c.form {
            ChamferForm::DistanceAngle { angle, .. } => angle,
            _ => return None,
        },
        (F::Shell(s), "thickness") => &mut s.thickness,
        (F::Draft(d), "angle") => &mut d.angle,
        (
            F::Direct(DirectEdit::MoveFaces {
                motion: DirectMotion::Offset { distance },
                ..
            }),
            "motion.distance",
        ) => distance,
        (F::SheetMetal(crate::model::SheetMetalFeature::EdgeFlange { length, .. }), "length") => {
            length
        }
        (F::SheetMetal(crate::model::SheetMetalFeature::EdgeFlange { angle, .. }), "angle") => {
            angle
        }
        _ => return None,
    })
}
