//! What the Sheet Metal tab's commands read from the part they edit:
//!
//! - whether a base flange's sketch is closed: a closed profile makes a flat plate and takes no
//!   depth, an open chain is swept `depth` along the sketch normal (the form always carries a
//!   depth, so the lowering writes it only for an open chain);
//! - the face a flat pattern holds still: the plate face of a base flange or a converted solid,
//!   as the guide's parts name it (`feature.base.face.sheet(plate, ordinal = 1)`), or for a
//!   bent section the flat of its sketch's first curve (`feature.base.face.sheet(l1, ordinal = 1)`).

use crate::edit::model;
use crate::picks::Picks;
use runmat_cad::model::{FeatureKind, PointExpr, SheetMetalFeature, SketchGeometry};
use runmat_cad::syntax::edit::EditFailure;
use serde_json::Value;

/// True when sketch `id` closes on itself: a circle or a closed spline, or lines, arcs and open
/// splines whose ends all meet in pairs. Construction geometry doesn't count.
pub fn closed_profile(picks: &Picks, id: &str) -> bool {
    let Some(FeatureKind::Sketch(sketch)) = picks.document.features.iter().find(|f| f.id.0 == id).map(|f| &f.kind) else {
        return false;
    };
    let mut ends: Vec<&PointExpr> = Vec::new();
    for entity in sketch.entities.iter().filter(|e| !e.construction) {
        match &entity.geometry {
            SketchGeometry::Circle { .. } | SketchGeometry::Spline { closed: true, .. } => return true,
            SketchGeometry::Line { start, end } | SketchGeometry::Arc { start, end, .. } => ends.extend([start, end]),
            SketchGeometry::Spline { points, closed: false } => ends.extend(points.first().into_iter().chain(points.last())),
            SketchGeometry::Point { .. } => {}
        }
    }
    !ends.is_empty() && ends.iter().all(|p| ends.iter().filter(|q| *q == p).count() % 2 == 0)
}

/// The flat pattern's fixed face: the plate face of the first base flange or converted solid,
/// or a bent base flange's first flat.
pub fn flat_fixed_face(picks: &Picks) -> Result<Value, EditFailure> {
    let base = picks
        .document
        .features
        .iter()
        .find(|f| {
            matches!(
                f.kind,
                FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { .. } | SheetMetalFeature::ConvertToSheetMetal { .. })
            )
        })
        .ok_or_else(|| model("a flat pattern unfolds a base flange or a solid converted to sheet metal; this part has neither"))?;
    // A plate holds its plate face still; a bent section the flat of its first curve.
    let FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange { profile, .. }) = &base.kind else {
        return picks.reference(&format!("feature.{}.face.sheet(plate, ordinal = 1)", base.id.0));
    };
    if closed_profile(picks, &profile.sketch.0) {
        return picks.reference(&format!("feature.{}.face.sheet(plate, ordinal = 1)", base.id.0));
    }
    let first = match picks.document.features.iter().find(|f| f.id == profile.sketch).map(|f| &f.kind) {
        Some(FeatureKind::Sketch(sketch)) => sketch.entities.iter().find(|e| !e.construction).map(|e| e.id.0.clone()),
        _ => None,
    }
    .ok_or_else(|| model("the base flange's sketch has no curve to hold still"))?;
    picks.reference(&format!("feature.{}.face.sheet({first}, ordinal = 1)", base.id.0))
}
