//! [`Params`] for the Features-tab kinds with options (extrude end conditions, holes, fillets,
//! chamfers, shells, drafts and patterns): one visitor feeds values, references and features
//! named directly.

use crate::model::feature::*;
use crate::model::visit::{Params, Visit};

fn extent<'a>(v: &mut Visit<'a>, prefix: &str, e: &'a Extent) {
    match e {
        Extent::Blind { distance } | Extent::Symmetric { distance } => v.value(distance),
        Extent::ThroughAll | Extent::UpToNext => {}
        Extent::UpTo { face } => v.reference(format!("{}up_to", prefix), face),
        Extent::UpToPlane { plane } => v.plane(&format!("{}up_to", prefix), plane),
        Extent::UpToVertex { vertex } => v.point(&format!("{}up_to", prefix), vertex),
        Extent::OffsetFromFace { face, offset, .. } => {
            v.reference(format!("{}up_to", prefix), face);
            v.value(offset);
        }
        Extent::UpToBody { body, offset } => {
            v.reference(format!("{}up_to", prefix), body);
            v.opt_value(offset);
        }
    }
}

fn mode<'a>(v: &mut Visit<'a>, m: &'a BodyMode) {
    v.references("targets", m.targets());
}

fn hole_form<'a>(v: &mut Visit<'a>, prefix: &str, f: &'a HoleForm) {
    let _ = prefix;
    match f {
        HoleForm::Simple => {}
        HoleForm::Counterbore { diameter, depth } => {
            v.opt_value(diameter);
            v.opt_value(depth);
        }
        HoleForm::Countersink { diameter, angle } => {
            v.opt_value(diameter);
            v.opt_value(angle);
        }
        HoleForm::Tapped {
            thread_depth,
            near_countersink,
            ..
        } => {
            v.opt_value(thread_depth);
            v.opt_value(near_countersink);
        }
        HoleForm::PipeTap { thread_depth, .. } => v.opt_value(thread_depth),
        HoleForm::Slot {
            length,
            angle,
            counterbore,
        } => {
            v.value(length);
            v.opt_value(angle);
            if let Some((d, h)) = counterbore {
                v.value(d);
                v.value(h);
            }
        }
    }
}

fn profile<'a>(v: &mut Visit<'a>, p: &'a FilletProfile) {
    if let FilletProfile::Conic { rho } = p {
        v.value(rho);
    }
}

impl Params for Extrude {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.profile_source(&self.profile);
        extent(v, "", &self.extent);
        if let Some(s) = &self.second {
            extent(v, "second.", s);
        }
        match &self.from {
            ExtrudeFrom::SketchPlane => {}
            ExtrudeFrom::Offset { distance } => v.value(distance),
            ExtrudeFrom::Surface { face } => v.reference("from", face),
            ExtrudeFrom::Vertex { vertex } => v.point("from", vertex),
        }
        if let Some(d) = &self.draft {
            v.value(&d.angle);
            v.opt_value(&d.second_angle);
        }
        if let Some(t) = &self.thin {
            v.value(&t.thickness);
            v.opt_value(&t.second_thickness);
            v.opt_value(&t.cap_ends);
        }
        mode(v, &self.mode);
    }
    fn kind_name(&self) -> &'static str {
        "extrude"
    }
}

impl Params for Hole {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.reference("face", &self.face);
        for (i, p) in self.positions.iter().enumerate() {
            v.point(&format!("positions[{}]", i), &p.at);
        }
        v.opt_value(&self.diameter);
        match &self.end {
            HoleEnd::Blind { depth } => v.value(depth),
            HoleEnd::ThroughAll | HoleEnd::UpToNext => {}
            HoleEnd::UpToFace { face, offset } => {
                v.reference("up_to", face);
                v.opt_value(offset);
            }
        }
        hole_form(v, "form", &self.form);
        v.opt_value(&self.drill_point);
        if let Some(s) = &self.series {
            v.references("series.bodies", &s.bodies);
            hole_form(v, "series.end", &s.end);
            v.opt_value(&s.middle_diameter);
        }
    }
    fn kind_name(&self) -> &'static str {
        "hole"
    }
    fn implicit_targets(&self) -> bool {
        true
    }
}

impl Params for Fillet {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.references("edges", &self.edges);
        v.value(&self.radius);
        for r in &self.variable {
            v.value(&r.at);
            v.value(&r.radius);
        }
        profile(v, &self.profile);
        for (i, s) in self.setbacks.iter().enumerate() {
            v.reference(format!("setbacks[{}]", i), &s.vertex);
            v.value(&s.distance);
        }
    }
    fn kind_name(&self) -> &'static str {
        if self.variable.is_empty() {
            "fillet"
        } else {
            "fillet.variable"
        }
    }
}

impl Params for FaceFillet {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.references("first", &self.first);
        v.references("second", &self.second);
        v.value(&self.radius);
        profile(v, &self.profile);
    }
    fn kind_name(&self) -> &'static str {
        "fillet.face"
    }
}

impl Params for FullRound {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.references("first", &self.first);
        v.references("center", &self.center);
        v.references("second", &self.second);
    }
    fn kind_name(&self) -> &'static str {
        "fillet.full_round"
    }
}

impl Params for Chamfer {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.references("edges", &self.edges);
        match &self.form {
            ChamferForm::Equal { distance } => v.value(distance),
            ChamferForm::Distances {
                distance,
                distance2,
            } => {
                v.value(distance);
                v.value(distance2);
            }
            ChamferForm::DistanceAngle { distance, angle } => {
                v.value(distance);
                v.value(angle);
            }
            ChamferForm::Vertex { distances } => distances.iter().for_each(|d| v.value(d)),
        }
    }
    fn kind_name(&self) -> &'static str {
        match self.form {
            ChamferForm::Vertex { .. } => "chamfer.vertex",
            _ => "chamfer",
        }
    }
}

impl Params for Shell {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.reference("body", &self.body);
        v.references("open_faces", &self.open_faces);
        v.value(&self.thickness);
        for (i, f) in self.faces.iter().enumerate() {
            v.references(&format!("faces[{}].faces", i), &f.faces);
            v.value(&f.thickness);
        }
    }
    fn kind_name(&self) -> &'static str {
        "shell"
    }
}

impl Params for Draft {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        v.references("faces", &self.faces);
        v.plane("neutral", &self.neutral);
        v.value(&self.angle);
        v.references("parting", self.kind.edges());
    }
    fn kind_name(&self) -> &'static str {
        match self.kind {
            DraftKind::NeutralPlane => "draft",
            DraftKind::PartingLine { .. } => "draft.parting_line",
            DraftKind::Step { .. } => "draft.step",
        }
    }
}

impl Params for Pattern {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        match &self.seeds {
            PatternSeeds::Features { features } => features.iter().for_each(|f| v.feature(f)),
            PatternSeeds::Bodies { bodies } => v.references("bodies", bodies),
        }
        match &self.layout {
            PatternLayout::Linear {
                direction,
                spacing,
                count,
            } => {
                v.axis("direction", direction);
                v.value(spacing);
                v.value(count);
            }
            PatternLayout::Grid {
                first,
                first_spacing,
                first_count,
                second,
                second_spacing,
                second_count,
            } => {
                v.axis("first", first);
                v.value(first_spacing);
                v.value(first_count);
                v.axis("second", second);
                v.value(second_spacing);
                v.value(second_count);
            }
            PatternLayout::Circular {
                axis, count, angle, ..
            } => {
                v.axis("axis", axis);
                v.value(count);
                v.value(angle);
            }
            PatternLayout::Table { rows } => {
                for r in rows {
                    v.value(&r.dx);
                    v.value(&r.dy);
                    v.value(&r.dz);
                }
            }
            PatternLayout::Curve {
                curve,
                count,
                spacing,
                ..
            } => {
                curve.visit(v, "curve");
                v.value(count);
                v.opt_value(spacing);
            }
            PatternLayout::Sketch { sketch, .. } => v.feature(sketch),
            PatternLayout::Fill {
                boundary,
                spacing,
                direction,
                margin,
                seed,
                ..
            } => {
                v.profile(boundary);
                v.value(spacing);
                if let Some(d) = direction {
                    v.axis("direction", d);
                }
                v.opt_value(margin);
                if let Some(s) = seed {
                    v.point("seed", s);
                }
            }
            PatternLayout::Variable { instances } => {
                for x in instances {
                    v.value(&x.dx);
                    v.value(&x.dy);
                    v.value(&x.dz);
                    x.values.values().for_each(|e| v.value(e));
                }
            }
        }
    }
    fn kind_name(&self) -> &'static str {
        match self.layout {
            PatternLayout::Linear { .. } => "pattern.linear",
            PatternLayout::Grid { .. } => "pattern.grid",
            PatternLayout::Circular { .. } => "pattern.circular",
            PatternLayout::Table { .. } => "pattern.table",
            PatternLayout::Curve { .. } => "pattern.curve",
            PatternLayout::Sketch { .. } => "pattern.sketch",
            PatternLayout::Fill { .. } => "pattern.fill",
            PatternLayout::Variable { .. } => "pattern.variable",
        }
    }
    fn implicit_targets(&self) -> bool {
        matches!(self.seeds, PatternSeeds::Features { .. })
    }
}
