//! Lowering of the feature families (direct edits, surfaces, sheet metal, weldments, mold,
//! details, reference curves, mate connectors and the added datum definitions) to canonical
//! kernel operations and typed records.
//!
//! Each family lowers to the kernel seam ([`KernelOp::Labeled`], [`KernelOp::MoveFaces`],
//! [`KernelOp::Surface`], ...) with labels from the definition, or to a [`FeatureRecord`] when
//! it makes no geometry. A case the seam has no operation for yet fails the feature with
//! `cad.kernel.not_supported` naming the case: never a quietly different body.

mod datums;
mod curves;
mod direct;
mod surface;
mod sheet_metal;
mod unfold;
mod flanges;
mod bend_chain;
mod weldment;
mod mold;
mod details;
mod tools;
pub(super) use tools::*;

use super::*;
use crate::model::curves::{ConnectorPlacement, CurveDef, CurveValue, MateConnectorDef};
use crate::model::detail::{Emboss, Rib, Thread};
use crate::model::direct::{DirectEdit, DirectMotion, LiveRelation};
use crate::model::mold::MoldFeature;
use crate::model::sheet_metal::{
    section, BendRule, BendRuleValue, HemKind, LengthReference, SheetMetalFeature, SheetMetalRules,
    StripRegion,
};
use crate::model::surface::SurfaceFeature;
use crate::model::weldment::{cut_list, member_cuts, CornerTreatment, WeldmentFeature};
use crate::refs::history::{FaceGeom, KernelTag};
use crate::refs::name::Role;
use crate::refs::naming::Restore;
use crate::regen::records::{FaceDraft, FeatureRecord, SheetRules};

pub(super) fn line_seg(curve: &str, a: [f64; 2], b: [f64; 2]) -> ProfileSegment {
    ProfileSegment {
        curve: crate::ids::SketchEntityId::new(curve),
        kind: SegmentKind::Line,
        start: a,
        end: b,
    }
}

pub(super) fn polygon(frame: Frame, pts: &[(String, [f64; 2])]) -> CanonicalProfile {
    let n = pts.len();
    CanonicalProfile {
        frame,
        loops: vec![(0..n)
            .map(|k| line_seg(&pts[k].0, pts[k].1, pts[(k + 1) % n].1))
            .collect()],
        merge_collinear: false,
    }
}

pub(super) fn frame_of(origin: [f64; 3], x: [f64; 3], y: [f64; 3]) -> Frame {
    let x = normalize(x);
    let y = normalize(sub(y, scale(x, dot(y, x))));
    Frame {
        origin,
        x,
        y,
        z: cross(x, y),
    }
}

pub(super) fn op(op: KernelOp, operands: Vec<usize>, naming: NamingContext) -> Compiled {
    Compiled::Op {
        op,
        operands,
        naming,
        import: false,
        instances: None,
        tools: None,
    }
}

pub(super) fn labeled(
    tools: Vec<LabeledTool>,
    carve: Vec<LabeledTool>,
    combine: Combine,
    separate: bool,
) -> KernelOp {
    KernelOp::Labeled {
        tools,
        carve,
        combine,
        separate,
    }
}

impl<'a, 'b> Run<'a, 'b> {
    pub(super) fn compile_family(
        &mut self,
        i: usize,
        f: &Feature,
    ) -> Result<Option<Compiled>, Halt> {
        let id = &f.id;
        Ok(Some(match &f.kind {
            FeatureKind::Plane(
                PlaneDef::ThreePoint { .. }
                | PlaneDef::Mid { .. }
                | PlaneDef::NormalToCurve { .. }
                | PlaneDef::TangentToFace { .. },
            )
            | FeatureKind::Axis(AxisDef::TwoPlanes { .. } | AxisDef::Normal { .. })
            | FeatureKind::Point(
                PointDef::CenterOf { .. }
                | PointDef::OnCurve { .. }
                | PointDef::Intersection { .. },
            ) => Compiled::Datum(self.datum_extra(i, id, &f.kind)?),
            FeatureKind::MateConnector(m) => Compiled::Datum(self.connector(i, id, m)?),
            FeatureKind::Curve(c) => self.curve(i, id, c)?,
            FeatureKind::Direct(d) => self.direct(i, id, d)?,
            FeatureKind::Surface(s) => self.surface_feature(i, id, s)?,
            FeatureKind::SheetMetal(s) => self.sheet_metal(i, id, s)?,
            FeatureKind::Weldment(w) => self.weldment(i, id, w)?,
            FeatureKind::Mold(m) => self.mold(i, id, m)?,
            FeatureKind::Thread(t) => self.thread(i, id, t)?,
            FeatureKind::Emboss(e) => self.emboss(i, id, e)?,
            FeatureKind::Rib(r) => self.rib(i, id, r)?,
            FeatureKind::Library(_) => {
                return Err(self.error(
                    id,
                    None,
                    DiagCode::RegenInvalid,
                    "library features are expanded before regeneration; this one was not \
                     (its definition is missing)"
                        .into(),
                    Vec::new(),
                ))
            }
            _ => return Ok(None),
        }))
    }

    pub(super) fn not_lowered(&mut self, id: &FeatureId, what: &str) -> Halt {
        self.error(
            id,
            None,
            DiagCode::KernelNotSupported,
            format!("not supported yet: {}", what),
            Vec::new(),
        )
    }

    pub(super) fn face_handle(
        &mut self,
        i: usize,
        id: &FeatureId,
        param: &str,
        r: &Reference,
    ) -> Result<EntityHandle, Halt> {
        let res = self.single(i, id, param, r, Some(TopoKind::Face))?;
        Ok(*res.entities().iter().next().expect("one"))
    }

    pub(super) fn face_normal(&self, h: &EntityHandle) -> Option<[f64; 3]> {
        match self.st.bodies[h.body as usize].topology.faces[h.index as usize]
            .geom
            .surface
        {
            SurfaceKind::Plane { normal } => Some(normalize(normal)),
            _ => None,
        }
    }

    pub(super) fn chain(
        &mut self,
        i: usize,
        f: &FeatureId,
        param: &str,
        p: &ProfileRef,
    ) -> Result<(Frame, Vec<ProfileSegment>), Halt> {
        self.gate_feature(i, f, &p.sketch, param)?;
        let Some((solved, frame)) = self.st.sketches.get(&p.sketch).cloned() else {
            return Err(self.error(
                f,
                Some(param),
                DiagCode::RefKindMismatch,
                format!("`{}` is not a sketch", p.sketch),
                Vec::new(),
            ));
        };
        let Some(FeatureKind::Sketch(sketch)) = self.doc.feature(&p.sketch).map(|x| &x.kind) else {
            unreachable!("gated")
        };
        build_path(sketch, &solved)
            .map(|segs| (frame, segs))
            .map_err(|m| self.error(f, Some(param), DiagCode::SketchOpenProfile, m, Vec::new()))
    }

    // --------------------------------------------------------------------- datums
}
