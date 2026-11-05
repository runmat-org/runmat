//! K1: RunMat CAD's geometry engine on OCCT.
//!
//! [`OcctKernel`] implements the kernel-neutral contract of `runmat-geometry-kernel`
//! ([`Kernel`], [`GeometryAnalyzer`], [`ExactQueries`]) on exact B-rep. Each canonical operation
//! is lowered here into a short program of OCCT instructions (`exact_kernel.cc`); the program
//! returns the output bodies with topology summaries and, per face and per sheet boundary edge,
//! the origins OCCT's history records give it. Labels are assigned here, from the canonical
//! operation only (profile curve ids, hole keys, the edge a blend replaces), and echoed back by
//! the C++ side; nothing is named from output order. An operation whose history OCCT cannot
//! report fails with [`KernelError::HistoryIncomplete`], and every case the backend does not
//! build fails with [`KernelError::NotSupported`]: never a quietly wrong body.
//!
//! Operations keep the reference kernel's semantics (`runmat_cad::regen::fake`): flush faces of
//! different sources merge, distinct faces of one operand and distinct profile curves stay
//! separate, roles and ordinals follow the definition.

use runmat_geometry_kernel::analysis::{
    draft_analysis, AnalysisRequest, AnalysisResult, EdgeContinuity, FaceValue, GeometryAnalyzer,
    MassProperties, SectionProperties,
};
use runmat_geometry_kernel::frame::Frame;
use runmat_geometry_kernel::history::{
    BodyRecord, BoundaryRecord, EdgeGeom, EdgeTopo, Evolution, FaceGeom, FaceRecord, FaceTopo,
    HistorySource, KernelTag, OpHistory, OperandEntity, SurfaceKind, Topology, VertexTopo,
};
use runmat_geometry_kernel::ids::SketchEntityId;
use runmat_geometry_kernel::names::{InstanceKey, PlaneSide, Role};
use runmat_geometry_kernel::op::{
    BlendProfile, BodyHandle, CanonicalProfile, ChamferSize, Combine, FaceLabel, FaceMotion,
    HoleDepth, Kernel, KernelBody, KernelError, KernelId, KernelOp, KernelOutput, LabeledTool,
    PrismBound, ProfileSegment, SegmentKind, SurfaceOp, ToolOp, ToolShape, Transform,
};
use runmat_geometry_kernel::queries::{
    ExactQueries, ExchangeDocument, ExchangeFormat, Placed, ProjectedCurve, ProjectedSource,
    ProjectionRequest, Proximity,
};
use runmat_geometry_kernel::sections::{hole_section, StageSpan};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

// ------------------------------------------------------------------------- the C++ boundary

/// One instruction of a kernel program (see `exact_kernel.hxx`). Read only by the native
/// backend.
#[cfg_attr(
    not(all(
        any(not(target_arch = "wasm32"), target_os = "emscripten"),
        feature = "occt-kernel"
    )),
    allow(dead_code)
)]
#[derive(Clone, Debug, Default)]
pub(crate) struct Instr {
    pub op: u32,
    pub ints: Vec<i64>,
    pub reals: Vec<f64>,
    pub strs: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ROrigin {
    pub kind: u8,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct RFace {
    pub origins: Vec<ROrigin>,
    pub kept: bool,
    pub surface: u8,
    pub geom: Vec<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct REdge {
    pub f0: u32,
    pub f1: u32,
    pub laminar: bool,
    pub geom: Vec<f64>,
    pub origins: Vec<ROrigin>,
}

#[derive(Clone, Debug)]
pub(crate) struct RVertex {
    pub edges: Vec<u32>,
    pub point: Vec<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct RBody {
    pub id: u64,
    pub output: u32,
    pub sheet: bool,
    pub faces: Vec<RFace>,
    pub edges: Vec<REdge>,
    pub vertices: Vec<RVertex>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RResult {
    pub bodies: Vec<RBody>,
    pub names: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct RProjected {
    pub body: u32,
    pub kind: u8,
    pub entity: u32,
    pub visible: bool,
    pub points: Vec<f64>,
}

/// A body's display mesh as the C++ side returns it, in summary order: per face its flat
/// positions and triangle indices, per edge its flat points, and the flat vertex points.
#[derive(Clone, Debug, Default)]
pub(crate) struct RDisplay {
    /// Per face: positions, triangles, normals (flat).
    pub faces: Vec<(Vec<f64>, Vec<u32>, Vec<f64>)>,
    pub edges: Vec<Vec<f64>>,
    pub vertices: Vec<f64>,
}

const ORIGIN_INPUT: u8 = 0;
const ORIGIN_LABEL: u8 = 1;
const ORIGIN_COPY: u8 = 2;
const ORIGIN_NAME: u8 = 3;

/// A cap that a later step of the same program removes (a fused seam, a trimmed far end).
const INTERNAL: i64 = 0xffff_fffe;

#[cfg(all(
    any(not(target_arch = "wasm32"), target_os = "emscripten"),
    feature = "occt-kernel"
))]
mod backend {
    use super::*;
    use crate::occt::kernel_ffi::bridge as b;

    fn err(e: cxx::Exception) -> KernelError {
        let m = e.what();
        if let Some(r) = m.strip_prefix("unsupported: ") {
            KernelError::NotSupported {
                what: format!("OCCT kernel: {r}"),
            }
        } else if let Some(r) = m.strip_prefix("history: ") {
            KernelError::HistoryIncomplete {
                message: format!("OCCT kernel: {r}"),
            }
        } else {
            KernelError::Failed {
                message: m.strip_prefix("failed: ").unwrap_or(m).to_string(),
            }
        }
    }

    fn instrs(p: &[Instr]) -> Vec<b::KInstr> {
        p.iter()
            .map(|i| b::KInstr {
                op: i.op,
                ints: i.ints.clone(),
                reals: i.reals.clone(),
                strs: i.strs.clone(),
            })
            .collect()
    }

    fn origins(o: &[b::KOrigin]) -> Vec<ROrigin> {
        o.iter()
            .map(|x| ROrigin {
                kind: x.kind,
                a: x.a,
                b: x.b,
                c: x.c,
            })
            .collect()
    }

    pub fn execute(program: &[Instr], operands: &[u64]) -> Result<RResult, KernelError> {
        let r = b::kernel_execute(&instrs(program), operands).map_err(err)?;
        Ok(RResult {
            names: r.names,
            bodies: r
                .bodies
                .into_iter()
                .map(|k| RBody {
                    id: k.id,
                    output: k.output,
                    sheet: k.sheet,
                    faces: k
                        .faces
                        .iter()
                        .map(|f| RFace {
                            origins: origins(&f.origins),
                            kept: f.kept,
                            surface: f.surface,
                            geom: f.geom.clone(),
                        })
                        .collect(),
                    edges: k
                        .edges
                        .iter()
                        .map(|e| REdge {
                            f0: e.f0,
                            f1: e.f1,
                            laminar: e.laminar,
                            geom: e.geom.clone(),
                            origins: origins(&e.origins),
                        })
                        .collect(),
                    vertices: k
                        .vertices
                        .iter()
                        .map(|v| RVertex {
                            edges: v.edges.clone(),
                            point: v.point.clone(),
                        })
                        .collect(),
                })
                .collect(),
        })
    }
    pub fn release(id: u64) {
        b::kernel_release(id)
    }
    pub fn live_bodies() -> u64 {
        b::kernel_live_bodies()
    }
    pub fn ray_hits(
        bodies: &[u64],
        filter: &[i64],
        ray: [f64; 6],
    ) -> Result<Vec<f64>, KernelError> {
        b::kernel_ray_hits(bodies, filter, &ray).map_err(err)
    }
    pub fn extent(bodies: &[u64], axis: [f64; 6]) -> Result<(f64, f64), KernelError> {
        let v = b::kernel_extent(bodies, &axis).map_err(err)?;
        Ok((v[0], v[1]))
    }
    pub fn analyze(kind: u32, reals: &[f64], bodies: &[u64]) -> Result<Vec<f64>, KernelError> {
        b::kernel_analyze(kind, reals, bodies).map_err(err)
    }
    pub fn project(
        bodies: &[u64],
        transforms: &[f64],
        frame: &[f64],
        flags: u32,
        tolerance: f64,
    ) -> Result<Vec<RProjected>, KernelError> {
        Ok(
            b::kernel_project(bodies, transforms, frame, flags, tolerance)
                .map_err(err)?
                .into_iter()
                .map(|p| RProjected {
                    body: p.body,
                    kind: p.kind,
                    entity: p.entity,
                    visible: p.visible,
                    points: p.points,
                })
                .collect(),
        )
    }
    pub fn proximity(a: u64, ta: &[f64], bb: u64, tb: &[f64]) -> Result<Vec<f64>, KernelError> {
        b::kernel_proximity(a, ta, bb, tb).map_err(err)
    }
    pub fn tessellate(id: u64, chord: f64, angle: f64) -> Result<RDisplay, KernelError> {
        let d = b::kernel_tessellate(id, chord, angle).map_err(err)?;
        Ok(RDisplay {
            faces: d
                .faces
                .into_iter()
                .map(|f| (f.positions, f.triangles, f.normals))
                .collect(),
            edges: d.edges.into_iter().map(|e| e.points).collect(),
            vertices: d.vertices,
        })
    }
    pub fn export_step(doc: &[Instr], bodies: &[u64]) -> Result<Vec<u8>, KernelError> {
        b::kernel_export_step(&instrs(doc), bodies).map_err(err)
    }
    pub fn step_products(path: &str) -> Result<Vec<String>, KernelError> {
        b::kernel_step_products(path).map_err(err)
    }
    pub fn version() -> String {
        b::kernel_occt_version()
    }
}

/// The browser host has no exact kernel program interpreter yet: every operation reports so.
#[cfg(not(all(
    any(not(target_arch = "wasm32"), target_os = "emscripten"),
    feature = "occt-kernel"
)))]
mod backend {
    use super::*;
    fn no() -> KernelError {
        KernelError::NotSupported {
            what: "OCCT kernel operations in the browser host".into(),
        }
    }
    pub fn execute(_: &[Instr], _: &[u64]) -> Result<RResult, KernelError> {
        Err(no())
    }
    pub fn release(_: u64) {}
    pub fn live_bodies() -> u64 {
        0
    }
    pub fn ray_hits(_: &[u64], _: &[i64], _: [f64; 6]) -> Result<Vec<f64>, KernelError> {
        Err(no())
    }
    pub fn extent(_: &[u64], _: [f64; 6]) -> Result<(f64, f64), KernelError> {
        Err(no())
    }
    pub fn analyze(_: u32, _: &[f64], _: &[u64]) -> Result<Vec<f64>, KernelError> {
        Err(no())
    }
    pub fn project(
        _: &[u64],
        _: &[f64],
        _: &[f64],
        _: u32,
        _: f64,
    ) -> Result<Vec<RProjected>, KernelError> {
        Err(no())
    }
    pub fn proximity(_: u64, _: &[f64], _: u64, _: &[f64]) -> Result<Vec<f64>, KernelError> {
        Err(no())
    }
    pub fn tessellate(_: u64, _: f64, _: f64) -> Result<RDisplay, KernelError> {
        Err(no())
    }
    pub fn export_step(_: &[Instr], _: &[u64]) -> Result<Vec<u8>, KernelError> {
        Err(no())
    }
    pub fn step_products(_: &str) -> Result<Vec<String>, KernelError> {
        Err(no())
    }
    pub fn version() -> String {
        "browser-host".into()
    }
}

/// Bodies the kernel holds alive (for leak checks).
pub fn occt_kernel_live_bodies() -> u64 {
    backend::live_bodies()
}

mod display;

// ------------------------------------------------------------------------- bodies

/// The payload of a [`BodyHandle`] this kernel made: a registry body, released on drop.
#[derive(Debug)]
pub struct OcctBody {
    id: u64,
    sheet: bool,
    topology: Topology,
}

impl OcctBody {
    pub fn is_sheet(&self) -> bool {
        self.sheet
    }
    pub fn topology(&self) -> &Topology {
        &self.topology
    }
}

impl Drop for OcctBody {
    fn drop(&mut self) {
        backend::release(self.id);
    }
}

/// A named box an import resolves to (test fixtures and simple stock).
#[derive(Clone, Debug, PartialEq)]
pub struct NamedBox {
    pub name: String,
    pub min: [f64; 3],
    pub max: [f64; 3],
    /// Names for the faces in order `-x, +x, -y, +y, -z, +z`.
    pub face_names: [String; 6],
}

/// What an import source resolves to.
#[derive(Clone, Debug, PartialEq)]
pub enum ImportSource {
    /// A STEP file on disk.
    Step(PathBuf),
    /// Named boxes.
    Boxes(Vec<NamedBox>),
}

/// The OCCT kernel.
#[derive(Clone, Debug, Default)]
pub struct OcctKernel {
    /// Imports by source path; a source not listed is read as a STEP file under `root`.
    pub imports: BTreeMap<String, ImportSource>,
    pub root: Option<PathBuf>,
}

fn gen(role: Role, tag: KernelTag, ordinal: Option<u32>) -> HistorySource {
    HistorySource::Generated {
        role,
        tag,
        ordinal,
        seed: None,
        instance: None,
    }
}

fn of_label(l: &FaceLabel) -> HistorySource {
    gen(l.role, l.tag.clone(), l.ordinal)
}

fn not_supported(what: impl Into<String>) -> KernelError {
    KernelError::NotSupported {
        what: format!("OCCT kernel: {}", what.into()),
    }
}

fn failed(message: impl Into<String>) -> KernelError {
    KernelError::Failed {
        message: message.into(),
    }
}

fn frame_reals(f: &Frame) -> Vec<f64> {
    let mut v = Vec::with_capacity(12);
    v.extend_from_slice(&f.origin);
    v.extend_from_slice(&f.x);
    v.extend_from_slice(&f.y);
    v.extend_from_slice(&f.z);
    v
}

fn transform_reals(t: &Transform) -> Vec<f64> {
    let mut v = Vec::with_capacity(12);
    for row in &t.rotation {
        v.extend_from_slice(row);
    }
    v.extend_from_slice(&t.translation);
    v
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}

// ------------------------------------------------------------------------- lowering

/// How the faces of one tool are labelled.
#[derive(Clone)]
struct ToolLabels<'l> {
    profile_index: u32,
    /// Side label for a profile curve; `None`: the default ([`Role::Side`] tagged by curve).
    sides: Option<&'l BTreeMap<SketchEntityId, FaceLabel>>,
    start: HistorySource,
    end: HistorySource,
    /// Applied to every label (pattern seed and instance).
    seed: Option<u32>,
    instance: Option<InstanceKey>,
}

impl<'l> ToolLabels<'l> {
    fn plain(profile_index: u32) -> Self {
        ToolLabels {
            profile_index,
            sides: None,
            start: gen(Role::StartCap, KernelTag::None, None),
            end: gen(Role::EndCap, KernelTag::None, None),
            seed: None,
            instance: None,
        }
    }
    fn side(&self, curve: &SketchEntityId, ordinal: Option<u32>) -> HistorySource {
        if let Some(l) = self.sides.and_then(|m| m.get(curve)) {
            return of_label(l);
        }
        gen(
            Role::Side,
            KernelTag::Curve {
                profile: self.profile_index,
                curve: curve.clone(),
            },
            ordinal,
        )
    }
    fn mark(&self, s: HistorySource) -> HistorySource {
        match s {
            HistorySource::Generated {
                role,
                tag,
                ordinal,
                seed,
                instance,
            } => HistorySource::Generated {
                role,
                tag,
                ordinal,
                seed: self.seed.or(seed),
                instance: self.instance.clone().or(instance),
            },
            other => other,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct OutputSpec {
    /// The body label of a body that holds no operand material.
    body: Option<HistorySource>,
    /// Bodies of a split carry the side of this frame's plane they lie on.
    split: Option<Frame>,
    /// Operands passed back untouched.
    kept: bool,
    /// The operands whose material a body can hold, when not every operand's can (a cut's
    /// tools leave faces on the result but no material, so the result does not continue them).
    material: Option<Vec<u32>>,
}

struct Lower<'a> {
    operands: &'a [&'a OcctBody],
    instrs: Vec<Instr>,
    labels: Vec<HistorySource>,
    label_ids: BTreeMap<HistorySource, i64>,
    instances: Vec<InstanceKey>,
    outputs: Vec<OutputSpec>,
    operand_slots: BTreeMap<u32, i64>,
}

/// A bounded end of a prism, trimmed after the long prism is built.
struct Trim {
    to: TrimTo,
    keep: i64,
    cut: i64,
}

enum TrimTo {
    /// By one operand face's surface, moved by `shift`.
    Face {
        operand: u32,
        face: u32,
        shift: [f64; 3],
    },
    /// Where each ray along `dir` first meets an operand face (up to next over several faces).
    Next { dir: [f64; 3] },
}

/// What the rays of an up-to-next bound meet first.
enum NextHit {
    One(u32, u32),
    Several,
}

impl<'a> Lower<'a> {
    fn new(operands: &'a [&'a OcctBody]) -> Self {
        Lower {
            operands,
            instrs: Vec::new(),
            labels: Vec::new(),
            label_ids: BTreeMap::new(),
            instances: Vec::new(),
            outputs: Vec::new(),
            operand_slots: BTreeMap::new(),
        }
    }

    fn push(&mut self, op: u32, ints: Vec<i64>, reals: Vec<f64>) -> i64 {
        self.instrs.push(Instr {
            op,
            ints,
            reals,
            strs: Vec::new(),
        });
        (self.instrs.len() - 1) as i64
    }

    fn label(&mut self, s: HistorySource) -> i64 {
        if let Some(id) = self.label_ids.get(&s) {
            return *id;
        }
        let id = self.labels.len() as i64;
        self.labels.push(s.clone());
        self.label_ids.insert(s, id);
        id
    }

    fn instance(&mut self, k: &InstanceKey) -> i64 {
        if let Some(i) = self.instances.iter().position(|x| x == k) {
            return i as i64;
        }
        self.instances.push(k.clone());
        (self.instances.len() - 1) as i64
    }

    fn operand(&mut self, i: u32) -> Result<i64, KernelError> {
        if i as usize >= self.operands.len() {
            return Err(failed(format!("operand {i} out of range")));
        }
        if let Some(s) = self.operand_slots.get(&i) {
            return Ok(*s);
        }
        let s = self.push(1, vec![i as i64], vec![]);
        self.operand_slots.insert(i, s);
        Ok(s)
    }

    fn body(&self, i: u32) -> Result<&'a OcctBody, KernelError> {
        self.operands
            .get(i as usize)
            .copied()
            .ok_or_else(|| failed(format!("operand {i} out of range")))
    }

    fn side_labels(&self) -> Vec<i64> {
        self.labels
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                matches!(
                    l,
                    HistorySource::Generated {
                        role: Role::Side,
                        tag: KernelTag::Curve { .. },
                        ..
                    }
                )
            })
            .map(|(i, _)| i as i64)
            .collect()
    }

    fn output(&mut self, slot: i64, merge_collinear: bool, spec: OutputSpec) {
        let sides = self.side_labels();
        let mut ints = vec![slot, merge_collinear as i64, sides.len() as i64];
        ints.extend(sides);
        self.push(22, ints, vec![]);
        self.outputs.push(spec);
    }

    /// Encode a profile. `sides(segment)` labels each segment's swept face; for sheets
    /// `caps(segment)` gives its start- and end-cap boundary labels and `ends` the chain-end
    /// vertex labels per loop.
    #[allow(clippy::type_complexity)]
    fn profile(
        &mut self,
        p: &CanonicalProfile,
        closed: bool,
        sides: &dyn Fn(&ProfileSegment) -> Option<HistorySource>,
        caps: Option<&dyn Fn(&ProfileSegment) -> (HistorySource, HistorySource)>,
        ends: Option<&dyn Fn(&[ProfileSegment]) -> (HistorySource, HistorySource)>,
    ) -> Result<i64, KernelError> {
        if p.loops.is_empty() || p.loops.iter().any(|l| l.is_empty()) {
            return Err(failed("a profile needs loops"));
        }
        let mut ints = vec![closed as i64, p.loops.len() as i64];
        let mut reals = frame_reals(&p.frame);
        for lp in &p.loops {
            ints.push(lp.len() as i64);
            match ends {
                Some(f) => {
                    let (a, b) = f(lp);
                    ints.push(self.label(a));
                    ints.push(self.label(b));
                }
                None => {
                    ints.push(-1);
                    ints.push(-1);
                }
            }
            for seg in lp {
                let kind = match &seg.kind {
                    SegmentKind::Line => {
                        reals.extend_from_slice(&[
                            seg.start[0],
                            seg.start[1],
                            seg.end[0],
                            seg.end[1],
                        ]);
                        0
                    }
                    SegmentKind::Arc { center, clockwise } => {
                        reals.extend_from_slice(&[
                            seg.start[0],
                            seg.start[1],
                            seg.end[0],
                            seg.end[1],
                            center[0],
                            center[1],
                        ]);
                        if *clockwise {
                            2
                        } else {
                            1
                        }
                    }
                    SegmentKind::Circle { center, radius } => {
                        reals.extend_from_slice(&[center[0], center[1], *radius]);
                        3
                    }
                    SegmentKind::Ellipse {
                        center,
                        major,
                        ratio,
                        clockwise,
                    } => {
                        if seg.is_closed_curve() {
                            reals.extend_from_slice(&[
                                center[0], center[1], major[0], major[1], *ratio,
                            ]);
                            5
                        } else {
                            reals.extend_from_slice(&[
                                seg.start[0],
                                seg.start[1],
                                seg.end[0],
                                seg.end[1],
                                center[0],
                                center[1],
                                major[0],
                                major[1],
                                *ratio,
                                *clockwise as i64 as f64,
                            ]);
                            4
                        }
                    }
                    SegmentKind::Spline {
                        degree,
                        poles,
                        weights,
                        knots,
                        multiplicities,
                    } => {
                        if !weights.is_empty() && weights.len() != poles.len() {
                            return Err(failed("a profile spline needs one weight per pole"));
                        }
                        if knots.len() != multiplicities.len() {
                            return Err(failed("a profile spline needs one multiplicity per knot"));
                        }
                        reals.push(*degree as f64);
                        reals.push(poles.len() as f64);
                        for p in poles {
                            reals.extend_from_slice(p);
                        }
                        for i in 0..poles.len() {
                            reals.push(weights.get(i).copied().unwrap_or(1.0));
                        }
                        reals.push(knots.len() as f64);
                        reals.extend_from_slice(knots);
                        reals.extend(multiplicities.iter().map(|m| *m as f64));
                        reals.push(0.0);
                        6
                    }
                };
                ints.push(kind);
                ints.push(match sides(seg) {
                    Some(l) => self.label(l),
                    None => -1,
                });
                match caps {
                    Some(f) => {
                        let (a, b) = f(seg);
                        ints.push(self.label(a));
                        ints.push(self.label(b));
                    }
                    None => {
                        ints.push(-1);
                        ints.push(-1);
                    }
                }
            }
        }
        Ok(self.push(2, ints, reals))
    }

    fn solid_profile(
        &mut self,
        p: &CanonicalProfile,
        labels: &ToolLabels<'_>,
        ordinal: Option<u32>,
    ) -> Result<i64, KernelError> {
        let side = |s: &ProfileSegment| Some(labels.mark(labels.side(&s.curve, ordinal)));
        self.profile(p, true, &side, None, None)
    }

    fn all_operand_ids(&self) -> Vec<u64> {
        self.operands.iter().map(|b| b.id).collect()
    }

    /// The range of `(p − origin)·dir` over every operand, with a margin; a unit range around
    /// the origin when there are none.
    fn reach(&self, origin: [f64; 3], dir: [f64; 3]) -> Result<(f64, f64), KernelError> {
        let ids = self.all_operand_ids();
        if ids.is_empty() {
            return Ok((-1.0, 1.0));
        }
        let (lo, hi) = backend::extent(
            &ids,
            [origin[0], origin[1], origin[2], dir[0], dir[1], dir[2]],
        )?;
        let m = (hi - lo).abs().max(1e-3) * 0.25;
        Ok((lo.min(0.0) - m, hi.max(0.0) + m))
    }

    /// The first operand face (operand, face) a ray meets past `from`.
    fn first_face(
        &self,
        bodies: &[u32],
        origin: [f64; 3],
        dir: [f64; 3],
    ) -> Result<Option<(u32, u32, f64)>, KernelError> {
        let ids: Vec<u64> = bodies
            .iter()
            .map(|i| self.operands[*i as usize].id)
            .collect();
        let hits = backend::ray_hits(
            &ids,
            &[],
            [origin[0], origin[1], origin[2], dir[0], dir[1], dir[2]],
        )?;
        for h in hits.chunks(4) {
            if h[0] > 1e-9 {
                return Ok(Some((bodies[h[1] as usize], h[2] as u32, h[0])));
            }
        }
        Ok(None)
    }

    /// What every profile point's ray from `from` (along the frame's z) along `dir` meets
    /// first among `bodies`.
    fn next_hit(
        &self,
        profile: &CanonicalProfile,
        bodies: &[u32],
        from: f64,
        dir: [f64; 3],
    ) -> Result<NextHit, KernelError> {
        let frame = profile.frame;
        let mut found: Option<(u32, u32)> = None;
        let mut several = false;
        for lp in &profile.loops {
            for seg in lp {
                for q in [seg.start, mid(seg)] {
                    let p = add(frame.to_world(q), scale(frame.z, from));
                    match self.first_face(bodies, p, dir)? {
                        Some((o, f, _)) => match found {
                            None => found = Some((o, f)),
                            Some(x) if x == (o, f) => {}
                            Some(_) => several = true,
                        },
                        None => {
                            return Err(failed("up to next: the profile does not meet a next face"))
                        }
                    }
                }
            }
        }
        let (o, f) = found.ok_or_else(|| failed("up to next: no face"))?;
        Ok(if several {
            NextHit::Several
        } else {
            NextHit::One(o, f)
        })
    }

    // ---- tools

    fn tool(
        &mut self,
        shape: &ToolShape,
        labels: &ToolLabels<'_>,
    ) -> Result<Vec<i64>, KernelError> {
        match shape {
            ToolShape::Prism {
                profile,
                start,
                end,
                start_bound,
                end_bound,
                draft,
                thin,
            } => Ok(vec![self.prism(
                profile,
                *start,
                *end,
                start_bound,
                end_bound,
                draft.map(|d| (d.end, d.start)),
                thin.as_ref(),
                labels,
            )?]),
            ToolShape::Revolve {
                profile,
                axis_point,
                axis_direction,
                angle,
            } => {
                let z = profile.frame.z;
                let off = dot(sub(*axis_point, profile.frame.origin), z);
                if off.abs() > 1e-7 || dot(norm(*axis_direction), z).abs() > 1e-7 {
                    return Err(failed("the revolve axis must lie in the profile plane"));
                }
                let p = self.solid_profile(profile, labels, None)?;
                let sl = self.label(labels.mark(labels.start.clone()));
                let el = self.label(labels.mark(labels.end.clone()));
                let mut r = axis_point.to_vec();
                r.extend_from_slice(&norm(*axis_direction));
                r.push(*angle);
                Ok(vec![self.push(4, vec![p, sl, el], r)])
            }
            ToolShape::Sweep {
                profile,
                path,
                path_curves,
            } => Ok(vec![self.sweep(
                profile,
                path,
                path_curves,
                labels,
                true,
            )?]),
            ToolShape::Loft { profiles } => Ok(vec![self.loft(profiles, labels, true)?]),
            ToolShape::Holes { .. } => self.holes(shape, labels),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prism(
        &mut self,
        profile: &CanonicalProfile,
        start: f64,
        end: f64,
        start_bound: &PrismBound,
        end_bound: &PrismBound,
        draft: Option<(f64, f64)>,
        thin: Option<&runmat_geometry_kernel::op::ThinWallSpec>,
        labels: &ToolLabels<'_>,
    ) -> Result<i64, KernelError> {
        let frame = profile.frame;
        let z = frame.z;
        let end_sense = if end != 0.0 {
            end.signum()
        } else {
            -start.signum()
        };
        let start_sense = if start != 0.0 {
            start.signum()
        } else {
            -end_sense
        };
        let sl = self.label(labels.mark(labels.start.clone()));
        let el = self.label(labels.mark(labels.end.clone()));
        let mut a = start;
        let mut b = end;
        let mut a_label = sl;
        let mut b_label = el;
        let mut trims: Vec<Trim> = Vec::new();
        let combining: Vec<u32> = (0..self.operands.len() as u32).collect();
        // Start bound.
        match start_bound {
            PrismBound::Blind => {}
            PrismBound::ThroughAll => {
                let (lo, hi) = self.reach(frame.origin, z)?;
                a = if start_sense > 0.0 { hi } else { lo };
            }
            PrismBound::FromFace { face } | PrismBound::UpToFace { face, .. } => {
                let (lo, hi) = self.reach(frame.origin, z)?;
                a = if end_sense > 0.0 { lo } else { hi };
                let offset = match start_bound {
                    PrismBound::UpToFace { offset, .. } => *offset,
                    _ => 0.0,
                };
                a_label = INTERNAL;
                trims.push(Trim {
                    to: TrimTo::Face {
                        operand: face.operand,
                        face: face.entity,
                        shift: scale(z, offset * start_sense),
                    },
                    keep: el,
                    cut: sl,
                });
            }
            PrismBound::UpToNext => {
                let (lo, hi) = self.reach(frame.origin, z)?;
                a = if start_sense > 0.0 { hi } else { lo };
                a_label = INTERNAL;
                let dir = scale(z, start_sense);
                let to = match self.next_hit(profile, &combining, 0.0, dir)? {
                    NextHit::One(operand, face) => TrimTo::Face {
                        operand,
                        face,
                        shift: [0.0; 3],
                    },
                    NextHit::Several => TrimTo::Next { dir },
                };
                trims.push(Trim {
                    to,
                    keep: el,
                    cut: sl,
                });
            }
            PrismBound::UpToBody { operand, offset } => {
                // The far cap's distance is known: build from it back to where the start side
                // first meets the body (op 31 with the caps' roles swapped).
                if !end_bound.is_blind() && !matches!(end_bound, PrismBound::ThroughAll)
                    || thin.is_some()
                    || draft.is_some()
                {
                    return Err(not_supported(
                        "an up-to-body start with a bounded far end, thin or draft",
                    ));
                }
                self.body(*operand)?;
                let b = match end_bound {
                    PrismBound::ThroughAll => {
                        let (lo, hi) = self.reach(frame.origin, z)?;
                        if end_sense > 0.0 {
                            hi
                        } else {
                            lo
                        }
                    }
                    _ => end,
                };
                let p = self.solid_profile(profile, labels, None)?;
                let (lo, hi) = self.reach(frame.origin, z)?;
                let far = if start_sense > 0.0 { hi } else { lo };
                return Ok(self.push(31, vec![p, el, sl, *operand as i64], vec![b, far, *offset]));
            }
        }
        let from_face = matches!(start_bound, PrismBound::FromFace { .. });
        match end_bound {
            PrismBound::Blind if from_face => {
                let face = match start_bound {
                    PrismBound::FromFace { face } => *face,
                    _ => unreachable!(),
                };
                let (lo, hi) = self.reach(frame.origin, z)?;
                let far = if end_sense > 0.0 { hi } else { lo };
                b = far + end;
                b_label = INTERNAL;
                trims.push(Trim {
                    to: TrimTo::Face {
                        operand: face.operand,
                        face: face.entity,
                        shift: scale(z, end),
                    },
                    keep: sl,
                    cut: el,
                });
            }
            PrismBound::Blind => {}
            PrismBound::ThroughAll => {
                let (lo, hi) = self.reach(frame.origin, z)?;
                b = if end_sense > 0.0 { hi } else { lo };
            }
            PrismBound::UpToFace { face, offset } => {
                let (lo, hi) = self.reach(frame.origin, z)?;
                b = if end_sense > 0.0 { hi } else { lo };
                b_label = INTERNAL;
                trims.push(Trim {
                    to: TrimTo::Face {
                        operand: face.operand,
                        face: face.entity,
                        shift: scale(z, offset * end_sense),
                    },
                    keep: sl,
                    cut: el,
                });
            }
            PrismBound::UpToNext => {
                let dir = scale(z, end_sense);
                let from = if start_bound.is_blind() { a } else { 0.0 };
                let to = match self.next_hit(profile, &combining, from, dir)? {
                    NextHit::One(operand, face) => TrimTo::Face {
                        operand,
                        face,
                        shift: [0.0; 3],
                    },
                    NextHit::Several => TrimTo::Next { dir },
                };
                let (lo, hi) = self.reach(frame.origin, z)?;
                b = if end_sense > 0.0 { hi } else { lo };
                b_label = INTERNAL;
                trims.push(Trim {
                    to,
                    keep: sl,
                    cut: el,
                });
            }
            PrismBound::UpToBody { operand, offset } => {
                if !trims.is_empty() || thin.is_some() || draft.is_some() {
                    return Err(not_supported(
                        "up to body with other end conditions, thin or draft",
                    ));
                }
                self.body(*operand)?;
                let p = self.solid_profile(profile, labels, None)?;
                let (lo, hi) = self.reach(frame.origin, z)?;
                let far = if end_sense > 0.0 { hi } else { lo };
                return Ok(self.push(31, vec![p, sl, el, *operand as i64], vec![a, far, *offset]));
            }
            PrismBound::FromFace { .. } => {
                return Err(failed("a far cap cannot start from a face"));
            }
        }
        // A drafted prism tapers from the profile plane wherever it is trimmed, so the long
        // drafted prism trimmed by its end faces is the drafted bounded extrude.
        let (de, ds) = draft.unwrap_or((0.0, 0.0));
        let tool = match thin {
            None => {
                let p = self.solid_profile(profile, labels, None)?;
                self.push(3, vec![p, a_label, b_label], vec![a, b, de, ds])
            }
            Some(t) => self.thin(profile, a, b, a_label, b_label, t, (de, ds), labels)?,
        };
        let mut cur = tool;
        for t in trims {
            cur = match t.to {
                TrimTo::Face {
                    operand,
                    face,
                    shift,
                } => {
                    self.body(operand)?;
                    self.push(
                        30,
                        vec![cur, operand as i64, face as i64, t.keep, t.cut],
                        shift.to_vec(),
                    )
                }
                TrimTo::Next { dir } => {
                    let (lo, hi) = self.reach(frame.origin, dir)?;
                    let mut ints = vec![cur, t.keep, t.cut, combining.len() as i64];
                    ints.extend(combining.iter().map(|o| *o as i64));
                    let mut r = dir.to_vec();
                    r.push((hi - lo).abs() * 2.0 + 1.0);
                    self.push(41, ints, r)
                }
            };
        }
        Ok(cur)
    }

    #[allow(clippy::too_many_arguments)]
    fn thin(
        &mut self,
        profile: &CanonicalProfile,
        a: f64,
        b: f64,
        a_label: i64,
        b_label: i64,
        t: &runmat_geometry_kernel::op::ThinWallSpec,
        (de, ds): (f64, f64),
        labels: &ToolLabels<'_>,
    ) -> Result<i64, KernelError> {
        if let Some(chain) = open_chain(profile) {
            if t.cap.is_some() {
                return Err(failed(
                    "a thin wall along an open profile has no ends to cap",
                ));
            }
            return self.open_thin(profile, chain, a, b, a_label, b_label, t, (de, ds), labels);
        }
        let base = self.profile(profile, true, &|_| None, None, None)?;
        let segs: Vec<&ProfileSegment> = profile.loops.iter().flatten().collect();
        let mut outer = vec![base, segs.len() as i64];
        let mut inner = vec![base, segs.len() as i64];
        for s in &segs {
            let o = self.label(labels.mark(labels.side(&s.curve, None)));
            outer.push(o);
            let i = self.label(labels.mark(gen(
                Role::ThinWall,
                KernelTag::Curve {
                    profile: labels.profile_index,
                    curve: s.curve.clone(),
                },
                None,
            )));
            inner.push(i);
        }
        let po = self.push(28, outer, vec![t.outside]);
        let pi = self.push(28, inner, vec![-t.inside]);
        let wall = self.push(3, vec![po, a_label, b_label], vec![a, b, de, ds]);
        let dir = (b - a).signum();
        let (ia, ib, ial, ibl) = match t.cap {
            Some(c) => {
                let l0 = self.label(labels.mark(gen(Role::ThinWall, KernelTag::None, Some(0))));
                let l1 = self.label(labels.mark(gen(Role::ThinWall, KernelTag::None, Some(1))));
                (a + dir * c, b - dir * c, l0, l1)
            }
            None => {
                // The hole runs past both caps. A drafted hole must not cross the profile plane
                // (each side would taper on its own): an end there stays flush with the wall's.
                let over = (b - a).abs() * 0.1 + 1e-3;
                let drafted = de != 0.0 || ds != 0.0;
                // `z` is one end of the wall, `other` the other end.
                let past = |z: f64, other: f64, s: f64| {
                    let p = z + s * over;
                    let crosses = p * other < 0.0 && z * other >= 0.0;
                    if drafted && crosses {
                        z
                    } else {
                        p
                    }
                };
                (past(a, b, -dir), past(b, a, dir), INTERNAL, INTERNAL)
            }
        };
        let hole = self.push(3, vec![pi, ial, ibl], vec![ia, ib, de, ds]);
        Ok(self.push(7, vec![1, 1, wall, 1, hole], vec![]))
    }

    /// A thin wall along an open chain: the chain offset `outside` to its right and `inside`
    /// to its left (as a counter-clockwise closed profile's outside and inside), closed by
    /// straight ends, extruded. Outer walls are [`Role::Side`], inner walls [`Role::ThinWall`]
    /// tagged by curve; the end faces are [`Role::ThinWall`] tagged by the first curve
    /// (ordinal 2) and the last (ordinal 3).
    #[allow(clippy::too_many_arguments)]
    fn open_thin(
        &mut self,
        profile: &CanonicalProfile,
        chain: &[ProfileSegment],
        a: f64,
        b: f64,
        a_label: i64,
        b_label: i64,
        t: &runmat_geometry_kernel::op::ThinWallSpec,
        (de, ds): (f64, f64),
        labels: &ToolLabels<'_>,
    ) -> Result<i64, KernelError> {
        if t.inside + t.outside <= 0.0 {
            return Err(failed("a thin wall needs a thickness"));
        }
        let right = offset_chain(chain, -t.outside)?;
        let left = offset_chain(chain, t.inside)?;
        let pi = labels.profile_index;
        let curve_tag = |c: &SketchEntityId| KernelTag::Curve {
            profile: pi,
            curve: c.clone(),
        };
        let first = &chain[0].curve;
        let last = &chain[chain.len() - 1].curve;
        let mut segs: Vec<ProfileSegment> = Vec::new();
        let mut sides: Vec<HistorySource> = Vec::new();
        for s in &right {
            segs.push(s.clone());
            sides.push(labels.mark(labels.side(&s.curve, None)));
        }
        let end_a = right.last().expect("chain").end;
        let end_b = left.last().expect("chain").end;
        segs.push(ProfileSegment {
            curve: last.clone(),
            kind: SegmentKind::Line,
            start: end_a,
            end: end_b,
        });
        sides.push(labels.mark(gen(Role::ThinWall, curve_tag(last), Some(3))));
        for s in left.iter().rev() {
            let mut r = s.clone();
            std::mem::swap(&mut r.start, &mut r.end);
            if let SegmentKind::Arc { clockwise, .. } = &mut r.kind {
                *clockwise = !*clockwise;
            }
            segs.push(r);
            sides.push(labels.mark(gen(Role::ThinWall, curve_tag(&s.curve), None)));
        }
        segs.push(ProfileSegment {
            curve: first.clone(),
            kind: SegmentKind::Line,
            start: left[0].start,
            end: right[0].start,
        });
        sides.push(labels.mark(gen(Role::ThinWall, curve_tag(first), Some(2))));
        let section = CanonicalProfile {
            frame: profile.frame,
            loops: vec![segs],
            merge_collinear: false,
        };
        let at = std::cell::Cell::new(0usize);
        let side = |_: &ProfileSegment| {
            let i = at.get();
            at.set(i + 1);
            sides.get(i).cloned()
        };
        let p = self.profile(&section, true, &side, None, None)?;
        Ok(self.push(3, vec![p, a_label, b_label], vec![a, b, de, ds]))
    }

    fn sweep(
        &mut self,
        profile: &CanonicalProfile,
        path: &[[f64; 3]],
        path_curves: &[u32],
        labels: &ToolLabels<'_>,
        solid: bool,
    ) -> Result<i64, KernelError> {
        if path.len() < 2 {
            return Err(failed("a sweep path needs two points"));
        }
        // Group path segments by path curve.
        let curve_of = |i: usize| -> u32 {
            if path_curves.is_empty() {
                i as u32
            } else {
                path_curves.get(i).copied().unwrap_or(i as u32)
            }
        };
        let mut groups: Vec<(u32, usize, usize)> = Vec::new();
        for i in 0..path.len() - 1 {
            let c = curve_of(i);
            match groups.last_mut() {
                Some(g) if g.0 == c => g.2 = i + 1,
                _ => groups.push((c, i, i + 1)),
            }
        }
        let several = groups.len() > 1;
        let n = groups.len();
        let mut pieces = Vec::new();
        for (gi, (c, i0, i1)) in groups.iter().enumerate() {
            let shift = sub(path[*i0], path[0]);
            let mut p = profile.clone();
            p.frame.origin = add(p.frame.origin, shift);
            let ord = several.then_some(*c);
            let slot = if solid {
                self.solid_profile(&p, labels, ord)?
            } else {
                self.sheet_profile(&p, labels, ord)?
            };
            let sl = if gi == 0 {
                self.label(labels.mark(labels.start.clone()))
            } else {
                INTERNAL
            };
            let el = if gi + 1 == n {
                self.label(labels.mark(labels.end.clone()))
            } else {
                INTERNAL
            };
            let mut r = Vec::new();
            for q in &path[*i0..=*i1] {
                r.extend_from_slice(q);
            }
            pieces.push(self.push(5, vec![slot, sl, el], r));
        }
        if pieces.len() == 1 {
            return Ok(pieces[0]);
        }
        let mut ints = vec![0, 1, pieces[0], (pieces.len() - 1) as i64];
        ints.extend_from_slice(&pieces[1..]);
        Ok(self.push(7, ints, vec![]))
    }

    fn loft(
        &mut self,
        profiles: &[CanonicalProfile],
        labels: &ToolLabels<'_>,
        solid: bool,
    ) -> Result<i64, KernelError> {
        let first = profiles
            .first()
            .ok_or_else(|| failed("a loft needs profiles"))?;
        if profiles.len() < 2 {
            return Err(failed("a loft needs two sections"));
        }
        let several = profiles.len() > 2;
        if profiles.iter().any(|p| p.loops.len() != first.loops.len()) {
            return Err(failed("loft sections need the same number of loops"));
        }
        // Sections with fewer curves in a loop are refined to the most curves any section has
        // there: their longest segments split at the middle, longest first (the earliest of
        // equal ones). Sides are tagged by the curves of the first section with the most curves
        // in each loop (the first section when every count agrees), by span.
        let counts: Vec<usize> = (0..first.loops.len())
            .map(|li| {
                profiles
                    .iter()
                    .map(|p| p.loops[li].len())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let names: Vec<Vec<SketchEntityId>> = (0..first.loops.len())
            .map(|li| {
                let fine = profiles
                    .iter()
                    .find(|p| p.loops[li].len() == counts[li])
                    .expect("a section with the most curves");
                fine.loops[li].iter().map(|s| s.curve.clone()).collect()
            })
            .collect();
        let mut refined: Vec<CanonicalProfile> = Vec::new();
        for p in profiles {
            let mut q = p.clone();
            for (li, l) in q.loops.iter_mut().enumerate() {
                *l = refine_loop(l, counts[li])?;
            }
            refined.push(q);
        }
        let mut slots = Vec::new();
        for (k, p) in refined.iter().enumerate() {
            let ord = several.then_some(k as u32);
            let mut q = p.clone();
            for (li, l) in q.loops.iter_mut().enumerate() {
                for (si, s) in l.iter_mut().enumerate() {
                    s.curve = names[li][si].clone();
                }
            }
            slots.push(if solid {
                self.solid_profile(&q, labels, ord)?
            } else {
                self.sheet_profile(&q, labels, ord)?
            });
        }
        let sl = self.label(labels.mark(labels.start.clone()));
        let el = self.label(labels.mark(labels.end.clone()));
        let mut ints = vec![sl, el, slots.len() as i64];
        ints.extend(slots);
        Ok(self.push(6, ints, vec![]))
    }

    fn holes(
        &mut self,
        shape: &ToolShape,
        labels: &ToolLabels<'_>,
    ) -> Result<Vec<i64>, KernelError> {
        let ToolShape::Holes {
            frame,
            positions,
            diameter,
            end,
            counterbore,
            countersink,
            drill_point,
            taper,
            thread,
            slot,
            stages,
        } = shape
        else {
            unreachable!("holes only")
        };
        let down = scale(frame.z, -1.0);
        let all: Vec<u32> = (0..self.operands.len() as u32).collect();
        let mut out = Vec::new();
        for (key, at) in positions {
            let origin = frame.to_world(*at);
            let depth = match end {
                HoleDepth::Blind { depth } => *depth,
                HoleDepth::ThroughAll => -self.reach(origin, frame.z)?.0,
                HoleDepth::UpToNext => self
                    .first_face(&all, origin, down)?
                    .map(|h| h.2)
                    .ok_or_else(|| failed(format!("hole `{key}` meets no face below its start")))?,
                HoleDepth::UpToFace { face, offset } => {
                    let b = self.body(face.operand)?;
                    let hits = backend::ray_hits(
                        &[b.id],
                        &[0, face.entity as i64],
                        [origin[0], origin[1], origin[2], down[0], down[1], down[2]],
                    )?;
                    hits.chunks(4)
                        .find(|h| h[0] > 1e-9)
                        .map(|h| h[0] + offset)
                        .ok_or_else(|| {
                            failed(format!("hole `{key}` does not reach its end face"))
                        })?
                }
            };
            if depth <= 0.0 {
                return Err(failed(format!("hole `{key}` has no depth")));
            }
            if let Some((length, angle)) = slot {
                out.extend(self.slot(
                    key,
                    frame,
                    *at,
                    *diameter,
                    *length,
                    *angle,
                    depth,
                    *counterbore,
                    labels,
                )?);
                continue;
            }
            let mut spans = Vec::new();
            for st in stages {
                let b = self.body(st.operand)?;
                let hits = backend::ray_hits(
                    &[b.id],
                    &[],
                    [origin[0], origin[1], origin[2], down[0], down[1], down[2]],
                )?;
                let exit = hits
                    .chunks(4)
                    .filter(|h| h[0] > 1e-9 && h[3] == 0.0)
                    .map(|h| h[0])
                    .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.max(t))))
                    .ok_or_else(|| {
                        failed(format!(
                            "hole `{key}` does not cross series body {}",
                            st.operand
                        ))
                    })?;
                spans.push(StageSpan {
                    exit: exit.min(depth),
                    radius: st.diameter / 2.0,
                    threaded: st.thread,
                });
            }
            let section = hole_section(
                key,
                diameter / 2.0,
                -depth,
                *counterbore,
                *countersink,
                *drill_point,
                *taper,
                thread.map(|t| t.length),
                &spans,
            );
            // The section's frame: x radial (the hole frame's x), y up the axis.
            let sf = Frame {
                origin,
                x: frame.x,
                y: frame.z,
                z: scale(frame.y, -1.0),
            };
            let segs: Vec<ProfileSegment> = section
                .iter()
                .filter(|s| (s.a[0] - s.b[0]).abs() + (s.a[1] - s.b[1]).abs() > 1e-15)
                .map(|s| ProfileSegment {
                    curve: SketchEntityId::new(""),
                    kind: SegmentKind::Line,
                    start: s.a,
                    end: s.b,
                })
                .collect();
            let lab: Vec<Option<HistorySource>> = section
                .iter()
                .filter(|s| (s.a[0] - s.b[0]).abs() + (s.a[1] - s.b[1]).abs() > 1e-15)
                .map(|s| {
                    // The closing segment lies on the axis and makes no face.
                    if s.a[0].abs() < 1e-15 && s.b[0].abs() < 1e-15 {
                        None
                    } else {
                        Some(labels.mark(s.label.clone()))
                    }
                })
                .collect();
            let prof = CanonicalProfile {
                frame: sf,
                loops: vec![segs.clone()],
                merge_collinear: false,
            };
            let lookup: Vec<(ProfileSegment, Option<HistorySource>)> =
                segs.into_iter().zip(lab).collect();
            let sides = |s: &ProfileSegment| {
                lookup
                    .iter()
                    .find(|(q, _)| q.start == s.start && q.end == s.end)
                    .and_then(|(_, l)| l.clone())
            };
            let p = self.profile(&prof, true, &sides, None, None)?;
            let mut r = origin.to_vec();
            r.extend_from_slice(&frame.z);
            r.push(std::f64::consts::TAU);
            out.push(self.push(4, vec![p, -1, -1], r));
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn slot(
        &mut self,
        key: &str,
        frame: &Frame,
        at: [f64; 2],
        diameter: f64,
        length: f64,
        angle: f64,
        depth: f64,
        counterbore: Option<(f64, f64)>,
        labels: &ToolLabels<'_>,
    ) -> Result<Vec<i64>, KernelError> {
        let (s, c) = angle.sin_cos();
        let u = [c, s];
        let k = || KernelTag::Key { key: key.into() };
        let obround = |w: f64| -> Vec<ProfileSegment> {
            let a = [at[0] - u[0] * length / 2.0, at[1] - u[1] * length / 2.0];
            let b = [at[0] + u[0] * length / 2.0, at[1] + u[1] * length / 2.0];
            let v = [-u[1] * w, u[0] * w];
            let p0 = [a[0] - v[0], a[1] - v[1]];
            let p1 = [b[0] - v[0], b[1] - v[1]];
            let p2 = [b[0] + v[0], b[1] + v[1]];
            let p3 = [a[0] + v[0], a[1] + v[1]];
            let seg = |i: u32, kind: SegmentKind, s: [f64; 2], e: [f64; 2]| ProfileSegment {
                curve: SketchEntityId::new(format!("{i}")),
                kind,
                start: s,
                end: e,
            };
            vec![
                seg(0, SegmentKind::Line, p0, p1),
                seg(
                    1,
                    SegmentKind::Arc {
                        center: b,
                        clockwise: false,
                    },
                    p1,
                    p2,
                ),
                seg(2, SegmentKind::Line, p2, p3),
                seg(
                    3,
                    SegmentKind::Arc {
                        center: a,
                        clockwise: false,
                    },
                    p3,
                    p0,
                ),
            ]
        };
        let mut out = Vec::new();
        let one = |me: &mut Self,
                   w: f64,
                   role: Role,
                   h: f64,
                   caps: [HistorySource; 2]|
         -> Result<i64, KernelError> {
            let prof = CanonicalProfile {
                frame: *frame,
                loops: vec![obround(w)],
                merge_collinear: false,
            };
            let sides = |s: &ProfileSegment| {
                let i: u32 = s.curve.as_str().parse().unwrap_or(0);
                Some(labels.mark(gen(role, k(), Some(i))))
            };
            let p = me.profile(&prof, true, &sides, None, None)?;
            let sl = me.label(labels.mark(caps[0].clone()));
            let el = me.label(labels.mark(caps[1].clone()));
            Ok(me.push(3, vec![p, sl, el], vec![0.0, -h, 0.0, 0.0]))
        };
        out.push(one(
            self,
            diameter / 2.0,
            Role::HoleWall,
            depth,
            [
                gen(Role::HoleBottom, k(), Some(1)),
                gen(Role::HoleBottom, k(), None),
            ],
        )?);
        if let Some((d, h)) = counterbore {
            out.push(one(
                self,
                d / 2.0,
                Role::HoleCounterboreWall,
                h,
                [
                    gen(Role::HoleBottom, k(), Some(1)),
                    gen(Role::HoleCounterbore, k(), None),
                ],
            )?);
        }
        Ok(out)
    }

    /// A sheet profile: every segment's swept face, boundary edges at its caps and chain ends.
    fn sheet_profile(
        &mut self,
        p: &CanonicalProfile,
        labels: &ToolLabels<'_>,
        ordinal: Option<u32>,
    ) -> Result<i64, KernelError> {
        let pi = labels.profile_index;
        let side = |s: &ProfileSegment| Some(labels.mark(labels.side(&s.curve, ordinal)));
        let bnd = |c: &SketchEntityId, o: u32| {
            labels.mark(gen(
                Role::Boundary,
                KernelTag::Curve {
                    profile: pi,
                    curve: c.clone(),
                },
                Some(o),
            ))
        };
        let caps = |s: &ProfileSegment| (bnd(&s.curve, 0), bnd(&s.curve, 1));
        let ends = |l: &[ProfileSegment]| {
            (
                bnd(&l.first().expect("loop").curve, 2),
                bnd(&l.last().expect("loop").curve, 3),
            )
        };
        self.profile(p, false, &side, Some(&caps), Some(&ends))
    }

    /// Combine tools with the operands per a tool operation's combine mode. Returns false when
    /// the result is the tools alone (new bodies).
    fn combine(
        &mut self,
        combine: Combine,
        n_comb: usize,
        tools: &[i64],
        merge: bool,
        body: Option<HistorySource>,
    ) -> Result<(), KernelError> {
        if tools.is_empty() {
            return Err(failed("the operation made no tool"));
        }
        let all_tools = if tools.len() == 1 {
            tools[0]
        } else {
            let mut ints = vec![0, 1, tools[0], (tools.len() - 1) as i64];
            ints.extend_from_slice(&tools[1..]);
            self.push(7, ints, vec![])
        };
        let spec = OutputSpec {
            body: Some(body.unwrap_or_else(|| gen(Role::Body, KernelTag::None, None))),
            ..Default::default()
        };
        if combine == Combine::New || n_comb == 0 {
            if (combine == Combine::Cut || combine == Combine::Intersect) && n_comb == 0 {
                return Err(failed("there is no body to combine the tool with"));
            }
            self.output(all_tools, merge, spec);
            return Ok(());
        }
        let op = match combine {
            Combine::Add => 0,
            Combine::Cut => 1,
            Combine::Intersect => 2,
            Combine::New => unreachable!(),
        };
        let mut args = Vec::new();
        for i in 0..n_comb {
            args.push(self.operand(i as u32)?);
        }
        let mut ints = vec![op, args.len() as i64];
        ints.extend(args);
        ints.push(1);
        ints.push(all_tools);
        let r = self.push(7, ints, vec![]);
        self.output(r, merge, spec);
        Ok(())
    }

    fn keep_operands(&mut self, range: std::ops::Range<usize>) -> Result<(), KernelError> {
        for i in range {
            let s = self.operand(i as u32)?;
            self.output(
                s,
                false,
                OutputSpec {
                    kept: true,
                    ..Default::default()
                },
            );
        }
        Ok(())
    }

    fn edge_labels(&mut self, operand: u32, role: Role) -> Result<i64, KernelError> {
        let n = self.body(operand)?.topology.edges.len();
        let mut base = -1;
        for e in 0..n {
            let id = self.label(gen(
                role,
                KernelTag::InputEdge {
                    edge: OperandEntity {
                        operand,
                        entity: e as u32,
                    },
                },
                None,
            ));
            if e == 0 {
                base = id;
            }
            if id != base + e as i64 {
                return Err(failed("labels are not contiguous"));
            }
        }
        Ok(base.max(0))
    }

    fn vertex_labels(&mut self, operand: u32, role: Role) -> Result<i64, KernelError> {
        let n = self.body(operand)?.topology.vertices.len();
        let mut base = -1;
        for v in 0..n {
            let id = self.label(gen(
                role,
                KernelTag::InputVertex {
                    vertex: OperandEntity {
                        operand,
                        entity: v as u32,
                    },
                },
                None,
            ));
            if v == 0 {
                base = id;
            }
            if id != base + v as i64 {
                return Err(failed("labels are not contiguous"));
            }
        }
        Ok(base.max(0))
    }

    fn face_labels(&mut self, operand: u32, role: Role) -> Result<i64, KernelError> {
        let n = self.body(operand)?.topology.faces.len();
        let mut base = -1;
        for f in 0..n {
            let id = self.label(gen(
                role,
                KernelTag::InputFace {
                    face: OperandEntity {
                        operand,
                        entity: f as u32,
                    },
                },
                None,
            ));
            if f == 0 {
                base = id;
            }
            if id != base + f as i64 {
                return Err(failed("labels are not contiguous"));
            }
        }
        Ok(base.max(0))
    }

    fn fillet(
        &mut self,
        edges: &[OperandEntity],
        radius: f64,
        variable: &[(f64, f64)],
    ) -> Result<(), KernelError> {
        let operand = single_operand(edges)?;
        let slot = self.operand(operand)?;
        let eb = self.edge_labels(operand, Role::Blend)?;
        let vb = self.vertex_labels(operand, Role::Blend)?;
        let mut ints = vec![slot, operand as i64, eb, vb, edges.len() as i64];
        ints.extend(edges.iter().map(|e| e.entity as i64));
        let mut reals = vec![radius, variable.len() as f64];
        for (t, r) in variable {
            reals.push(*t);
            reals.push(*r);
        }
        let r = self.push(8, ints, reals);
        self.output(r, false, OutputSpec::default());
        self.keep_others(operand)
    }

    /// Conic and curvature-continuous blends of straight edges between planar faces (op 43).
    fn section_blend(
        &mut self,
        edges: &[OperandEntity],
        radius: f64,
        profile: BlendProfile,
    ) -> Result<(), KernelError> {
        let operand = single_operand(edges)?;
        let slot = self.operand(operand)?;
        let eb = self.edge_labels(operand, Role::Blend)?;
        let (kind, rho) = match profile {
            BlendProfile::Conic { rho } => (0, rho),
            BlendProfile::Curvature => (1, 0.0),
            BlendProfile::Circular => unreachable!("circular blends are fillets"),
        };
        let mut ints = vec![slot, operand as i64, eb, edges.len() as i64];
        ints.extend(edges.iter().map(|e| e.entity as i64));
        ints.push(kind);
        let r = self.push(43, ints, vec![radius, rho]);
        self.output(r, false, OutputSpec::default());
        self.keep_others(operand)
    }

    fn keep_others(&mut self, used: u32) -> Result<(), KernelError> {
        for i in 0..self.operands.len() as u32 {
            if i != used {
                let s = self.operand(i)?;
                self.output(
                    s,
                    false,
                    OutputSpec {
                        kept: true,
                        ..Default::default()
                    },
                );
            }
        }
        Ok(())
    }
}

fn mid(seg: &ProfileSegment) -> [f64; 2] {
    match &seg.kind {
        SegmentKind::Line => [
            (seg.start[0] + seg.end[0]) / 2.0,
            (seg.start[1] + seg.end[1]) / 2.0,
        ],
        SegmentKind::Arc { center, clockwise } => {
            let r =
                ((seg.start[0] - center[0]).powi(2) + (seg.start[1] - center[1]).powi(2)).sqrt();
            let a0 = (seg.start[1] - center[1]).atan2(seg.start[0] - center[0]);
            let a1 = (seg.end[1] - center[1]).atan2(seg.end[0] - center[0]);
            let mut sweep = a1 - a0;
            if *clockwise {
                while sweep >= 0.0 {
                    sweep -= std::f64::consts::TAU;
                }
            } else {
                while sweep <= 0.0 {
                    sweep += std::f64::consts::TAU;
                }
            }
            let a = a0 + sweep / 2.0;
            [center[0] + r * a.cos(), center[1] + r * a.sin()]
        }
        SegmentKind::Circle { center, radius } => [center[0] + radius, center[1]],
        SegmentKind::Ellipse { .. } | SegmentKind::Spline { .. } => seg.point_at(0.5),
    }
}

/// The profile's one loop when it is an open chain (its last curve ends away from where its
/// first starts).
fn open_chain(p: &CanonicalProfile) -> Option<&[ProfileSegment]> {
    let [l] = &p.loops[..] else {
        return None;
    };
    let (f, z) = (l.first()?, l.last()?);
    let gap = ((z.end[0] - f.start[0]).powi(2) + (z.end[1] - f.start[1]).powi(2)).sqrt();
    (gap > 1e-9 && !f.is_closed_curve()).then_some(&l[..])
}

/// A chain offset by `d` to its left (negative: to its right), each curve trimmed or extended
/// to meet its neighbours. Lines and arcs only.
fn offset_chain(chain: &[ProfileSegment], d: f64) -> Result<Vec<ProfileSegment>, KernelError> {
    // Each curve offset on its own.
    let mut out: Vec<ProfileSegment> = Vec::new();
    for s in chain {
        let mut o = s.clone();
        match &s.kind {
            SegmentKind::Line => {
                let u = [s.end[0] - s.start[0], s.end[1] - s.start[1]];
                let l = (u[0] * u[0] + u[1] * u[1]).sqrt();
                if l <= 0.0 {
                    return Err(failed("a profile line has no length"));
                }
                let n = [-u[1] / l * d, u[0] / l * d];
                o.start = [s.start[0] + n[0], s.start[1] + n[1]];
                o.end = [s.end[0] + n[0], s.end[1] + n[1]];
            }
            SegmentKind::Arc { center, clockwise } => {
                // The left of a counter-clockwise arc is towards its centre.
                let r =
                    ((s.start[0] - center[0]).powi(2) + (s.start[1] - center[1]).powi(2)).sqrt();
                let r2 = if *clockwise { r + d } else { r - d };
                if r2 <= 1e-12 {
                    return Err(failed("the thin wall is thicker than an arc's radius"));
                }
                let at = |p: [f64; 2]| {
                    [
                        center[0] + (p[0] - center[0]) * r2 / r,
                        center[1] + (p[1] - center[1]) * r2 / r,
                    ]
                };
                o.start = at(s.start);
                o.end = at(s.end);
            }
            _ => {
                return Err(not_supported(
                    "thin walls along ellipses or splines of open profiles",
                ))
            }
        }
        out.push(o);
    }
    if d == 0.0 {
        return Ok(out);
    }
    // Corners: neighbours meet where their offset curves cross, nearest the naive joint.
    for i in 0..out.len().saturating_sub(1) {
        let (a, b) = (&out[i], &out[i + 1]);
        let gap = ((a.end[0] - b.start[0]).powi(2) + (a.end[1] - b.start[1]).powi(2)).sqrt();
        if gap <= 1e-12 {
            continue;
        }
        let guess = [(a.end[0] + b.start[0]) / 2.0, (a.end[1] + b.start[1]) / 2.0];
        let p = meet(a, b, guess).ok_or_else(|| {
            failed("the thin wall's offset curves do not meet at a corner of the profile")
        })?;
        out[i].end = p;
        out[i + 1].start = p;
    }
    Ok(out)
}

/// The crossing of two offset curves (as whole lines and circles) nearest `guess`.
fn meet(a: &ProfileSegment, b: &ProfileSegment, guess: [f64; 2]) -> Option<[f64; 2]> {
    enum G {
        L([f64; 2], [f64; 2]),
        C([f64; 2], f64),
    }
    let g = |s: &ProfileSegment| match &s.kind {
        SegmentKind::Arc { center, .. } => G::C(
            *center,
            ((s.start[0] - center[0]).powi(2) + (s.start[1] - center[1]).powi(2)).sqrt(),
        ),
        _ => G::L(s.start, [s.end[0] - s.start[0], s.end[1] - s.start[1]]),
    };
    let line_circle = |p: [f64; 2], u: [f64; 2], c: [f64; 2], r: f64| -> Vec<[f64; 2]> {
        let f = [p[0] - c[0], p[1] - c[1]];
        let aa = u[0] * u[0] + u[1] * u[1];
        let bb = 2.0 * (f[0] * u[0] + f[1] * u[1]);
        let cc = f[0] * f[0] + f[1] * f[1] - r * r;
        let disc = bb * bb - 4.0 * aa * cc;
        if disc < 0.0 || aa == 0.0 {
            return Vec::new();
        }
        let q = disc.sqrt();
        [(-bb - q) / (2.0 * aa), (-bb + q) / (2.0 * aa)]
            .iter()
            .map(|t| [p[0] + t * u[0], p[1] + t * u[1]])
            .collect()
    };
    let cands: Vec<[f64; 2]> = match (g(a), g(b)) {
        (G::L(p, u), G::L(q, v)) => {
            let den = u[0] * v[1] - u[1] * v[0];
            if den.abs() < 1e-15 {
                return None;
            }
            let w = [q[0] - p[0], q[1] - p[1]];
            let s = (w[0] * v[1] - w[1] * v[0]) / den;
            vec![[p[0] + s * u[0], p[1] + s * u[1]]]
        }
        (G::L(p, u), G::C(c, r)) | (G::C(c, r), G::L(p, u)) => line_circle(p, u, c, r),
        (G::C(c1, r1), G::C(c2, r2)) => {
            let dx = [c2[0] - c1[0], c2[1] - c1[1]];
            let dd = (dx[0] * dx[0] + dx[1] * dx[1]).sqrt();
            if dd < 1e-15 || dd > r1 + r2 || dd < (r1 - r2).abs() {
                return None;
            }
            let aa = (r1 * r1 - r2 * r2 + dd * dd) / (2.0 * dd);
            let h = (r1 * r1 - aa * aa).max(0.0).sqrt();
            let m = [c1[0] + aa * dx[0] / dd, c1[1] + aa * dx[1] / dd];
            vec![
                [m[0] - h * dx[1] / dd, m[1] + h * dx[0] / dd],
                [m[0] + h * dx[1] / dd, m[1] - h * dx[0] / dd],
            ]
        }
    };
    cands.into_iter().min_by(|x, y| {
        let dx = (x[0] - guess[0]).powi(2) + (x[1] - guess[1]).powi(2);
        let dy = (y[0] - guess[0]).powi(2) + (y[1] - guess[1]).powi(2);
        dx.partial_cmp(&dy).unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// How the faces of a direct edit move: each by an offset along its normal, or each turned
/// about its own hinge line.
enum FacePlan {
    Offset(Vec<f64>),
    Hinge(Vec<Hinge>),
}

/// A planar face turned by `angle` about the line where its plane meets the plane through
/// `point` normal to `pull` (the pull lies in the face's plane, across the line).
struct Hinge {
    point: [f64; 3],
    pull: [f64; 3],
    angle: f64,
}

/// A planar face's unit normal and centroid; `what` names the unsupported case otherwise.
fn plane_of(t: &Topology, face: u32, what: &str) -> Result<([f64; 3], [f64; 3]), KernelError> {
    let f = t
        .faces
        .get(face as usize)
        .ok_or_else(|| failed("face out of range"))?;
    match f.geom.surface {
        SurfaceKind::Plane { normal } => Ok((norm(normal), f.geom.centroid)),
        _ => Err(not_supported(what)),
    }
}

impl Lower<'_> {
    /// One instruction moving `faces` of `operand` per the plan; `labels` (per face) replace
    /// the moved faces' origins.
    fn move_faces(
        &mut self,
        slot: i64,
        operand: u32,
        faces: &[OperandEntity],
        plan: &FacePlan,
        labels: Option<&[i64]>,
    ) -> i64 {
        let mut ints = vec![slot, operand as i64, faces.len() as i64];
        ints.extend(faces.iter().map(|f| f.entity as i64));
        let (op, reals) = match plan {
            FacePlan::Offset(d) => {
                if let Some(ls) = labels {
                    ints.extend_from_slice(ls);
                }
                (21, d.clone())
            }
            FacePlan::Hinge(hs) => {
                match labels {
                    Some(ls) => ints.extend_from_slice(ls),
                    None => ints.extend(std::iter::repeat_n(-1, faces.len())),
                }
                let mut r = Vec::new();
                for h in hs {
                    r.extend_from_slice(&h.point);
                    r.extend_from_slice(&norm(h.pull));
                    r.push(h.angle);
                }
                (38, r)
            }
        };
        self.push(op, ints, reals)
    }
}

/// A segment's length in its frame.
fn segment_length(seg: &ProfileSegment) -> f64 {
    match &seg.kind {
        SegmentKind::Circle { radius, .. } => 2.0 * std::f64::consts::PI * radius,
        SegmentKind::Arc { center, clockwise } => {
            let (a0, a1, r) = arc_angles(seg, *center, *clockwise);
            r * (a1 - a0).abs()
        }
        SegmentKind::Line => {
            let d = [seg.end[0] - seg.start[0], seg.end[1] - seg.start[1]];
            (d[0] * d[0] + d[1] * d[1]).sqrt()
        }
        SegmentKind::Ellipse { .. } | SegmentKind::Spline { .. } => seg
            .polyline(256)
            .windows(2)
            .map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt())
            .sum(),
    }
}

/// An arc's start and end angles about its centre, the end past the start in the arc's sense,
/// and its radius.
fn arc_angles(seg: &ProfileSegment, c: [f64; 2], clockwise: bool) -> (f64, f64, f64) {
    let tau = 2.0 * std::f64::consts::PI;
    let a0 = (seg.start[1] - c[1]).atan2(seg.start[0] - c[0]);
    let mut a1 = (seg.end[1] - c[1]).atan2(seg.end[0] - c[0]);
    if clockwise {
        while a1 >= a0 {
            a1 -= tau;
        }
    } else {
        while a1 <= a0 {
            a1 += tau;
        }
    }
    let r = ((seg.start[0] - c[0]).powi(2) + (seg.start[1] - c[1]).powi(2)).sqrt();
    (a0, a1, r)
}

/// A segment split at its middle into two of the same curve.
fn split_segment(seg: &ProfileSegment) -> Result<[ProfileSegment; 2], KernelError> {
    let half = |kind: SegmentKind, a: [f64; 2], b: [f64; 2]| ProfileSegment {
        curve: seg.curve.clone(),
        kind,
        start: a,
        end: b,
    };
    match &seg.kind {
        SegmentKind::Circle { center, radius } => {
            let c = *center;
            let r = *radius;
            let d = [seg.start[0] - c[0], seg.start[1] - c[1]];
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
            let u = if (l - r).abs() <= 1e-9 * r.max(1.0) && l > 0.0 {
                [d[0] / l, d[1] / l]
            } else {
                [1.0, 0.0]
            };
            let p = [c[0] + r * u[0], c[1] + r * u[1]];
            let q = [c[0] - r * u[0], c[1] - r * u[1]];
            let arc = SegmentKind::Arc {
                center: c,
                clockwise: false,
            };
            Ok([half(arc.clone(), p, q), half(arc, q, p)])
        }
        SegmentKind::Arc { center, clockwise } => {
            let (a0, a1, r) = arc_angles(seg, *center, *clockwise);
            let am = 0.5 * (a0 + a1);
            let m = [center[0] + r * am.cos(), center[1] + r * am.sin()];
            Ok([
                half(seg.kind.clone(), seg.start, m),
                half(seg.kind.clone(), m, seg.end),
            ])
        }
        SegmentKind::Line | SegmentKind::Ellipse { .. } => {
            let m = mid(seg);
            Ok([
                half(seg.kind.clone(), seg.start, m),
                half(seg.kind.clone(), m, seg.end),
            ])
        }
        SegmentKind::Spline {
            degree,
            poles,
            weights,
            knots,
            multiplicities,
        } => {
            let [(pa, wa, ka, ma), (pb, wb, kb, mb)] =
                split_spline(*degree, poles, weights, knots, multiplicities)?;
            let spline =
                |poles: Vec<[f64; 2]>, weights, knots, multiplicities| SegmentKind::Spline {
                    degree: *degree,
                    poles,
                    weights,
                    knots,
                    multiplicities,
                };
            let m = *pa.last().expect("poles");
            Ok([
                half(spline(pa, wa, ka, ma), seg.start, m),
                half(spline(pb, wb, kb, mb), m, seg.end),
            ])
        }
    }
}

/// One piece of a split spline: poles, weights, distinct knots and their multiplicities.
type SplinePiece = (Vec<[f64; 2]>, Vec<f64>, Vec<f64>, Vec<u32>);

/// A clamped (rational) B-spline split exactly at the middle of its knot range: the knot is
/// inserted (Boehm) until its multiplicity is the degree, so the pole there is on the curve
/// and the two halves are the same curve.
fn split_spline(
    degree: u32,
    poles: &[[f64; 2]],
    weights: &[f64],
    knots: &[f64],
    multiplicities: &[u32],
) -> Result<[SplinePiece; 2], KernelError> {
    let p = degree as usize;
    let rational = !weights.is_empty();
    if p == 0 || knots.len() != multiplicities.len() || knots.len() < 2 {
        return Err(failed("a profile spline needs one multiplicity per knot"));
    }
    if rational && weights.len() != poles.len() {
        return Err(failed("a rational spline needs one weight per pole"));
    }
    let mut u_all: Vec<f64> = knots
        .iter()
        .zip(multiplicities)
        .flat_map(|(k, m)| std::iter::repeat_n(*k, *m as usize))
        .collect();
    if u_all.len() != poles.len() + p + 1 {
        return Err(failed("a spline's knots do not match its poles and degree"));
    }
    // Homogeneous poles.
    let w = |i: usize| if rational { weights[i] } else { 1.0 };
    let mut pw: Vec<[f64; 3]> = (0..poles.len())
        .map(|i| [poles[i][0] * w(i), poles[i][1] * w(i), w(i)])
        .collect();
    let (lo, hi) = (u_all[p], u_all[u_all.len() - p - 1]);
    let u = 0.5 * (lo + hi);
    let tol = 1e-12 * (hi - lo).abs().max(1.0);
    let mult = |u_all: &[f64]| u_all.iter().filter(|k| (**k - u).abs() <= tol).count();
    while mult(&u_all) < p {
        // The span holding u: u_all[k] <= u < u_all[k + 1].
        let k = (p..u_all.len() - p - 1)
            .rev()
            .find(|&k| u_all[k] <= u + tol)
            .unwrap_or(p);
        let mut q = Vec::with_capacity(pw.len() + 1);
        for i in 0..=pw.len() {
            if i + p <= k {
                q.push(pw[i]);
            } else if i > k {
                q.push(pw[i - 1]);
            } else {
                let a = (u - u_all[i]) / (u_all[i + p] - u_all[i]);
                let (x, y) = (pw[i - 1], pw[i]);
                q.push([
                    (1.0 - a) * x[0] + a * y[0],
                    (1.0 - a) * x[1] + a * y[1],
                    (1.0 - a) * x[2] + a * y[2],
                ]);
            }
        }
        pw = q;
        u_all.insert(k + 1, u);
    }
    let j = u_all
        .iter()
        .position(|k| (*k - u).abs() <= tol)
        .expect("inserted");
    let piece = |range: std::ops::Range<usize>, knots: Vec<f64>| -> SplinePiece {
        let pts = &pw[range];
        let poles = pts.iter().map(|h| [h[0] / h[2], h[1] / h[2]]).collect();
        let weights = if rational {
            pts.iter().map(|h| h[2]).collect()
        } else {
            Vec::new()
        };
        let mut distinct: Vec<f64> = Vec::new();
        let mut counts: Vec<u32> = Vec::new();
        for k in knots {
            match distinct.last() {
                Some(last) if (k - last).abs() <= tol => *counts.last_mut().expect("count") += 1,
                _ => {
                    distinct.push(k);
                    counts.push(1);
                }
            }
        }
        (poles, weights, distinct, counts)
    };
    let mut left: Vec<f64> = u_all[..j].to_vec();
    left.extend(std::iter::repeat_n(u, p + 1));
    let mut right: Vec<f64> = std::iter::repeat_n(u, p + 1).collect();
    right.extend_from_slice(&u_all[j + p..]);
    Ok([piece(0..j, left), piece(j - 1..pw.len(), right)])
}

/// A loop refined to `count` segments by splitting the longest one, repeatedly.
fn refine_loop(l: &[ProfileSegment], count: usize) -> Result<Vec<ProfileSegment>, KernelError> {
    let mut out = l.to_vec();
    while out.len() < count {
        let mut best = 0;
        for (i, s) in out.iter().enumerate() {
            if segment_length(s) > segment_length(&out[best]) + 1e-12 {
                best = i;
            }
        }
        let [a, b] = split_segment(&out[best])?;
        out.splice(best..best + 1, [a, b]);
    }
    Ok(out)
}

fn single_operand(es: &[OperandEntity]) -> Result<u32, KernelError> {
    let first = es
        .first()
        .ok_or_else(|| failed("the operation names no entities"))?;
    if es.iter().any(|e| e.operand != first.operand) {
        return Err(not_supported(
            "one operation over entities of several bodies",
        ));
    }
    Ok(first.operand)
}

// ------------------------------------------------------------------------- the kernel

impl OcctKernel {
    pub fn new() -> Self {
        Self::default()
    }

    /// The OCCT version this backend was built against.
    pub fn occt_version() -> String {
        backend::version()
    }

    fn kernel_id() -> KernelId {
        KernelId {
            name: "occt".into(),
            version: format!("k1/{}", backend::version()),
        }
    }

    fn bodies<'h>(operands: &[&'h BodyHandle]) -> Result<Vec<&'h OcctBody>, KernelError> {
        operands
            .iter()
            .enumerate()
            .map(|(i, h)| {
                h.downcast::<OcctBody>()
                    .ok_or(KernelError::ForeignOperand { operand: i as u32 })
            })
            .collect()
    }

    fn lower(&self, op: &KernelOp, l: &mut Lower<'_>) -> Result<(), KernelError> {
        let n = l.operands.len();
        match op {
            KernelOp::Tool(t) => self.lower_tool(t, l),
            KernelOp::Labeled {
                tools,
                carve,
                combine,
                separate,
            } => self.lower_labeled(tools, carve, *combine, *separate, l),
            KernelOp::Fillet {
                edges,
                radius,
                variable,
                profile,
                setbacks,
            } => {
                if !setbacks.is_empty() {
                    return Err(not_supported("fillet setbacks"));
                }
                if *profile != BlendProfile::Circular {
                    if !variable.is_empty() {
                        return Err(not_supported(
                            "variable conic and curvature-continuous blends",
                        ));
                    }
                    return l.section_blend(edges, *radius, *profile);
                }
                l.fillet(edges, *radius, variable)
            }
            KernelOp::FaceBlend {
                first,
                second,
                radius,
                profile,
            } => {
                let operand = single_operand(&[first.clone(), second.clone()].concat())?;
                let t = &l.body(operand)?.topology;
                let a: BTreeSet<u32> = first.iter().map(|e| e.entity).collect();
                let b: BTreeSet<u32> = second.iter().map(|e| e.entity).collect();
                let edges: Vec<OperandEntity> = t
                    .edges
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| {
                        let [f, g] = e.faces;
                        (a.contains(&f) && b.contains(&g)) || (a.contains(&g) && b.contains(&f))
                    })
                    .map(|(i, _)| OperandEntity {
                        operand,
                        entity: i as u32,
                    })
                    .collect();
                if edges.is_empty() {
                    return Err(not_supported("face blends between faces that do not meet"));
                }
                if *profile != BlendProfile::Circular {
                    return l.section_blend(&edges, *radius, *profile);
                }
                l.fillet(&edges, *radius, &[])
            }
            KernelOp::FullRound {
                first,
                center,
                second,
            } => self.lower_full_round(first, center, second, l),
            KernelOp::Chamfer { edges, size } => {
                let operand = single_operand(edges)?;
                let slot = l.operand(operand)?;
                let eb = l.edge_labels(operand, Role::Chamfer)?;
                let mut ints = vec![slot, operand as i64, eb, edges.len() as i64];
                let mut reals = Vec::new();
                for e in edges {
                    let (mode, a, b, second) = match *size {
                        ChamferSize::Equal { distance } => (0, distance, 0.0, false),
                        ChamferSize::Distances { first, second } => (1, first, second, false),
                        ChamferSize::DistanceAngle {
                            distance,
                            angle,
                            on_second,
                        } => (2, distance, angle, on_second),
                    };
                    ints.extend_from_slice(&[e.entity as i64, mode, second as i64]);
                    reals.push(a);
                    reals.push(b);
                }
                let r = l.push(9, ints, reals);
                l.output(r, false, OutputSpec::default());
                l.keep_others(operand)
            }
            KernelOp::VertexChamfer {
                vertices,
                distances,
            } => {
                let operand = single_operand(vertices)?;
                let slot = l.operand(operand)?;
                let vb = l.vertex_labels(operand, Role::Chamfer)?;
                let mut ints = vec![slot, operand as i64, vb, vertices.len() as i64];
                ints.extend(vertices.iter().map(|v| v.entity as i64));
                let r = l.push(10, ints, distances.to_vec());
                l.output(r, false, OutputSpec::default());
                l.keep_others(operand)
            }
            KernelOp::Shell {
                open_faces,
                thickness,
                faces,
                outward,
            } => {
                let operand = if open_faces.is_empty() && faces.is_empty() {
                    0
                } else {
                    single_operand(
                        &[open_faces.clone(), faces.iter().map(|f| f.0).collect()].concat(),
                    )?
                };
                let slot = l.operand(operand)?;
                let wall = if *outward {
                    Role::ShellOuter
                } else {
                    Role::ShellInner
                };
                let ib = l.face_labels(operand, wall)?;
                let rb = l.face_labels(operand, Role::ShellRim)?;
                let mut ints = vec![
                    slot,
                    operand as i64,
                    ib,
                    rb,
                    *outward as i64,
                    open_faces.len() as i64,
                ];
                ints.extend(open_faces.iter().map(|f| f.entity as i64));
                ints.push(faces.len() as i64);
                ints.extend(faces.iter().map(|f| f.0.entity as i64));
                let mut reals = vec![*thickness];
                reals.extend(faces.iter().map(|f| f.1));
                let r = l.push(11, ints, reals);
                l.output(r, false, OutputSpec::default());
                l.keep_others(operand)
            }
            KernelOp::Draft {
                faces,
                neutral,
                angle,
                parting,
                step,
            } => {
                let operand = single_operand(faces)?;
                if *step {
                    // A step face is needed only where two drafted neighbours hinge at different
                    // heights; where they all hinge at one height the step draft is the
                    // parting-line draft.
                    let t = &l.body(operand)?.topology;
                    let pull = norm(neutral.z);
                    let height = |f: u32| -> Option<f64> {
                        let hs: Vec<f64> = parting
                            .iter()
                            .filter_map(|p| t.edges.get(p.entity as usize))
                            .filter(|e| e.faces.contains(&f))
                            .flat_map(|e| [e.geom.start, e.geom.end])
                            .map(|q| dot(sub(q, neutral.origin), pull))
                            .collect();
                        (!hs.is_empty()).then(|| hs.iter().sum::<f64>() / hs.len() as f64)
                    };
                    let drafted: BTreeSet<u32> = faces.iter().map(|f| f.entity).collect();
                    for e in &t.edges {
                        let [f, g] = e.faces;
                        if f == g || !drafted.contains(&f) || !drafted.contains(&g) {
                            continue;
                        }
                        if let (Some(a), Some(b)) = (height(f), height(g)) {
                            if (a - b).abs() > 1e-9 {
                                return Err(not_supported(
                                    "step faces between drafted faces hinged at different heights",
                                ));
                            }
                        }
                    }
                }
                let slot = l.operand(operand)?;
                let mut ints = vec![slot, operand as i64, faces.len() as i64];
                ints.extend(faces.iter().map(|f| f.entity as i64));
                ints.push(parting.len() as i64);
                ints.extend(parting.iter().map(|e| e.entity as i64));
                let mut reals = frame_reals(neutral);
                reals.push(*angle);
                let r = l.push(12, ints, reals);
                l.output(r, false, OutputSpec::default());
                l.keep_others(operand)
            }
            KernelOp::PatternTools {
                seeds,
                instances,
                varied,
            } => self.lower_pattern_tools(seeds, instances, varied, l),
            KernelOp::PatternBodies { instances } => {
                for i in 0..n as u32 {
                    let s = l.operand(i)?;
                    l.output(
                        s,
                        false,
                        OutputSpec {
                            kept: true,
                            ..Default::default()
                        },
                    );
                }
                for inst in instances {
                    let ii = l.instance(&inst.key);
                    for i in 0..n as u32 {
                        let s = l.operand(i)?;
                        let c = l.push(13, vec![s, ii], transform_reals(&inst.transform));
                        l.output(
                            c,
                            false,
                            OutputSpec {
                                body: Some(HistorySource::Copy {
                                    from: OperandEntity {
                                        operand: i,
                                        entity: 0,
                                    },
                                    instance: inst.key.clone(),
                                }),
                                ..Default::default()
                            },
                        );
                    }
                }
                Ok(())
            }
            KernelOp::Boolean {
                combine,
                keep_tools,
            } => {
                if n < 2 {
                    return Err(failed("a boolean needs a target and a tool"));
                }
                let op = match combine {
                    Combine::Add => 0,
                    Combine::Cut => 1,
                    Combine::Intersect => 2,
                    Combine::New => return Err(failed("a boolean cannot make a new body")),
                };
                let target = l.operand(0)?;
                let mut tools = Vec::new();
                for i in 1..n as u32 {
                    tools.push(l.operand(i)?);
                }
                let mut ints = vec![op, 1, target, tools.len() as i64];
                ints.extend(&tools);
                let r = l.push(7, ints, vec![]);
                l.output(
                    r,
                    false,
                    OutputSpec {
                        material: (op == 1).then(|| vec![0]),
                        ..OutputSpec::default()
                    },
                );
                if *keep_tools {
                    l.keep_operands(1..n)?;
                }
                Ok(())
            }
            KernelOp::Split { frame } => {
                let pos = l.label(gen(
                    Role::SplitFace,
                    KernelTag::Side {
                        side: PlaneSide::Positive,
                    },
                    None,
                ));
                let neg = l.label(gen(
                    Role::SplitFace,
                    KernelTag::Side {
                        side: PlaneSide::Negative,
                    },
                    None,
                ));
                for i in 0..n as u32 {
                    let s = l.operand(i)?;
                    let r = l.push(15, vec![s, pos, neg], frame_reals(frame));
                    l.output(
                        r,
                        false,
                        OutputSpec {
                            split: Some(*frame),
                            ..Default::default()
                        },
                    );
                }
                Ok(())
            }
            KernelOp::Transform { transform } => {
                for i in 0..n as u32 {
                    let s = l.operand(i)?;
                    let r = l.push(13, vec![s, -1], transform_reals(transform));
                    l.output(r, false, OutputSpec::default());
                }
                Ok(())
            }
            KernelOp::Scale { center, factors } => {
                for i in 0..n as u32 {
                    let s = l.operand(i)?;
                    let mut reals = center.to_vec();
                    reals.extend_from_slice(factors);
                    let r = l.push(14, vec![s], reals);
                    l.output(r, false, OutputSpec::default());
                }
                Ok(())
            }
            KernelOp::Import { source, digest } => self.lower_import(source, digest, l),
            KernelOp::MoveFaces { faces, motion } => {
                let operand = single_operand(faces)?;
                let plan = match motion {
                    FaceMotion::Offset { distance } => {
                        FacePlan::Offset(vec![*distance; faces.len()])
                    }
                    FaceMotion::Translate { vector } => {
                        // Each planar face's plane moves by the translation's part along its
                        // normal; the part along the face slides it within its own plane,
                        // where its neighbours bound it.
                        let t = &l.body(operand)?.topology;
                        let mut d = Vec::new();
                        for f in faces {
                            let (n, _) = plane_of(t, f.entity, "translating curved faces")?;
                            d.push(dot(n, *vector));
                        }
                        FacePlan::Offset(d)
                    }
                    FaceMotion::Rotate {
                        axis_point,
                        axis_direction,
                        angle,
                    } => {
                        let t = &l.body(operand)?.topology;
                        let a = norm(*axis_direction);
                        let mut hinges = Vec::new();
                        for f in faces {
                            let (n, p0) = plane_of(t, f.entity, "rotating curved faces")?;
                            if dot(n, a).abs() > 1e-9 {
                                return Err(not_supported(
                                    "rotating a face about an axis that crosses its plane",
                                ));
                            }
                            // The turned plane meets the face's plane on a line parallel to
                            // the axis: the face hinges there.
                            let h = dot(n, sub(p0, *axis_point));
                            let d = cross(a, n);
                            let foot = add(*axis_point, scale(n, h));
                            let hinge = add(foot, scale(d, h * (angle / 2.0).tan()));
                            hinges.push(Hinge {
                                point: hinge,
                                pull: d,
                                angle: *angle,
                            });
                        }
                        FacePlan::Hinge(hinges)
                    }
                };
                let slot = l.operand(operand)?;
                let r = l.move_faces(slot, operand, faces, &plan, None);
                l.output(r, false, OutputSpec::default());
                l.keep_others(operand)
            }
            KernelOp::DeleteFaces { faces, heal } => {
                let operand = single_operand(faces)?;
                let slot = l.operand(operand)?;
                let mut ints = vec![slot, operand as i64, faces.len() as i64];
                ints.extend(faces.iter().map(|f| f.entity as i64));
                let r = l.push(if *heal { 19 } else { 20 }, ints, vec![]);
                l.output(r, false, OutputSpec::default());
                l.keep_others(operand)
            }
            KernelOp::ReplaceFaces { faces, with } => {
                // Each planar face moves onto the replacement plane: a parallel face by an
                // offset, a tilted one by turning about the line where the two planes meet.
                if faces.iter().any(|f| f.operand != 0) || with.operand == 0 {
                    return Err(failed(
                        "replace face takes faces of operand 0 and a face of another operand",
                    ));
                }
                let sheet = l.body(with.operand)?;
                let (m, q0) = plane_of(&sheet.topology, with.entity, "replacing by a curved face")?;
                let t = &l.body(0)?.topology;
                let mut offsets = Vec::new();
                let mut hinges = Vec::new();
                for f in faces {
                    let (n, p0) = plane_of(t, f.entity, "replacing a curved face")?;
                    // The replacement's side that faces the same way as the face.
                    let m = if dot(m, n) < 0.0 { scale(m, -1.0) } else { m };
                    let c = dot(m, n).clamp(-1.0, 1.0);
                    let axis = cross(n, m);
                    let s = dot(axis, axis).sqrt();
                    if s < 1e-9 {
                        offsets.push(dot(n, sub(q0, p0)));
                        continue;
                    }
                    let a = scale(axis, 1.0 / s);
                    let angle = s.atan2(c);
                    // A point on both planes: from the face centroid towards the replacement
                    // plane, along the face plane's direction across the axis.
                    let d = cross(a, n);
                    let dist = dot(m, sub(q0, p0));
                    let along = dist / dot(m, d);
                    hinges.push(Hinge {
                        point: add(p0, scale(d, along)),
                        pull: d,
                        angle,
                    });
                }
                if !offsets.is_empty() && !hinges.is_empty() {
                    return Err(not_supported(
                        "replacing faces of which some are parallel to the replacement and some \
                         are not",
                    ));
                }
                let labels: Vec<i64> = faces
                    .iter()
                    .map(|f| {
                        l.label(gen(
                            Role::Replacement,
                            KernelTag::InputFace { face: *f },
                            None,
                        ))
                    })
                    .collect();
                let plan = if hinges.is_empty() {
                    FacePlan::Offset(offsets)
                } else {
                    FacePlan::Hinge(hinges)
                };
                let slot = l.operand(0)?;
                let r = l.move_faces(slot, 0, faces, &plan, Some(&labels));
                l.output(r, false, OutputSpec::default());
                Ok(())
            }
            KernelOp::Surface(s) => self.lower_surface(s, l),
        }
    }

    fn lower_tool(&self, t: &ToolOp, l: &mut Lower<'_>) -> Result<(), KernelError> {
        let n = l.operands.len();
        let n_ref = (t.references as usize).min(n);
        let n_comb = n - n_ref;
        let labels = ToolLabels::plain(0);
        let tools = l.tool(&t.shape, &labels)?;
        let merge = t.shape.merge_collinear();
        l.combine(
            t.combine,
            if t.combine == Combine::New { 0 } else { n_comb },
            &tools,
            merge,
            None,
        )?;
        if t.combine == Combine::New {
            l.keep_operands(0..n)
        } else {
            l.keep_operands(n_comb..n)
        }
    }

    fn lower_labeled(
        &self,
        tools: &[LabeledTool],
        carve: &[LabeledTool],
        combine: Combine,
        separate: bool,
        l: &mut Lower<'_>,
    ) -> Result<(), KernelError> {
        let n = l.operands.len();
        let make = |l: &mut Lower<'_>, t: &LabeledTool| -> Result<i64, KernelError> {
            let sides: BTreeMap<SketchEntityId, FaceLabel> = t.sides.iter().cloned().collect();
            let mut labels = ToolLabels::plain(0);
            labels.sides = Some(&sides);
            if let Some(s) = &t.start {
                labels.start = of_label(s);
            }
            if let Some(e) = &t.end {
                labels.end = of_label(e);
            }
            let parts = l.tool(&t.shape, &labels)?;
            let mut cur = if parts.len() == 1 {
                parts[0]
            } else {
                let mut ints = vec![0, 1, parts[0], (parts.len() - 1) as i64];
                ints.extend_from_slice(&parts[1..]);
                l.push(7, ints, vec![])
            };
            for c in &t.clip {
                let fl = l.label(of_label(&c.label));
                let mut r = c.origin.to_vec();
                r.extend_from_slice(&norm(c.normal));
                cur = l.push(16, vec![cur, fl, -1], r);
            }
            Ok(cur)
        };
        // Carve first: the operands minus the carve tools.
        let mut carved: Vec<i64> = Vec::new();
        let carve_slots: Vec<i64> = carve.iter().map(|t| make(l, t)).collect::<Result<_, _>>()?;
        for i in 0..n as u32 {
            let s = l.operand(i)?;
            if carve_slots.is_empty() {
                carved.push(s);
            } else {
                let mut ints = vec![1, 1, s, carve_slots.len() as i64];
                ints.extend(&carve_slots);
                carved.push(l.push(7, ints, vec![]));
            }
        }
        let tool_slots: Vec<(i64, Option<HistorySource>)> = tools
            .iter()
            .map(|t| Ok((make(l, t)?, t.body.as_ref().map(of_label))))
            .collect::<Result<_, KernelError>>()?;
        let merge = tools.iter().all(|t| t.shape.merge_collinear()) && !tools.is_empty();
        if combine == Combine::New {
            if separate {
                for (s, b) in &tool_slots {
                    l.output(
                        *s,
                        merge,
                        OutputSpec {
                            body: Some(
                                b.clone()
                                    .unwrap_or_else(|| gen(Role::Body, KernelTag::None, None)),
                            ),
                            ..Default::default()
                        },
                    );
                }
            } else if !tool_slots.is_empty() {
                let mut ints = vec![0, 1, tool_slots[0].0, (tool_slots.len() - 1) as i64];
                ints.extend(tool_slots[1..].iter().map(|t| t.0));
                let r = if tool_slots.len() == 1 {
                    tool_slots[0].0
                } else {
                    l.push(7, ints, vec![])
                };
                l.output(
                    r,
                    merge,
                    OutputSpec {
                        body: Some(
                            tool_slots[0]
                                .1
                                .clone()
                                .unwrap_or_else(|| gen(Role::Body, KernelTag::None, None)),
                        ),
                        ..Default::default()
                    },
                );
            }
            for s in carved {
                l.output(
                    s,
                    false,
                    OutputSpec {
                        kept: carve.is_empty(),
                        ..Default::default()
                    },
                );
            }
            return Ok(());
        }
        let op = match combine {
            Combine::Add => 0,
            Combine::Cut => 1,
            Combine::Intersect => 2,
            Combine::New => unreachable!(),
        };
        if tool_slots.is_empty() {
            for s in carved {
                l.output(s, false, OutputSpec::default());
            }
            return Ok(());
        }
        if carved.is_empty() {
            return Err(failed("there is no body to combine the tools with"));
        }
        let mut ints = vec![op, carved.len() as i64];
        ints.extend(&carved);
        ints.push(tool_slots.len() as i64);
        ints.extend(tool_slots.iter().map(|t| t.0));
        let r = l.push(7, ints, vec![]);
        l.output(r, merge, OutputSpec::default());
        Ok(())
    }

    fn lower_full_round(
        &self,
        first: &[OperandEntity],
        center: &[OperandEntity],
        second: &[OperandEntity],
        l: &mut Lower<'_>,
    ) -> Result<(), KernelError> {
        let operand = single_operand(&[first.to_vec(), center.to_vec(), second.to_vec()].concat())?;
        let t = l.body(operand)?.topology.clone();
        // The width between the side faces (parallel planes) sets the radius.
        let plane = |f: &OperandEntity| match t.faces[f.entity as usize].geom.surface {
            SurfaceKind::Plane { normal } => {
                Some((norm(normal), t.faces[f.entity as usize].geom.centroid))
            }
            _ => None,
        };
        let (Some((na, ca)), Some((nb, cb))) = (
            first.first().and_then(plane),
            second.first().and_then(plane),
        ) else {
            return Err(not_supported("full rounds between curved side faces"));
        };
        if dot(na, nb) > -1.0 + 1e-9 {
            return Err(not_supported(
                "full rounds between side faces that are not parallel",
            ));
        }
        let width = dot(sub(cb, ca), na).abs();
        // One planar centre face square to the sides: the round is the half cylinder of radius
        // width / 2 whose axis runs along the centre face, width / 2 below it, midway between
        // the sides. (Two edge fillets of exactly half the width meet and consume the centre
        // face, which OCCT's fillet does not build.)
        let [cf] = center else {
            return Err(not_supported("full rounds over several centre faces"));
        };
        let cg = t.faces[cf.entity as usize].geom;
        let SurfaceKind::Plane { normal } = cg.surface else {
            return Err(not_supported("full rounds over a curved centre face"));
        };
        let nc = norm(normal);
        if dot(nc, na).abs() > 1e-9 {
            return Err(not_supported(
                "full rounds whose centre face is not square to the sides",
            ));
        }
        let r = width / 2.0;
        // The rib lies behind the first side (its outward normal `na`); the mid-plane is r in.
        let mid = dot(ca, na) - r;
        let on_mid = add(cg.centroid, scale(na, mid - dot(cg.centroid, na)));
        let axis = add(on_mid, scale(nc, -r));
        let d = norm(cross(nc, na));
        let ydir = cross(nc, d);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for k in 0..8 {
            let corner = |a: usize| {
                if k & (1 << a) == 0 {
                    cg.bbox_min[a]
                } else {
                    cg.bbox_max[a]
                }
            };
            let s = dot(sub([corner(0), corner(1), corner(2)], axis), d);
            lo = lo.min(s);
            hi = hi.max(s);
        }
        let slot = l.operand(operand)?;
        let lb = l.label(gen(
            Role::Blend,
            KernelTag::InputFace {
                face: OperandEntity {
                    operand,
                    entity: cf.entity,
                },
            },
            None,
        ));
        let mut reals = add(axis, scale(d, lo)).to_vec();
        reals.extend_from_slice(&d);
        reals.extend_from_slice(&ydir);
        reals.extend_from_slice(&nc);
        reals.push(hi - lo);
        reals.push(r);
        let out = l.push(33, vec![slot, lb], reals);
        l.output(out, false, OutputSpec::default());
        l.keep_others(operand)
    }

    fn lower_pattern_tools(
        &self,
        seeds: &[ToolOp],
        instances: &[runmat_geometry_kernel::op::Instance],
        varied: &[Vec<ToolOp>],
        l: &mut Lower<'_>,
    ) -> Result<(), KernelError> {
        if seeds.is_empty() {
            return Err(failed("pattern with no seeds"));
        }
        if !varied.is_empty() && varied.len() != instances.len() {
            return Err(failed("a variable pattern needs tools for every instance"));
        }
        let seed_at = |ii: usize, si: usize| -> &ToolOp {
            if varied.is_empty() {
                &seeds[si]
            } else {
                &varied[ii][si]
            }
        };
        let n = l.operands.len();
        // One combine mode for every seed: the tools of all instances combine at once.
        let combine = seeds[0].combine;
        if combine != Combine::New && seeds.iter().all(|s| s.combine == combine) {
            let mut tools = Vec::new();
            for (ii, inst) in instances.iter().enumerate() {
                let these = if varied.is_empty() {
                    seeds
                } else {
                    &varied[ii]
                };
                for (si, seed) in these.iter().enumerate() {
                    tools.extend(self.instance_tools(seed, si, inst, l)?);
                }
            }
            return l.combine(combine, n, &tools, false, None);
        }
        if varied.iter().any(|v| v.len() != seeds.len()) {
            return Err(failed(
                "a variable pattern's instances need one tool per seed",
            ));
        }
        // Mixed seeds apply in seed order, each over every instance; a new-body seed makes one
        // body per instance and leaves the operands as they are.
        let mut current: Option<i64> = None;
        let mut touched = false;
        for (si, seed) in seeds.iter().enumerate() {
            let mode = seed.combine;
            if (0..instances.len()).any(|ii| seed_at(ii, si).combine != mode) {
                return Err(failed(
                    "a variable pattern's instances must combine as their seed",
                ));
            }
            for (ii, inst) in instances.iter().enumerate() {
                let tools = self.instance_tools(seed_at(ii, si), si, inst, l)?;
                if mode == Combine::New {
                    let all = if tools.len() == 1 {
                        tools[0]
                    } else {
                        let mut ints = vec![0, 1, tools[0], (tools.len() - 1) as i64];
                        ints.extend_from_slice(&tools[1..]);
                        l.push(7, ints, vec![])
                    };
                    let body = HistorySource::Generated {
                        role: Role::Body,
                        tag: KernelTag::None,
                        ordinal: None,
                        seed: Some(si as u32),
                        instance: Some(inst.key.clone()),
                    };
                    l.output(
                        all,
                        false,
                        OutputSpec {
                            body: Some(body),
                            ..Default::default()
                        },
                    );
                    continue;
                }
                if n == 0 && current.is_none() && mode != Combine::Add {
                    return Err(failed("there is no body to combine the tool with"));
                }
                let op = match mode {
                    Combine::Add => 0,
                    Combine::Cut => 1,
                    Combine::Intersect => 2,
                    Combine::New => unreachable!(),
                };
                let args: Vec<i64> = match current {
                    Some(c) => vec![c],
                    None => (0..n as u32)
                        .map(|i| l.operand(i))
                        .collect::<Result<_, _>>()?,
                };
                let r = if args.is_empty() {
                    let mut ints = vec![0, 1, tools[0], (tools.len() - 1) as i64];
                    ints.extend_from_slice(&tools[1..]);
                    l.push(7, ints, vec![])
                } else {
                    let mut ints = vec![op, args.len() as i64];
                    ints.extend(args);
                    ints.push(tools.len() as i64);
                    ints.extend(tools);
                    l.push(7, ints, vec![])
                };
                current = Some(r);
                touched = true;
            }
        }
        match current {
            Some(c) if touched => {
                l.output(
                    c,
                    false,
                    OutputSpec {
                        body: Some(gen(Role::Body, KernelTag::None, None)),
                        ..Default::default()
                    },
                );
                Ok(())
            }
            _ => l.keep_operands(0..n),
        }
    }

    /// One seed's tools placed at one instance, labelled by seed and instance.
    fn instance_tools(
        &self,
        seed: &ToolOp,
        si: usize,
        inst: &runmat_geometry_kernel::op::Instance,
        l: &mut Lower<'_>,
    ) -> Result<Vec<i64>, KernelError> {
        let mut labels = ToolLabels::plain(si as u32);
        labels.seed = Some(si as u32);
        labels.instance = Some(inst.key.clone());
        let mut out = Vec::new();
        for t in l.tool(&seed.shape, &labels)? {
            out.push(l.push(13, vec![t, -1], transform_reals(&inst.transform)));
        }
        Ok(out)
    }

    fn lower_import(
        &self,
        source: &str,
        digest: &str,
        l: &mut Lower<'_>,
    ) -> Result<(), KernelError> {
        let resolved = match self.imports.get(source) {
            Some(s) => s.clone(),
            None => {
                let p = match &self.root {
                    Some(r) => r.join(source),
                    None => PathBuf::from(source),
                };
                ImportSource::Step(p)
            }
        };
        match resolved {
            ImportSource::Boxes(boxes) => {
                for b in &boxes {
                    let mut ints = Vec::new();
                    for i in 0..6 {
                        ints.push(l.label(gen(
                            Role::Side,
                            KernelTag::CadName {
                                name: b.face_names[i].clone(),
                            },
                            None,
                        )));
                    }
                    let mut reals = b.min.to_vec();
                    reals.extend_from_slice(&b.max);
                    let s = l.push(17, ints, reals);
                    l.output(
                        s,
                        false,
                        OutputSpec {
                            body: Some(gen(
                                Role::Body,
                                KernelTag::CadName {
                                    name: b.name.clone(),
                                },
                                None,
                            )),
                            ..Default::default()
                        },
                    );
                }
                Ok(())
            }
            ImportSource::Step(path) => {
                let bytes = std::fs::read(&path)
                    .map_err(|e| failed(format!("cannot read `{source}`: {e}")))?;
                if digest.len() == 64 && digest.bytes().all(|c| c.is_ascii_hexdigit()) {
                    use sha2::Digest as _;
                    let got = hex::encode(sha2::Sha256::digest(&bytes));
                    if !got.eq_ignore_ascii_case(digest) {
                        return Err(failed(format!(
                            "`{source}` changed since its digest was recorded"
                        )));
                    }
                }
                let p = path.to_string_lossy().to_string();
                let products = backend::step_products(&p)?;
                for (i, name) in products.iter().enumerate() {
                    l.instrs.push(Instr {
                        op: 18,
                        ints: vec![i as i64],
                        reals: vec![],
                        strs: vec![p.clone()],
                    });
                    let s = (l.instrs.len() - 1) as i64;
                    l.output(
                        s,
                        false,
                        OutputSpec {
                            body: Some(gen(
                                Role::Body,
                                KernelTag::CadName { name: name.clone() },
                                None,
                            )),
                            ..Default::default()
                        },
                    );
                }
                Ok(())
            }
        }
    }

    fn lower_surface(&self, s: &SurfaceOp, l: &mut Lower<'_>) -> Result<(), KernelError> {
        let n = l.operands.len();
        match s {
            SurfaceOp::Planar { patches } => {
                for (pi, patch) in patches.iter().enumerate() {
                    let face = l.label(of_label(&patch.label));
                    let bnd = |c: &SketchEntityId| {
                        gen(
                            Role::Boundary,
                            KernelTag::Curve {
                                profile: pi as u32,
                                curve: c.clone(),
                            },
                            None,
                        )
                    };
                    let p =
                        l.profile(&patch.profile, true, &|s| Some(bnd(&s.curve)), None, None)?;
                    let r = l.push(26, vec![p, face], vec![]);
                    l.output(r, false, OutputSpec::default());
                }
                l.keep_operands(0..n)
            }
            SurfaceOp::Swept { tool } => {
                let sides: BTreeMap<SketchEntityId, FaceLabel> =
                    tool.sides.iter().cloned().collect();
                let mut labels = ToolLabels::plain(0);
                labels.sides = Some(&sides);
                let slot = match &tool.shape {
                    ToolShape::Prism {
                        profile,
                        start,
                        end,
                        start_bound,
                        end_bound,
                        draft,
                        thin,
                    } => {
                        if !start_bound.is_blind()
                            || !end_bound.is_blind()
                            || draft.is_some()
                            || thin.is_some()
                        {
                            return Err(not_supported(
                                "swept sheets with end conditions, draft or thin",
                            ));
                        }
                        let p = l.sheet_profile(profile, &labels, None)?;
                        l.push(3, vec![p, -1, -1], vec![*start, *end, 0.0, 0.0])
                    }
                    ToolShape::Revolve {
                        profile,
                        axis_point,
                        axis_direction,
                        angle,
                    } => {
                        let p = l.sheet_profile(profile, &labels, None)?;
                        let mut r = axis_point.to_vec();
                        r.extend_from_slice(&norm(*axis_direction));
                        r.push(*angle);
                        l.push(4, vec![p, -1, -1], r)
                    }
                    ToolShape::Sweep {
                        profile,
                        path,
                        path_curves,
                    } => l.sweep(profile, path, path_curves, &labels, false)?,
                    ToolShape::Loft { profiles } => l.loft(profiles, &labels, false)?,
                    ToolShape::Holes { .. } => return Err(not_supported("swept sheets of holes")),
                };
                l.output(
                    slot,
                    false,
                    OutputSpec {
                        body: tool.body.as_ref().map(of_label),
                        ..Default::default()
                    },
                );
                l.keep_operands(0..n)
            }
            SurfaceOp::Knit { solid } => {
                let mut ints = vec![*solid as i64, n as i64];
                for i in 0..n as u32 {
                    ints.push(l.operand(i)?);
                }
                let r = l.push(23, ints, vec![]);
                l.output(r, false, OutputSpec::default());
                Ok(())
            }
            SurfaceOp::Thicken { distance, both } => {
                if n != 1 {
                    return Err(failed("thicken takes one sheet"));
                }
                let slot = l.operand(0)?;
                let fb = l.face_labels(0, Role::ThickenFace)?;
                let rb = l.edge_labels(0, Role::ThickenRim)?;
                let r = l.push(
                    24,
                    vec![slot, 0, fb, rb],
                    vec![*distance, *both as i64 as f64],
                );
                l.output(r, false, OutputSpec::default());
                Ok(())
            }
            SurfaceOp::Trim { frame, keep } => {
                // Keep the side `keep` of the frame's plane: cut away the other half-space.
                let fl = l.label(gen(Role::Boundary, KernelTag::Side { side: *keep }, None));
                let normal = match keep {
                    PlaneSide::Positive => scale(frame.z, -1.0),
                    PlaneSide::Negative => frame.z,
                };
                for i in 0..n as u32 {
                    let s = l.operand(i)?;
                    let mut r = frame.origin.to_vec();
                    r.extend_from_slice(&normal);
                    let c = l.push(16, vec![s, fl, fl], r);
                    l.output(c, false, OutputSpec::default());
                }
                Ok(())
            }
            SurfaceOp::Offset { faces, distance } => {
                // Operands are kept (sheets first, as they are), then one new sheet per operand
                // offset, then the solids.
                let sheets: Vec<u32> = (0..n as u32)
                    .filter(|i| l.operands[*i as usize].sheet)
                    .collect();
                for i in &sheets {
                    l.keep_operands(*i as usize..*i as usize + 1)?;
                }
                for i in 0..n as u32 {
                    let mine: Vec<i64> = faces
                        .iter()
                        .filter(|e| e.operand == i)
                        .map(|e| e.entity as i64)
                        .collect();
                    if !faces.is_empty() && mine.is_empty() {
                        continue;
                    }
                    let s = l.operand(i)?;
                    let ob = l.face_labels(i, Role::OffsetFace)?;
                    let nb = l.face_labels(i, Role::Boundary)?;
                    let none = l.label(gen(Role::Boundary, KernelTag::None, None));
                    let mut ints = vec![s, i as i64, ob, nb, none, mine.len() as i64];
                    ints.extend(mine);
                    let r = l.push(39, ints, vec![*distance]);
                    l.output(
                        r,
                        false,
                        OutputSpec {
                            body: Some(gen(Role::Body, KernelTag::None, None)),
                            material: Some(Vec::new()),
                            ..Default::default()
                        },
                    );
                }
                for i in 0..n as u32 {
                    if !l.operands[i as usize].sheet {
                        l.keep_operands(i as usize..i as usize + 1)?;
                    }
                }
                Ok(())
            }
            SurfaceOp::Extend { edges, distance } => {
                for i in 0..n as u32 {
                    let mine: Vec<&OperandEntity> =
                        edges.iter().filter(|e| e.operand == i).collect();
                    if mine.is_empty() {
                        l.keep_operands(i as usize..i as usize + 1)?;
                        continue;
                    }
                    let s = l.operand(i)?;
                    let mut ints = vec![s, i as i64, mine.len() as i64];
                    for e in mine {
                        let lb = l.label(gen(
                            Role::Extension,
                            KernelTag::InputEdge { edge: *e },
                            None,
                        ));
                        ints.extend_from_slice(&[e.entity as i64, lb]);
                    }
                    let r = l.push(37, ints, vec![*distance]);
                    l.output(r, false, OutputSpec::default());
                }
                Ok(())
            }
            SurfaceOp::MidSurface { pairs } => {
                if pairs.is_empty() {
                    return Err(failed("a mid-surface needs face pairs"));
                }
                if pairs.iter().any(|(a, c)| a.operand != 0 || c.operand != 0) {
                    return Err(failed("mid-surface pairs are faces of operand 0"));
                }
                let nb = l.face_labels(0, Role::Boundary)?;
                let none = l.label(gen(Role::Boundary, KernelTag::None, None));
                let mut ints = vec![0, nb, none, pairs.len() as i64];
                for (a, c) in pairs {
                    let fl = l.label(gen(
                        Role::MidSurface,
                        KernelTag::InputFace { face: *a },
                        None,
                    ));
                    ints.extend_from_slice(&[a.entity as i64, c.entity as i64, fl]);
                }
                let r = l.push(40, ints, vec![]);
                l.output(
                    r,
                    false,
                    OutputSpec {
                        body: Some(gen(Role::Body, KernelTag::None, None)),
                        ..Default::default()
                    },
                );
                l.keep_operands(0..n)
            }
            SurfaceOp::DeleteHole { edges } => {
                // Each sheet with selected edges loses those holes; the others come back as
                // they are.
                for i in 0..n as u32 {
                    let mine: Vec<i64> = edges
                        .iter()
                        .filter(|e| e.operand == i)
                        .map(|e| e.entity as i64)
                        .collect();
                    if mine.is_empty() {
                        l.keep_operands(i as usize..i as usize + 1)?;
                        continue;
                    }
                    let s = l.operand(i)?;
                    let mut ints = vec![s, i as i64, mine.len() as i64];
                    ints.extend(mine);
                    let r = l.push(35, ints, vec![]);
                    l.output(r, false, OutputSpec::default());
                }
                Ok(())
            }
            SurfaceOp::Fill { edges } => {
                // The operands come back as they are, then the patch as a sheet of its own.
                l.keep_operands(0..n)?;
                let face = l.label(gen(Role::SurfacePatch, KernelTag::None, None));
                let mut ints = vec![face, edges.len() as i64];
                for (e, c) in edges {
                    let continuity = match c {
                        runmat_geometry_kernel::names::Continuity::G0 => 0,
                        runmat_geometry_kernel::names::Continuity::G1 => 1,
                        runmat_geometry_kernel::names::Continuity::G2 => 2,
                    };
                    let b = l.label(gen(Role::Boundary, KernelTag::InputEdge { edge: *e }, None));
                    ints.extend_from_slice(&[e.operand as i64, e.entity as i64, continuity, b]);
                }
                let r = l.push(36, ints, vec![]);
                l.output(
                    r,
                    false,
                    OutputSpec {
                        body: Some(gen(Role::Body, KernelTag::None, None)),
                        ..Default::default()
                    },
                );
                Ok(())
            }
            SurfaceOp::Ruled {
                edges,
                distance,
                angle,
            } => {
                // The operands come back as they are, then one sheet of strips.
                l.keep_operands(0..n)?;
                let mut ints = vec![edges.len() as i64];
                for e in edges {
                    let face = l.label(gen(Role::Ruled, KernelTag::InputEdge { edge: *e }, None));
                    ints.extend_from_slice(&[e.operand as i64, e.entity as i64, face]);
                    for k in 0..4 {
                        ints.push(l.label(gen(
                            Role::Boundary,
                            KernelTag::InputEdge { edge: *e },
                            Some(k),
                        )));
                    }
                }
                let r = l.push(34, ints, vec![*distance, *angle]);
                l.output(
                    r,
                    false,
                    OutputSpec {
                        body: Some(gen(Role::Body, KernelTag::None, None)),
                        ..Default::default()
                    },
                );
                Ok(())
            }
            SurfaceOp::CutWithSurface { keep } => {
                if n != 2 {
                    return Err(failed("cut with surface needs a solid and a sheet"));
                }
                let solid = l.operand(0)?;
                let sheet = l.operand(1)?;
                let fl = l.label(gen(Role::SplitFace, KernelTag::Side { side: *keep }, None));
                let r = l.push(
                    32,
                    vec![solid, sheet, (*keep == PlaneSide::Positive) as i64, fl],
                    vec![],
                );
                l.output(r, false, OutputSpec::default());
                l.keep_operands(1..2)
            }
        }
    }

    /// Turn the program's result into bodies and the history record.
    fn assemble(&self, l: &Lower<'_>, r: RResult) -> Result<KernelOutput, KernelError> {
        let source_of = |o: &ROrigin| -> Result<HistorySource, KernelError> {
            Ok(match o.kind {
                ORIGIN_INPUT => HistorySource::Input {
                    from: OperandEntity {
                        operand: o.a,
                        entity: o.b,
                    },
                },
                ORIGIN_LABEL => {
                    if o.a as i64 == INTERNAL {
                        return Err(KernelError::HistoryIncomplete {
                            message: "OCCT kernel: an internal cap survived".into(),
                        });
                    }
                    l.labels
                        .get(o.a as usize)
                        .cloned()
                        .ok_or_else(|| failed("unknown label"))?
                }
                ORIGIN_COPY => HistorySource::Copy {
                    from: OperandEntity {
                        operand: o.a,
                        entity: o.b,
                    },
                    instance: l
                        .instances
                        .get(o.c as usize)
                        .cloned()
                        .ok_or_else(|| failed("unknown instance"))?,
                },
                ORIGIN_NAME => gen(
                    Role::Side,
                    KernelTag::CadName {
                        name: r.names.get(o.a as usize).cloned().unwrap_or_default(),
                    },
                    None,
                ),
                _ => return Err(failed("unknown origin")),
            })
        };
        let mut bodies = Vec::new();
        let mut history = OpHistory::default();
        let mut kept_faces: Vec<bool> = Vec::new();
        for (bi, b) in r.bodies.iter().enumerate() {
            let spec = l
                .outputs
                .get(b.output as usize)
                .cloned()
                .unwrap_or_default();
            let mut topo = Topology::default();
            for f in &b.faces {
                let g = &f.geom;
                let surface = match f.surface {
                    0 => SurfaceKind::Plane {
                        normal: [g[0], g[1], g[2]],
                    },
                    1 => SurfaceKind::Cylinder {
                        axis_point: [g[0], g[1], g[2]],
                        axis_direction: [g[3], g[4], g[5]],
                        radius: g[6],
                    },
                    _ => SurfaceKind::Other,
                };
                topo.faces.push(FaceTopo {
                    geom: FaceGeom {
                        surface,
                        centroid: [g[7], g[8], g[9]],
                        area: g[10],
                        bbox_min: [g[11], g[12], g[13]],
                        bbox_max: [g[14], g[15], g[16]],
                    },
                });
            }
            for e in &b.edges {
                let g = &e.geom;
                topo.edges.push(EdgeTopo {
                    faces: [e.f0, e.f1],
                    geom: EdgeGeom {
                        start: [g[0], g[1], g[2]],
                        end: [g[3], g[4], g[5]],
                        length: g[6],
                    },
                    laminar: e.laminar,
                });
            }
            for v in &b.vertices {
                topo.vertices.push(VertexTopo {
                    edges: v.edges.clone(),
                    point: [v.point[0], v.point[1], v.point[2]],
                });
            }
            // Faces.
            let mut operand_bodies: BTreeSet<u32> = BTreeSet::new();
            let mut copies: BTreeSet<(u32, InstanceKey)> = BTreeSet::new();
            for (fi, f) in b.faces.iter().enumerate() {
                let mut sources: Vec<HistorySource> =
                    f.origins.iter().map(source_of).collect::<Result<_, _>>()?;
                sources.sort();
                sources.dedup();
                for s in &sources {
                    match s {
                        HistorySource::Input { from }
                            if spec
                                .material
                                .as_ref()
                                .is_none_or(|m| m.contains(&from.operand)) =>
                        {
                            operand_bodies.insert(from.operand);
                        }
                        HistorySource::Copy { from, instance } => {
                            copies.insert((from.operand, instance.clone()));
                        }
                        _ => {}
                    }
                }
                kept_faces.push(f.kept || spec.kept);
                history.faces.push(FaceRecord {
                    body: bi as u32,
                    face: fi as u32,
                    evolution: Evolution::Modified,
                    sources,
                });
            }
            // Boundary edges of sheets.
            for (ei, e) in b.edges.iter().enumerate() {
                if !e.laminar {
                    continue;
                }
                let mut sources: Vec<HistorySource> =
                    e.origins.iter().map(source_of).collect::<Result<_, _>>()?;
                sources.sort();
                sources.dedup();
                history.boundaries.push(BoundaryRecord {
                    body: bi as u32,
                    edge: ei as u32,
                    sources,
                });
            }
            // The body.
            let mut bsrc: Vec<HistorySource> = operand_bodies
                .iter()
                .map(|o| HistorySource::Input {
                    from: OperandEntity {
                        operand: *o,
                        entity: 0,
                    },
                })
                .collect();
            if bsrc.is_empty() {
                bsrc.extend(copies.iter().map(|(o, k)| HistorySource::Copy {
                    from: OperandEntity {
                        operand: *o,
                        entity: 0,
                    },
                    instance: k.clone(),
                }));
            }
            if bsrc.is_empty() {
                bsrc.push(
                    spec.body
                        .clone()
                        .unwrap_or_else(|| gen(Role::Body, KernelTag::None, None)),
                );
            }
            if let Some(fr) = spec.split {
                // Which side of the plane the body lies on: any vertex off it decides.
                let mut side = None;
                for v in &topo.vertices {
                    let d = dot(sub(v.point, fr.origin), fr.z);
                    if d.abs() > 1e-9 {
                        side = Some(if d > 0.0 {
                            PlaneSide::Positive
                        } else {
                            PlaneSide::Negative
                        });
                        break;
                    }
                }
                let side =
                    side.ok_or_else(|| failed("a split piece has no extent off the plane"))?;
                bsrc.push(gen(Role::Body, KernelTag::Side { side }, None));
            }
            history.bodies.push(BodyRecord {
                body: bi as u32,
                sources: bsrc,
            });
            bodies.push((b.id, b.sheet, topo));
        }
        // Evolution (as the reference kernel defines it) and deletions.
        let mut uses: BTreeMap<OperandEntity, u32> = BTreeMap::new();
        for rcd in &history.faces {
            for s in &rcd.sources {
                if let HistorySource::Input { from } | HistorySource::Copy { from, .. } = s {
                    *uses.entry(*from).or_default() += 1;
                }
            }
        }
        for (rcd, kept) in history.faces.iter_mut().zip(&kept_faces) {
            let inputs: Vec<&OperandEntity> = rcd
                .sources
                .iter()
                .filter_map(|s| match s {
                    HistorySource::Input { from } => Some(from),
                    _ => None,
                })
                .collect();
            rcd.evolution = if inputs.is_empty() {
                Evolution::Generated
            } else if inputs.len() > 1 || rcd.sources.len() > inputs.len() {
                Evolution::Merged
            } else if uses.get(inputs[0]).copied().unwrap_or(0) > 1 {
                Evolution::Split
            } else if *kept {
                Evolution::Kept
            } else {
                Evolution::Modified
            };
        }
        for (k, o) in l.operands.iter().enumerate() {
            for f in 0..o.topology.faces.len() {
                let oe = OperandEntity {
                    operand: k as u32,
                    entity: f as u32,
                };
                if !uses.contains_key(&oe) {
                    history.deleted_faces.push(oe);
                }
            }
            let used = history.bodies.iter().any(|b| {
                b.sources.iter().any(|s| matches!(s, HistorySource::Input { from } | HistorySource::Copy { from, .. } if from.operand == k as u32))
            });
            if !used {
                history.deleted_bodies.push(k as u32);
            }
        }
        history.deleted_faces.sort();
        history.deleted_bodies.sort();
        let id = Self::kernel_id();
        let out = bodies
            .into_iter()
            .map(|(bid, sheet, topology)| KernelBody {
                handle: BodyHandle::new(
                    id.clone(),
                    OcctBody {
                        id: bid,
                        sheet,
                        topology: topology.clone(),
                    },
                ),
                topology,
            })
            .collect();
        Ok(KernelOutput {
            bodies: out,
            history,
        })
    }
}

impl Kernel for OcctKernel {
    fn id(&self) -> KernelId {
        Self::kernel_id()
    }

    fn execute(
        &self,
        op: &KernelOp,
        operands: &[&BodyHandle],
    ) -> Result<KernelOutput, KernelError> {
        let bodies = Self::bodies(operands)?;
        let mut l = Lower::new(&bodies);
        self.lower(op, &mut l)?;
        let ids: Vec<u64> = bodies.iter().map(|b| b.id).collect();
        let result = backend::execute(&l.instrs, &ids)?;
        // Registry bodies the result made are owned by the handles built in `assemble`; a
        // failure there releases them.
        let made: Vec<u64> = result.bodies.iter().map(|b| b.id).collect();
        match self.assemble(&l, result) {
            Ok(out) => Ok(out),
            Err(e) => {
                for id in made {
                    backend::release(id);
                }
                Err(e)
            }
        }
    }
}

// ------------------------------------------------------------------------- analyses

impl GeometryAnalyzer for OcctKernel {
    fn analyze(
        &self,
        request: &AnalysisRequest,
        handles: &[&BodyHandle],
    ) -> Result<AnalysisResult, KernelError> {
        let bodies = Self::bodies(handles)?;
        let ids: Vec<u64> = bodies.iter().map(|b| b.id).collect();
        let first = bodies.first().ok_or_else(|| failed("no body"))?;
        Ok(match request {
            AnalysisRequest::MassProperties { density } => {
                let v = backend::analyze(0, &[*density], &ids)?;
                let mut inertia = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        inertia[i][j] = v[5 + 3 * i + j];
                    }
                }
                AnalysisResult::MassProperties(MassProperties {
                    volume: v[0],
                    mass: v[1],
                    centroid: [v[2], v[3], v[4]],
                    inertia,
                })
            }
            AnalysisRequest::SectionProperties { origin, normal } => {
                let n = norm(*normal);
                let seed = if n[0].abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let e1 = norm(sub(seed, scale(n, dot(seed, n))));
                let e2 = [
                    n[1] * e1[2] - n[2] * e1[1],
                    n[2] * e1[0] - n[0] * e1[2],
                    n[0] * e1[1] - n[1] * e1[0],
                ];
                let mut r = origin.to_vec();
                r.extend_from_slice(&n);
                r.extend_from_slice(&e1);
                r.extend_from_slice(&e2);
                let v = backend::analyze(1, &r, &ids)?;
                AnalysisResult::SectionProperties(SectionProperties {
                    area: v[0],
                    centroid: [v[1], v[2], v[3]],
                    ixx: v[4],
                    iyy: v[5],
                    ixy: v[6],
                })
            }
            AnalysisRequest::Thickness { min, max } => {
                let v = backend::analyze(2, &[], &ids[..1])?;
                AnalysisResult::Thickness {
                    faces: v
                        .chunks(3)
                        .map(|c| FaceValue {
                            face: c[0] as u32,
                            min: c[1],
                            max: c[2],
                        })
                        .filter(|f| f.min < *min || max.map(|m| f.max > m).unwrap_or(false))
                        .collect(),
                }
            }
            AnalysisRequest::Draft { pull, min } => {
                AnalysisResult::Draft(draft_analysis(&first.topology, *pull, *min))
            }
            AnalysisRequest::Curvature => {
                let v = backend::analyze(3, &[], &ids[..1])?;
                AnalysisResult::Curvature {
                    faces: v
                        .chunks(3)
                        .map(|c| FaceValue {
                            face: c[0] as u32,
                            min: c[1],
                            max: c[2],
                        })
                        .collect(),
                }
            }
            AnalysisRequest::Zebra => {
                let v = backend::analyze(4, &[], &ids[..1])?;
                AnalysisResult::Zebra {
                    edges: v
                        .chunks(2)
                        .map(|c| {
                            (
                                c[0] as u32,
                                match c[1] as u32 {
                                    2 => EdgeContinuity::G2,
                                    1 => EdgeContinuity::G1,
                                    _ => EdgeContinuity::G0,
                                },
                            )
                        })
                        .collect(),
                }
            }
            AnalysisRequest::Deviation => {
                if ids.len() < 2 {
                    return Err(failed("a deviation compares two bodies"));
                }
                let v = backend::analyze(5, &[], &ids[..2])?;
                AnalysisResult::Deviation {
                    max: v[0],
                    rms: v[1],
                }
            }
            AnalysisRequest::Undercut { pull } => {
                let v = backend::analyze(6, &norm(*pull), &ids[..1])?;
                AnalysisResult::Undercut {
                    faces: v.iter().map(|f| *f as u32).collect(),
                }
            }
        })
    }
}

// ------------------------------------------------------------------------- exact queries

impl ExactQueries for OcctKernel {
    fn project(
        &self,
        placed: &[Placed<'_>],
        request: &ProjectionRequest,
    ) -> Result<Vec<ProjectedCurve>, KernelError> {
        let handles: Vec<&BodyHandle> = placed.iter().map(|p| p.body).collect();
        let bodies = Self::bodies(&handles)?;
        let ids: Vec<u64> = bodies.iter().map(|b| b.id).collect();
        let mut transforms = Vec::new();
        for p in placed {
            transforms.extend(transform_reals(&p.transform));
        }
        let flags = (request.hidden as u32) | ((request.smooth as u32) << 1);
        let out = backend::project(
            &ids,
            &transforms,
            &frame_reals(&request.frame),
            flags,
            request.tolerance,
        )?;
        Ok(out
            .into_iter()
            .map(|p| ProjectedCurve {
                source: if p.kind == 0 {
                    ProjectedSource::Edge {
                        body: p.body,
                        edge: p.entity,
                    }
                } else {
                    ProjectedSource::Silhouette {
                        body: p.body,
                        face: p.entity,
                    }
                },
                visible: p.visible,
                polyline: p.points.chunks(2).map(|c| [c[0], c[1]]).collect(),
            })
            .collect())
    }

    fn proximity(&self, a: &Placed<'_>, b: &Placed<'_>) -> Result<Proximity, KernelError> {
        let bodies = Self::bodies(&[a.body, b.body])?;
        let v = backend::proximity(
            bodies[0].id,
            &transform_reals(&a.transform),
            bodies[1].id,
            &transform_reals(&b.transform),
        )?;
        Ok(Proximity {
            distance: v[0],
            common_volume: v[1],
        })
    }

    fn export(
        &self,
        doc: &ExchangeDocument<'_>,
        format: ExchangeFormat,
    ) -> Result<Vec<u8>, KernelError> {
        let mut instrs = vec![Instr {
            op: 1,
            ints: vec![match format {
                ExchangeFormat::Ap242 => 0,
                ExchangeFormat::Ap214 => 1,
            }],
            reals: vec![],
            strs: vec![doc.name.clone()],
        }];
        let mut ids: Vec<u64> = Vec::new();
        for (pi, part) in doc.parts.iter().enumerate() {
            let bodies = Self::bodies(&part.bodies)?;
            let mut ints = vec![bodies.len() as i64];
            for b in &bodies {
                ints.push(ids.len() as i64);
                ids.push(b.id);
            }
            let reals = match part.color {
                Some(c) => vec![1.0, c[0], c[1], c[2]],
                None => vec![0.0, 0.0, 0.0, 0.0],
            };
            instrs.push(Instr {
                op: 2,
                ints,
                reals,
                strs: vec![part.name.clone(), part.material.clone().unwrap_or_default()],
            });
            for f in &part.face_names {
                instrs.push(Instr {
                    op: 3,
                    ints: vec![pi as i64, f.body as i64, f.face as i64],
                    reals: vec![],
                    strs: vec![f.name.clone()],
                });
            }
            for f in &part.face_colors {
                instrs.push(Instr {
                    op: 4,
                    ints: vec![pi as i64, f.body as i64, f.face as i64],
                    reals: f.color.to_vec(),
                    strs: vec![],
                });
            }
        }
        for o in &doc.occurrences {
            instrs.push(Instr {
                op: 5,
                ints: vec![
                    o.part.map(|p| p as i64).unwrap_or(-1),
                    o.parent.map(|p| p as i64).unwrap_or(-1),
                ],
                reals: transform_reals(&o.transform),
                strs: vec![o.name.clone()],
            });
        }
        backend::export_step(&instrs, &ids)
    }
}

/// A shared kernel handle, for callers that hold the kernel behind an `Arc`.
pub type SharedOcctKernel = Arc<OcctKernel>;
