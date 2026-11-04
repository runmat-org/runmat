//! K1 operations the design-layer suites do not reach yet, run on the OCCT kernel directly:
//! each case checks exact geometry (volume, centroid, face planes) and the history contract,
//! and compares history with the reference kernel where the reference kernel builds the case.
//!
//! Exact-vs-faceted differences: the reference kernel works on an axis-aligned grid, so cases
//! it cannot build (tilted planes, sliding faces, curved profiles) are checked against
//! closed-form oracles only.

use crate::kernel_under_test::KernelUnderTest;
use runmat_cad::evaluate::{AnalysisRequest, AnalysisResult, GeometryAnalyzer};
use runmat_cad::regen::kernel::*;
use runmat_cad::regen::FakeKernel;
use runmat_geometry_kernel::frame::Frame;
use runmat_geometry_kernel::history::{
    BoundaryRecord, Evolution, HistorySource, KernelTag, OperandEntity, SurfaceKind,
};
use runmat_geometry_kernel::ids::SketchEntityId;
use runmat_geometry_kernel::names::Role;

pub const L: f64 = 0.1;
pub const W: f64 = 0.05;
pub const H: f64 = 0.01;

pub fn line(c: &str, a: [f64; 2], b: [f64; 2]) -> ProfileSegment {
    ProfileSegment {
        curve: SketchEntityId::new(c),
        kind: SegmentKind::Line,
        start: a,
        end: b,
    }
}

pub fn rect(frame: Frame, x0: f64, y0: f64, x1: f64, y1: f64) -> CanonicalProfile {
    CanonicalProfile {
        frame,
        loops: vec![vec![
            line("c1", [x0, y0], [x1, y0]),
            line("c2", [x1, y0], [x1, y1]),
            line("c3", [x1, y1], [x0, y1]),
            line("c4", [x0, y1], [x0, y0]),
        ]],
        merge_collinear: false,
    }
}

pub fn new_tool(shape: ToolShape) -> KernelOp {
    KernelOp::Tool(ToolOp::new(shape, Combine::New))
}

/// The L × W × H block on the world XY plane.
pub fn block(k: &dyn CadKernel) -> KernelBody {
    let out = k
        .execute(
            &new_tool(ToolShape::prism(rect(Frame::WORLD, 0.0, 0.0, L, W), 0.0, H)),
            &[],
        )
        .expect("block");
    out.bodies.into_iter().next().expect("one body")
}

pub fn side(curve: &str) -> HistorySource {
    HistorySource::Generated {
        role: Role::Side,
        tag: KernelTag::Curve {
            profile: 0,
            curve: SketchEntityId::new(curve),
        },
        ordinal: None,
        seed: None,
        instance: None,
    }
}

pub fn generated(role: Role, tag: KernelTag) -> HistorySource {
    HistorySource::Generated {
        role,
        tag,
        ordinal: None,
        seed: None,
        instance: None,
    }
}

pub fn end_cap() -> HistorySource {
    generated(Role::EndCap, KernelTag::None)
}

pub fn start_cap() -> HistorySource {
    generated(Role::StartCap, KernelTag::None)
}

/// The face of body 0 of a tool output whose only source is `s`.
pub fn face_of(out: &KernelOutput, body: u32, s: &HistorySource) -> u32 {
    out.history
        .faces
        .iter()
        .find(|r| r.body == body && r.sources == vec![s.clone()])
        .unwrap_or_else(|| panic!("no face from {s:?}"))
        .face
}

pub fn entity(operand: u32, entity: u32) -> OperandEntity {
    OperandEntity { operand, entity }
}

pub fn volume(k: &KernelUnderTest, b: &KernelBody) -> (f64, [f64; 3]) {
    match k
        .analyze(
            &AnalysisRequest::MassProperties { density: 1.0 },
            &[&b.handle],
        )
        .expect("mass properties")
    {
        AnalysisResult::MassProperties(m) => (m.volume, m.centroid),
        other => panic!("unexpected {other:?}"),
    }
}

pub fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1e-12)
}

pub fn normal(b: &KernelBody, face: u32) -> [f64; 3] {
    match b.topology.faces[face as usize].geom.surface {
        SurfaceKind::Plane { normal } => normal,
        ref s => panic!("face {face} is not planar: {s:?}"),
    }
}

/// History as sets that do not depend on body, face or edge order.
#[derive(Debug, PartialEq)]
pub struct Shape {
    bodies: Vec<BodyShape>,
    deleted_faces: Vec<OperandEntity>,
    deleted_bodies: Vec<u32>,
}

#[derive(Debug, PartialEq, PartialOrd)]
struct BodyShape {
    sources: Vec<HistorySource>,
    faces: Vec<(Vec<HistorySource>, Evolution)>,
    boundaries: Vec<Vec<HistorySource>>,
}

/// The reference kernel rebuilds every face of a body it modifies from its grid, so it reports
/// `Modified` where the exact kernel returns the untouched face itself (`Kept`).
fn comparable(e: Evolution) -> Evolution {
    match e {
        Evolution::Kept => Evolution::Modified,
        other => other,
    }
}

pub fn history_shape(out: &KernelOutput) -> Shape {
    let mut bodies: Vec<BodyShape> = (0..out.bodies.len() as u32)
        .map(|b| {
            let mut faces: Vec<(Vec<HistorySource>, Evolution)> = out
                .history
                .faces
                .iter()
                .filter(|r| r.body == b)
                .map(|r| (r.sources.clone(), comparable(r.evolution)))
                .collect();
            faces.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
            let mut boundaries: Vec<Vec<HistorySource>> = out
                .history
                .boundaries
                .iter()
                .filter(|r: &&BoundaryRecord| r.body == b)
                .map(|r| r.sources.clone())
                .collect();
            boundaries.sort();
            let sources = out
                .history
                .bodies
                .iter()
                .find(|r| r.body == b)
                .map(|r| r.sources.clone())
                .unwrap_or_default();
            BodyShape {
                sources,
                faces,
                boundaries,
            }
        })
        .collect();
    bodies.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    Shape {
        bodies,
        deleted_faces: out.history.deleted_faces.clone(),
        deleted_bodies: out.history.deleted_bodies.clone(),
    }
}

/// Run `op` on both kernels, operands built by `setup` on each, and require the same history.
pub fn same_history_as_reference(
    setup: &dyn Fn(&dyn CadKernel) -> Vec<KernelBody>,
    op: &KernelOp,
) -> KernelOutput {
    let occt = KernelUnderTest::new();
    let fake = FakeKernel::new();
    let a = setup(&occt);
    let b = setup(&fake);
    let ra: Vec<&BodyHandle> = a.iter().map(|x| &x.handle).collect();
    let rb: Vec<&BodyHandle> = b.iter().map(|x| &x.handle).collect();
    let oa = occt.execute(op, &ra).expect("OCCT");
    let ob = fake.execute(op, &rb).expect("reference");
    assert_eq!(
        history_shape(&oa),
        history_shape(&ob),
        "OCCT and the reference kernel disagree on history"
    );
    oa
}

// ------------------------------------------------------------------- direct face edits

#[test]
fn k1_offset_and_translate_faces_follow_their_normals() {
    let k = KernelUnderTest::new();
    let b = block(&k);
    let top = face_of_block(&k, &end_cap());
    let right = face_of_block(&k, &side("c2"));
    // A translation with parts along and across each face: each plane moves by its normal part.
    let out = k
        .execute(
            &KernelOp::MoveFaces {
                faces: vec![entity(0, top), entity(0, right)],
                motion: FaceMotion::Translate {
                    vector: [0.002, 0.001, 0.003],
                },
            },
            &[&b.handle],
        )
        .expect("translate");
    let body = &out.bodies[0];
    let (v, c) = volume(&k, body);
    assert!(close(v, (L + 0.002) * W * (H + 0.003), 1e-9), "{v}");
    assert!(close(c[0], (L + 0.002) / 2.0, 1e-9));
    assert_eq!(body.topology.faces.len(), 6);
    for s in [end_cap(), side("c2")] {
        let r = out
            .history
            .faces
            .iter()
            .find(|r| {
                matches!(&r.sources[..], [HistorySource::Input { .. }]) && {
                    let HistorySource::Input { from } = &r.sources[0] else {
                        unreachable!()
                    };
                    from.entity == face_of_block(&k, &s)
                }
            })
            .expect("a moved face continues its input");
        assert_eq!(r.evolution, Evolution::Modified);
    }
    // Along the normals only, the reference kernel builds it too: same history.
    same_history_as_reference(
        &|k| vec![block(k)],
        &KernelOp::MoveFaces {
            faces: vec![entity(0, top)],
            motion: FaceMotion::Translate {
                vector: [0.0, 0.0, 0.004],
            },
        },
    );
}

/// The block's face index for a source, from a fresh block.
pub fn face_of_block(k: &dyn CadKernel, s: &HistorySource) -> u32 {
    let out = k
        .execute(
            &new_tool(ToolShape::prism(rect(Frame::WORLD, 0.0, 0.0, L, W), 0.0, H)),
            &[],
        )
        .expect("block");
    face_of(&out, 0, s)
}

#[test]
fn k1_rotate_face_turns_about_the_axis() {
    let k = KernelUnderTest::new();
    let b = block(&k);
    let top = face_of_block(&k, &end_cap());
    let theta = 5f64.to_radians();
    let op = KernelOp::MoveFaces {
        faces: vec![entity(0, top)],
        motion: FaceMotion::Rotate {
            axis_point: [0.0, 0.0, H],
            axis_direction: [1.0, 0.0, 0.0],
            angle: theta,
        },
    };
    let out = k.execute(&op, &[&b.handle]).expect("rotate");
    let body = &out.bodies[0];
    // The top plane z = H + y·tan θ.
    let (v, _) = volume(&k, body);
    let want = L * (H * W + theta.tan() * W * W / 2.0);
    assert!(close(v, want, 1e-7), "{v} vs {want}");
    let moved = out
        .history
        .faces
        .iter()
        .find(|r| {
            r.sources
                == vec![HistorySource::Input {
                    from: entity(0, top),
                }]
        })
        .expect("the rotated face continues its input");
    let n = normal(body, moved.face);
    assert!(
        (n[1] + theta.sin()).abs() < 1e-9 && (n[2] - theta.cos()).abs() < 1e-9,
        "{n:?}"
    );
    // An axis off the face's plane (parallel to it) turns and shifts the plane.
    let out = k
        .execute(
            &KernelOp::MoveFaces {
                faces: vec![entity(0, top)],
                motion: FaceMotion::Rotate {
                    axis_point: [0.0, W / 2.0, 0.0],
                    axis_direction: [1.0, 0.0, 0.0],
                    angle: theta,
                },
            },
            &[&b.handle],
        )
        .expect("rotate about a parallel axis");
    // Rotating the plane z = H about the x axis through (y = W/2, z = 0): the centre line of
    // the top goes to y = W/2 − H sin θ, z = H cos θ; the volume is L·W times the height there.
    let (v, _) = volume(&k, &out.bodies[0]);
    let y0 = W / 2.0 - H * theta.sin();
    let z0 = H * theta.cos();
    let z_mid = z0 + (W / 2.0 - y0) * theta.tan();
    assert!(close(v, L * W * z_mid, 1e-7), "{v}");
    same_history_as_reference(&|k| vec![block(k)], &op);
}

/// A planar sheet: one patch on `frame` over the rectangle, labelled `patch`.
pub fn patch(k: &dyn CadKernel, frame: Frame, x0: f64, y0: f64, x1: f64, y1: f64) -> KernelBody {
    let out = k
        .execute(
            &KernelOp::Surface(SurfaceOp::Planar {
                patches: vec![SurfacePatch {
                    profile: rect(frame, x0, y0, x1, y1),
                    label: FaceLabel::key(Role::SurfacePatch, "patch"),
                }],
            }),
            &[],
        )
        .expect("patch");
    out.bodies.into_iter().next().expect("sheet")
}

pub fn lifted(z: f64) -> Frame {
    Frame {
        origin: [0.0, 0.0, z],
        ..Frame::WORLD
    }
}

#[test]
fn k1_replace_face_moves_it_onto_the_replacement() {
    let k = KernelUnderTest::new();
    let top = face_of_block(&k, &end_cap());
    let op = KernelOp::ReplaceFaces {
        faces: vec![entity(0, top)],
        with: entity(1, 0),
    };
    let out = same_history_as_reference(
        &|k| {
            vec![
                block(k),
                patch(k, lifted(0.015), -0.01, -0.01, L + 0.01, W + 0.01),
            ]
        },
        &op,
    );
    let (v, _) = volume(&k, &out.bodies[0]);
    assert!(close(v, L * W * 0.015, 1e-9), "{v}");
    let replaced = generated(
        Role::Replacement,
        KernelTag::InputFace {
            face: entity(0, top),
        },
    );
    let f = face_of(&out, 0, &replaced);
    assert_eq!(
        out.history
            .faces
            .iter()
            .find(|r| r.face == f && r.body == 0)
            .map(|r| r.evolution),
        Some(Evolution::Generated)
    );
    assert!(out.history.deleted_faces.contains(&entity(0, top)));
    // A tilted replacement plane through (·, 0, H) rising along y at 10°.
    let t = 10f64.to_radians();
    let tilted = Frame {
        origin: [0.0, 0.0, H],
        x: [1.0, 0.0, 0.0],
        y: [0.0, t.cos(), t.sin()],
        z: [0.0, -t.sin(), t.cos()],
    };
    let b = block(&k);
    let s = patch(&k, tilted, -0.01, -0.01, L + 0.01, W / t.cos() + 0.01);
    let out = k.execute(&op, &[&b.handle, &s.handle]).expect("tilted");
    let (v, _) = volume(&k, &out.bodies[0]);
    let want = L * (H * W + t.tan() * W * W / 2.0);
    assert!(close(v, want, 1e-7), "{v} vs {want}");
    let f = face_of(&out, 0, &replaced);
    let n = normal(&out.bodies[0], f);
    assert!(
        (n[1] + t.sin()).abs() < 1e-9 && (n[2] - t.cos()).abs() < 1e-9,
        "{n:?}"
    );
}

// ------------------------------------------------------------------- surfaces

#[test]
fn k1_offset_surface_of_a_sheet_and_of_solid_faces() {
    let k = KernelUnderTest::new();
    // A sheet: every face, boundaries continue the sheet's boundary edges.
    let op = KernelOp::Surface(SurfaceOp::Offset {
        faces: Vec::new(),
        distance: 0.002,
    });
    let out = same_history_as_reference(&|k| vec![patch(k, lifted(H), 0.0, 0.0, L, W)], &op);
    assert_eq!(out.bodies.len(), 2);
    let offset = &out.bodies[1];
    let g = &offset.topology.faces[0].geom;
    assert!(
        (g.centroid[2] - (H + 0.002)).abs() < 1e-9,
        "{:?}",
        g.centroid
    );
    assert!(close(g.area, L * W, 1e-9));
    // Faces of a solid: the top alone, its boundaries named by the side faces across them.
    let top = face_of_block(&k, &end_cap());
    let op = KernelOp::Surface(SurfaceOp::Offset {
        faces: vec![entity(0, top)],
        distance: -0.003,
    });
    let out = same_history_as_reference(&|k| vec![block(k)], &op);
    let sheet = out
        .bodies
        .iter()
        .position(|b| b.topology.faces.len() == 1)
        .expect("the offset sheet");
    let g = &out.bodies[sheet].topology.faces[0].geom;
    assert!((g.centroid[2] - (H - 0.003)).abs() < 1e-9);
    let mut tags: Vec<HistorySource> = out
        .history
        .boundaries
        .iter()
        .filter(|r| r.body == sheet as u32)
        .flat_map(|r| r.sources.clone())
        .collect();
    tags.sort();
    tags.dedup();
    assert_eq!(tags.len(), 4, "{tags:?}");
}

#[test]
fn k1_mid_surface_of_a_slab() {
    let k = KernelUnderTest::new();
    let top = face_of_block(&k, &end_cap());
    let bottom = face_of_block(&k, &start_cap());
    let op = KernelOp::Surface(SurfaceOp::MidSurface {
        pairs: vec![(entity(0, top), entity(0, bottom))],
    });
    let out = same_history_as_reference(&|k| vec![block(k)], &op);
    let mid = out
        .bodies
        .iter()
        .find(|b| b.topology.faces.len() == 1)
        .expect("the mid-surface");
    let g = &mid.topology.faces[0].geom;
    assert!((g.centroid[2] - H / 2.0).abs() < 1e-9);
    assert!(close(g.area, L * W, 1e-9));
    assert_eq!(normal(mid, 0), [0.0, 0.0, 1.0]);
}

// ------------------------------------------------------------------- bounded prisms

/// The block with a step: x in [L/2, L] rises to 2H.
pub fn stepped(k: &dyn CadKernel) -> KernelBody {
    let b = block(k);
    let out = k
        .execute(
            &KernelOp::Tool(ToolOp::new(
                ToolShape::prism(rect(Frame::WORLD, L / 2.0, 0.0, L, W), H, 2.0 * H),
                Combine::Add,
            )),
            &[&b.handle],
        )
        .expect("step");
    out.bodies.into_iter().next().expect("stepped")
}

fn bounded(
    profile: CanonicalProfile,
    start: f64,
    end: f64,
    start_bound: PrismBound,
    end_bound: PrismBound,
    draft: Option<PrismDraft>,
    thin: Option<ThinWallSpec>,
) -> KernelOp {
    KernelOp::Tool(ToolOp {
        shape: ToolShape::Prism {
            profile,
            start,
            end,
            start_bound,
            end_bound,
            draft,
            thin,
        },
        combine: Combine::New,
        references: 1,
    })
}

fn new_body(out: &KernelOutput) -> &KernelBody {
    let i = out
        .history
        .bodies
        .iter()
        .find(|r| {
            !r.sources
                .iter()
                .any(|s| matches!(s, HistorySource::Input { .. }))
        })
        .expect("a new body")
        .body;
    &out.bodies[i as usize]
}

#[test]
fn k1_up_to_next_over_several_faces_end_and_start() {
    let k = KernelUnderTest::new();
    let s = stepped(&k);
    let above = rect(lifted(3.0 * H), L / 4.0, 0.0, 3.0 * L / 4.0, W);
    // Down from z = 3H to the step's two tops (H on the left, 2H on the right).
    let op = bounded(
        above.clone(),
        0.0,
        -1.0,
        PrismBound::Blind,
        PrismBound::UpToNext,
        None,
        None,
    );
    let out = k.execute(&op, &[&s.handle]).expect("up to next");
    let (v, _) = volume(&k, new_body(&out));
    assert!(close(v, (L / 4.0) * W * (2.0 * H + H), 1e-7), "{v}");
    let caps = out
        .history
        .faces
        .iter()
        .filter(|r| r.sources == vec![end_cap()])
        .count();
    assert!(caps >= 2, "the far cap takes both faces' shape: {caps}");
    // As a start bound, with a blind far end up from the sketch plane.
    let op = bounded(
        above,
        -1.0,
        0.5 * H,
        PrismBound::UpToNext,
        PrismBound::Blind,
        None,
        None,
    );
    let out = k.execute(&op, &[&s.handle]).expect("up-to-next start");
    let (v, _) = volume(&k, new_body(&out));
    let want = (L / 4.0) * W * (3.0 * H) + (L / 2.0) * W * (0.5 * H);
    assert!(close(v, want, 1e-7), "{v} vs {want}");
    assert!(out
        .history
        .faces
        .iter()
        .any(|r| r.sources == vec![start_cap()]));
    // Up to one face as a start bound (every ray meets the block's top): the reference kernel
    // builds it as well.
    let small = rect(lifted(3.0 * H), 0.01, 0.01, 0.03, 0.03);
    same_history_as_reference(
        &|k| vec![block(k)],
        &bounded(
            small,
            -1.0,
            0.5 * H,
            PrismBound::UpToNext,
            PrismBound::Blind,
            None,
            None,
        ),
    );
}

#[test]
fn k1_drafted_bounded_and_thin_prisms() {
    let k = KernelUnderTest::new();
    let b = block(&k);
    let top = face_of_block(&k, &end_cap());
    let alpha = 3f64.to_radians();
    let (a, c) = (0.04, 0.02);
    let op = bounded(
        rect(lifted(3.0 * H), 0.03, 0.015, 0.03 + a, 0.015 + c),
        0.0,
        -1.0,
        PrismBound::Blind,
        PrismBound::UpToFace {
            face: entity(0, top),
            offset: 0.0,
        },
        Some(PrismDraft {
            end: alpha,
            start: 0.0,
        }),
        None,
    );
    let out = k.execute(&op, &[&b.handle]).expect("drafted up to face");
    let (v, _) = volume(&k, new_body(&out));
    // A frustum from the sketch plane down 2H, each side in by t·tan α.
    let d = 2.0 * H;
    let t = alpha.tan();
    let want = a * c * d - (a + c) * t * d * d + 4.0 * t * t * d * d * d / 3.0;
    assert!(close(v, want, 1e-7), "{v} vs {want}");
    // A drafted thin wall: the same wall as the reference kernel builds.
    let thin = ThinWallSpec {
        inside: 0.002,
        outside: 0.0,
        cap: None,
    };
    let op = KernelOp::Tool(ToolOp::new(
        ToolShape::Prism {
            profile: rect(Frame::WORLD, 0.0, 0.0, L, W),
            start: 0.0,
            end: H,
            start_bound: PrismBound::Blind,
            end_bound: PrismBound::Blind,
            draft: Some(PrismDraft {
                end: alpha,
                start: 0.0,
            }),
            thin: Some(thin),
        },
        Combine::New,
    ));
    let out = k.execute(&op, &[]).expect("drafted thin");
    let (v, _) = volume(&k, &out.bodies[0]);
    // Both walls lean in by the same angle: the wall keeps its width across the draft, so the
    // volume is the band's area at mid-height times the height.
    let tm = alpha.tan() * H / 2.0;
    let (lo, wo) = (L - 2.0 * tm, W - 2.0 * tm);
    let (li, wi) = (lo - 0.004, wo - 0.004);
    let want = (lo * wo - li * wi) * H;
    assert!(close(v, want, 1e-4), "{v} vs {want}");
    assert!(out.history.faces.iter().any(|r| r.sources
        == vec![generated(
            Role::ThinWall,
            KernelTag::Curve {
                profile: 0,
                curve: SketchEntityId::new("c1"),
            },
        )]));
}

// ------------------------------------------------------------------- hole tables

struct OneModel<'a> {
    doc: &'a runmat_cad::model::PartDocument,
    result: &'a runmat_cad::regen::RegenResult,
}

impl runmat_cad::kernel_seams::RegeneratedModels for OneModel<'_> {
    fn model(
        &self,
        _model: &runmat_cad::drawing::ModelRef,
    ) -> Result<runmat_cad::kernel_seams::RegeneratedModel<'_>, String> {
        Ok(runmat_cad::kernel_seams::RegeneratedModel {
            doc: self.doc,
            result: self.result,
        })
    }
}

struct TopView;

impl runmat_cad::kernel_seams::ViewModels for TopView {
    fn view_frame(&self, _view: &runmat_cad::drawing::View) -> Result<Frame, String> {
        Ok(Frame::WORLD)
    }
    fn bodies(
        &self,
        _view: &runmat_cad::drawing::View,
        _model: &runmat_cad::drawing::ModelRef,
    ) -> Result<Vec<runmat_cad::kernel_seams::ViewBody>, String> {
        Ok(Vec::new())
    }
}

#[test]
fn k1_hole_table_reads_the_regenerated_part() {
    use crate::common::{block, mm, Env};
    use runmat_cad::drawing::hole_table::{hole_table, HoleGeometry, HoleTableSpec};
    use runmat_cad::drawing::*;
    use runmat_cad::ids::FilePath;
    use runmat_cad::model::*;
    use runmat_cad::refs::*;
    let env = Env::new();
    let mut d = block();
    let at = |x: f64, y: f64| PointRef::Coords {
        x: mm(x),
        y: mm(y),
        z: mm(10.0),
    };
    d.features.push(Feature::new(
        "mount",
        FeatureKind::Hole(Hole {
            face: Reference::end_cap("body"),
            positions: vec![
                HolePosition {
                    key: "a".into(),
                    at: at(10.0, 10.0),
                },
                HolePosition {
                    key: "b".into(),
                    at: at(70.0, 40.0),
                },
            ],
            diameter: Some(mm(6.6)),
            end: HoleEnd::ThroughAll,
            form: HoleForm::Simple,
            spec: None,
            drill_point: None,
            series: None,
        }),
    ));
    let r = env.regen(&d);
    assert!(r.status("mount").unwrap().ran());
    let models = OneModel {
        doc: &d,
        result: &r,
    };
    let g = runmat_cad::kernel_seams::RegenHoleGeometry {
        models: &models,
        views: &TopView,
    };
    let model = ModelRef {
        file: FilePath::new("bracket.part"),
        configuration: None,
        variant: Default::default(),
    };
    let holes = g.holes(&model).expect("holes");
    assert_eq!(holes.len(), 2);
    // The datum origin: the block's corner at the origin, on its top.
    let corner = Reference::vertex_of(vec![
        Reference::end_cap("body"),
        Reference::side("body", "s1", "l1"),
        Reference::side("body", "s1", "l4"),
    ]);
    let p = g.point(&model, &corner).expect("corner");
    assert!(
        p.iter()
            .zip([0.0, 0.0, 0.01])
            .all(|(a, b)| (a - b).abs() < 1e-9),
        "{p:?}"
    );
    // A hole's wall names its centre.
    let wall = Reference::gen_face("mount", Role::HoleWall, SourceTag::Key { key: "b".into() });
    let c = g.point(&model, &wall).expect("hole centre");
    assert!(
        (c[0] - 0.07).abs() < 1e-9 && (c[1] - 0.04).abs() < 1e-9,
        "{c:?}"
    );
    let view = View {
        name: "top".into(),
        sheet: "1".into(),
        model: model.clone(),
        kind: ViewKind::Standard {
            orientation: Orientation::Top,
            frame: None,
        },
        at: [0.0, 0.0],
        scale: None,
        display: Display::VisibleLines,
        exclude: vec![],
        exploded: None,
    };
    let t = hole_table("holes", &HoleTableSpec::default(), &view, &corner, &g);
    let rows: Vec<(String, String)> = t
        .rows
        .iter()
        .map(|r| (r.x_text.clone(), r.y_text.clone()))
        .collect();
    assert_eq!(
        rows,
        vec![
            ("10.00".to_string(), "10.00".to_string()),
            ("70.00".to_string(), "40.00".to_string())
        ]
    );
    assert!(t.rows.iter().all(|r| r.size == "Ø6.60 THRU"));
}

// ------------------------------------------------------------------- patterns and lofts

fn instances(n: u32, dx: f64) -> Vec<Instance> {
    (1..=n)
        .map(|step| Instance {
            key: runmat_geometry_kernel::names::InstanceKey::Linear { step },
            transform: Transform::translation([dx * step as f64, 0.0, 0.0]),
        })
        .collect()
}

#[test]
fn k1_patterns_of_mixed_and_new_body_seeds() {
    let k = KernelUnderTest::new();
    let b = block(&k);
    let pocket = ToolOp::new(
        ToolShape::prism(
            rect(Frame::WORLD, 0.002, 0.002, 0.006, 0.006),
            H - 0.002,
            H + 0.001,
        ),
        Combine::Cut,
    );
    let boss = ToolOp::new(
        ToolShape::prism(rect(Frame::WORLD, 0.010, 0.002, 0.014, 0.006), H, H + 0.003),
        Combine::Add,
    );
    let out = k
        .execute(
            &KernelOp::PatternTools {
                seeds: vec![pocket, boss.clone()],
                instances: instances(2, 0.03),
                varied: Vec::new(),
            },
            &[&b.handle],
        )
        .expect("mixed pattern");
    assert_eq!(out.bodies.len(), 1);
    let (v, _) = volume(&k, &out.bodies[0]);
    let cell = 0.004 * 0.004;
    let want = L * W * H - 2.0 * cell * 0.002 + 2.0 * cell * 0.003;
    assert!(close(v, want, 1e-9), "{v} vs {want}");
    // Every pocket floor and boss top carries its seed and instance.
    for seed in [0u32, 1] {
        for step in [1u32, 2] {
            let want = HistorySource::Generated {
                role: if seed == 0 {
                    Role::StartCap
                } else {
                    Role::EndCap
                },
                tag: KernelTag::None,
                ordinal: None,
                seed: Some(seed),
                instance: Some(runmat_geometry_kernel::names::InstanceKey::Linear { step }),
            };
            assert!(
                out.history
                    .faces
                    .iter()
                    .any(|r| r.sources == vec![want.clone()]),
                "no face from {want:?}"
            );
        }
    }
    // A new-body seed: one body per instance; the operand comes back as it was.
    let post = ToolOp::new(
        ToolShape::prism(rect(Frame::WORLD, 0.0, -0.02, 0.004, -0.016), 0.0, 0.01),
        Combine::New,
    );
    let out = k
        .execute(
            &KernelOp::PatternTools {
                seeds: vec![post],
                instances: instances(3, 0.02),
                varied: Vec::new(),
            },
            &[&b.handle],
        )
        .expect("new-body pattern");
    assert_eq!(out.bodies.len(), 4);
    let mut news = 0;
    for r in &out.history.bodies {
        let (v, _) = volume(&k, &out.bodies[r.body as usize]);
        if r.sources
            .iter()
            .any(|s| matches!(s, HistorySource::Input { .. }))
        {
            assert!(close(v, L * W * H, 1e-9));
        } else {
            assert!(close(v, 0.004 * 0.004 * 0.01, 1e-9));
            assert!(matches!(
                r.sources[..],
                [HistorySource::Generated {
                    role: Role::Body,
                    seed: Some(0),
                    instance: Some(_),
                    ..
                }]
            ));
            news += 1;
        }
    }
    assert_eq!(news, 3);
}

fn polygon(frame: Frame, names: &[&str], pts: &[[f64; 2]]) -> CanonicalProfile {
    let n = pts.len();
    CanonicalProfile {
        frame,
        loops: vec![(0..n)
            .map(|i| line(names[i], pts[i], pts[(i + 1) % n]))
            .collect()],
        merge_collinear: false,
    }
}

fn shoelace(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    (0..n)
        .map(|i| p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1])
        .sum::<f64>()
        .abs()
        / 2.0
}

#[test]
fn k1_loft_between_sections_with_different_curve_counts() {
    let k = KernelUnderTest::new();
    let tri = [[0.0, 0.0], [0.04, 0.0], [0.02, 0.03]];
    let sq = [[0.0, 0.0], [0.04, 0.0], [0.04, 0.04], [0.0, 0.04]];
    let out = k
        .execute(
            &new_tool(ToolShape::Loft {
                profiles: vec![
                    polygon(Frame::WORLD, &["t1", "t2", "t3"], &tri),
                    polygon(lifted(H), &["q1", "q2", "q3", "q4"], &sq),
                ],
            }),
            &[],
        )
        .expect("loft");
    let body = &out.bodies[0];
    // The triangle's longest side splits at its middle; the ruled loft between corresponding
    // vertices has a quadratic section area (prismatoid rule).
    let refined = [[0.0, 0.0], [0.02, 0.0], [0.04, 0.0], [0.02, 0.03]];
    let midsec: Vec<[f64; 2]> = refined
        .iter()
        .zip(&sq)
        .map(|(a, b)| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0])
        .collect();
    let want = H / 6.0 * (shoelace(&refined) + 4.0 * shoelace(&midsec) + shoelace(&sq));
    let (v, _) = volume(&k, body);
    assert!(close(v, want, 1e-6), "{v} vs {want}");
    // Sides are named by the section with more curves.
    for c in ["q1", "q2", "q3", "q4"] {
        face_of(&out, 0, &side(c));
    }
    assert_eq!(body.topology.faces.len(), 6);
}

// ------------------------------------------------------------------- drafts

#[test]
fn k1_step_draft_with_one_hinge_height_is_the_parting_line_draft() {
    let k = KernelUnderTest::new();
    let b = block(&k);
    let bottom = face_of_block(&k, &start_cap());
    let sides: Vec<u32> = ["c1", "c2", "c3", "c4"]
        .iter()
        .map(|c| face_of_block(&k, &side(c)))
        .collect();
    let parting: Vec<OperandEntity> = b
        .topology
        .edges
        .iter()
        .enumerate()
        .filter(|(_, e)| e.faces.contains(&bottom) && sides.iter().any(|s| e.faces.contains(s)))
        .map(|(i, _)| entity(0, i as u32))
        .collect();
    assert_eq!(parting.len(), 4);
    let op = |step: bool| KernelOp::Draft {
        faces: sides.iter().map(|f| entity(0, *f)).collect(),
        neutral: Frame::WORLD,
        angle: 4f64.to_radians(),
        parting: parting.clone(),
        step,
    };
    let stepped = k.execute(&op(true), &[&b.handle]).expect("step draft");
    let plain = k.execute(&op(false), &[&b.handle]).expect("parting draft");
    // The reference kernel drafts one side of the block (its grid cannot rebuild four).
    let one = KernelOp::Draft {
        faces: vec![entity(0, sides[0])],
        neutral: Frame::WORLD,
        angle: 4f64.to_radians(),
        parting: parting
            .iter()
            .filter(|e| {
                b.topology.edges[e.entity as usize]
                    .faces
                    .contains(&sides[0])
            })
            .copied()
            .collect(),
        step: true,
    };
    same_history_as_reference(&|k| vec![block(k)], &one);
    let (vs, _) = volume(&k, &stepped.bodies[0]);
    let (vp, _) = volume(&k, &plain.bodies[0]);
    assert!(close(vs, vp, 1e-12), "{vs} vs {vp}");
    let t = 4f64.to_radians().tan();
    let narrowing = L * W * H - (L + W) * t * H * H + 4.0 * t * t * H * H * H / 3.0;
    assert!(close(vs, narrowing, 1e-7), "{vs} vs {narrowing}");
}

// ------------------------------------------------------------------- free-form profile curves

fn prism_of(segs: Vec<ProfileSegment>, h: f64) -> KernelOp {
    new_tool(ToolShape::prism(
        CanonicalProfile {
            frame: Frame::WORLD,
            loops: vec![segs],
            merge_collinear: false,
        },
        0.0,
        h,
    ))
}

#[test]
fn k1_ellipse_and_spline_profiles() {
    let k = KernelUnderTest::new();
    let (a, b) = (0.03, 0.012);
    // A whole ellipse, tilted 30°.
    let t = 30f64.to_radians();
    let start = [a * t.cos(), a * t.sin()];
    let whole = ProfileSegment {
        curve: SketchEntityId::new("e"),
        kind: SegmentKind::Ellipse {
            center: [0.0, 0.0],
            major: start,
            ratio: b / a,
            clockwise: false,
        },
        start,
        end: start,
    };
    let out = k.execute(&prism_of(vec![whole], H), &[]).expect("ellipse");
    let (v, c) = volume(&k, &out.bodies[0]);
    assert!(close(v, std::f64::consts::PI * a * b * H, 1e-9), "{v}");
    assert!(c[0].abs() < 1e-12 && c[1].abs() < 1e-12);
    face_of(&out, 0, &side("e"));
    // Half an ellipse closed by its major axis, clockwise from (-a, 0) over the top... drawn
    // as the counter-clockwise arc from (a, 0) to (-a, 0) and the line back.
    let half = vec![
        ProfileSegment {
            curve: SketchEntityId::new("arc"),
            kind: SegmentKind::Ellipse {
                center: [0.0, 0.0],
                major: [a, 0.0],
                ratio: b / a,
                clockwise: false,
            },
            start: [a, 0.0],
            end: [-a, 0.0],
        },
        line("base", [-a, 0.0], [a, 0.0]),
    ];
    let out = k.execute(&prism_of(half, H), &[]).expect("half ellipse");
    let (v, c) = volume(&k, &out.bodies[0]);
    assert!(
        close(v, std::f64::consts::PI * a * b * H / 2.0, 1e-9),
        "{v}"
    );
    // The half-ellipse centroid: 4b / 3π above the axis.
    assert!(
        (c[1] - 4.0 * b / (3.0 * std::f64::consts::PI)).abs() < 1e-9,
        "{c:?}"
    );
    // A cubic spline edge closing a region with two lines; and a rational one.
    for weights in [Vec::new(), vec![1.0, 0.6, 0.6, 1.0]] {
        let spline = ProfileSegment {
            curve: SketchEntityId::new("s"),
            kind: SegmentKind::Spline {
                degree: 3,
                poles: vec![[L, 0.0], [L, W], [L / 2.0, W], [0.0, W]],
                weights: weights.clone(),
                knots: vec![0.0, 1.0],
                multiplicities: vec![4, 4],
            },
            start: [L, 0.0],
            end: [0.0, W],
        };
        let pts = spline.polyline(20000);
        let segs = vec![
            line("l1", [0.0, 0.0], [L, 0.0]),
            spline,
            line("l2", [0.0, W], [0.0, 0.0]),
        ];
        let mut poly = vec![[0.0, 0.0]];
        poly.extend(pts);
        let area = shoelace(&poly);
        let out = k.execute(&prism_of(segs, H), &[]).expect("spline");
        let (v, _) = volume(&k, &out.bodies[0]);
        // OCCT integrates the free-form side face to about 1e-7 relative.
        assert!(close(v, area * H, 1e-6), "{v} vs {}", area * H);
        assert_eq!(out.bodies[0].topology.faces.len(), 5);
        face_of(&out, 0, &side("s"));
    }
}

#[test]
fn k1_thin_walls_along_open_profiles() {
    let k = KernelUnderTest::new();
    // An L: along x then up y, 2 mm inside (left of travel), 1 mm outside.
    let chain = vec![
        line("a", [0.0, 0.0], [0.04, 0.0]),
        line("b", [0.04, 0.0], [0.04, 0.03]),
    ];
    let thin = ThinWallSpec {
        inside: 0.002,
        outside: 0.001,
        cap: None,
    };
    let op = |chain: Vec<ProfileSegment>| {
        new_tool(ToolShape::Prism {
            profile: CanonicalProfile {
                frame: Frame::WORLD,
                loops: vec![chain],
                merge_collinear: false,
            },
            start: 0.0,
            end: H,
            start_bound: PrismBound::Blind,
            end_bound: PrismBound::Blind,
            draft: None,
            thin: Some(thin),
        })
    };
    let out = k.execute(&op(chain), &[]).expect("open thin");
    let body = &out.bodies[0];
    // Outer corner at (0.041, -0.001), inner at (0.038, 0.002): the L's area.
    let outer = [
        [0.0, -0.001],
        [0.041, -0.001],
        [0.041, 0.03],
        [0.038, 0.03],
        [0.038, 0.002],
        [0.0, 0.002],
    ];
    let (v, _) = volume(&k, body);
    assert!(close(v, shoelace(&outer) * H, 1e-9), "{v}");
    assert_eq!(body.topology.faces.len(), 8);
    for c in ["a", "b"] {
        face_of(&out, 0, &side(c));
        face_of(
            &out,
            0,
            &generated(
                Role::ThinWall,
                KernelTag::Curve {
                    profile: 0,
                    curve: SketchEntityId::new(c),
                },
            ),
        );
    }
    let end = |c: &str, o: u32| HistorySource::Generated {
        role: Role::ThinWall,
        tag: KernelTag::Curve {
            profile: 0,
            curve: SketchEntityId::new(c),
        },
        ordinal: Some(o),
        seed: None,
        instance: None,
    };
    face_of(&out, 0, &end("a", 2));
    face_of(&out, 0, &end("b", 3));
    // A line into a quarter arc (tangent): the wall follows the arc at radius 10 ± mm.
    let r = 0.01;
    let chain = vec![
        line("l", [0.0, 0.0], [0.02, 0.0]),
        ProfileSegment {
            curve: SketchEntityId::new("q"),
            kind: SegmentKind::Arc {
                center: [0.02, r],
                clockwise: false,
            },
            start: [0.02, 0.0],
            end: [0.02 + r, r],
        },
    ];
    let out = k.execute(&op(chain), &[]).expect("open thin with an arc");
    let (v, _) = volume(&k, &out.bodies[0]);
    let band = 0.003;
    let ring = std::f64::consts::PI / 4.0 * ((r + 0.001).powi(2) - (r - 0.002).powi(2));
    assert!(close(v, (0.02 * band + ring) * H, 1e-9), "{v}");
}

// ------------------------------------------------------------------- conic and G2 blends

/// The region between a blend section and its corner, for a 90° edge with setback `s`, the
/// section given as a polyline from one tangent point to the other in edge-local coordinates
/// (the corner at the origin, the faces along the axes).
fn corner_area(section: &[[f64; 2]]) -> f64 {
    let mut p = section.to_vec();
    p.push([0.0, 0.0]);
    shoelace(&p)
}

fn bezier(poles: &[[f64; 2]], t: f64) -> [f64; 2] {
    let mut q = poles.to_vec();
    while q.len() > 1 {
        q = q
            .windows(2)
            .map(|w| {
                [
                    w[0][0] + t * (w[1][0] - w[0][0]),
                    w[0][1] + t * (w[1][1] - w[0][1]),
                ]
            })
            .collect();
    }
    q[0]
}

#[test]
fn k1_conic_and_curvature_blends_of_straight_edges() {
    let k = KernelUnderTest::new();
    let b = block(&k);
    let top = face_of_block(&k, &end_cap());
    let front = face_of_block(&k, &side("c1"));
    let edge = b
        .topology
        .edges
        .iter()
        .position(|e| e.faces.contains(&top) && e.faces.contains(&front))
        .expect("top front edge") as u32;
    let r = 0.003;
    let blend = generated(
        Role::Blend,
        KernelTag::InputEdge {
            edge: entity(0, edge),
        },
    );
    // A conic of rho 0.5 is a parabola: the corner region is a sixth of the setback squared.
    let out = k
        .execute(
            &KernelOp::Fillet {
                edges: vec![entity(0, edge)],
                radius: r,
                variable: Vec::new(),
                profile: BlendProfile::Conic { rho: 0.5 },
                setbacks: Vec::new(),
            },
            &[&b.handle],
        )
        .expect("conic");
    let (v, _) = volume(&k, &out.bodies[0]);
    assert!(close(v, L * W * H - L * r * r / 6.0, 1e-9), "{v}");
    face_of(&out, 0, &blend);
    assert_eq!(out.bodies[0].topology.faces.len(), 7);
    // Curvature-continuous: the quartic through the corner.
    let poles = [
        [r, 0.0],
        [r / 2.0, 0.0],
        [0.0, 0.0],
        [0.0, r / 2.0],
        [0.0, r],
    ];
    let section: Vec<[f64; 2]> = (0..=4000)
        .map(|i| bezier(&poles, i as f64 / 4000.0))
        .collect();
    let out = k
        .execute(
            &KernelOp::FaceBlend {
                first: vec![entity(0, top)],
                second: vec![entity(0, front)],
                radius: r,
                profile: BlendProfile::Curvature,
            },
            &[&b.handle],
        )
        .expect("curvature");
    let (v, _) = volume(&k, &out.bodies[0]);
    let want = L * W * H - L * corner_area(&section);
    assert!(close(v, want, 1e-7), "{v} vs {want}");
    face_of(&out, 0, &blend);
    // A concave edge (the step's foot) is filled.
    let s = stepped(&k);
    let foot = s
        .topology
        .edges
        .iter()
        .position(|e| {
            [e.geom.start, e.geom.end]
                .iter()
                .all(|p| (p[0] - L / 2.0).abs() < 1e-9 && (p[2] - H).abs() < 1e-9)
        })
        .expect("the step's foot") as u32;
    let out = k
        .execute(
            &KernelOp::Fillet {
                edges: vec![entity(0, foot)],
                radius: r,
                variable: Vec::new(),
                profile: BlendProfile::Conic { rho: 0.5 },
                setbacks: Vec::new(),
            },
            &[&s.handle],
        )
        .expect("concave conic");
    let (v0, _) = volume(&k, &s);
    let (v, _) = volume(&k, &out.bodies[0]);
    assert!(close(v, v0 + W * r * r / 6.0, 1e-9), "{v} vs {v0}");
    face_of(
        &out,
        0,
        &generated(
            Role::Blend,
            KernelTag::InputEdge {
                edge: entity(0, foot),
            },
        ),
    );
}

// ------------------------------------------------------------------- sheet metal hems

#[test]
fn k1_closed_and_open_hems_name_their_faces_as_the_reference_kernel_does() {
    use crate::common::{mm, sketch, Env};
    use runmat_cad::ids::FeatureId;
    use runmat_cad::model::sheet_metal::*;
    use runmat_cad::model::*;
    use runmat_cad::refs::*;
    use runmat_cad::regen::{regenerate, RegenCache, RegenContext, RegenRequest};
    let mut s = Sketch {
        plane: PlaneRef::Datum {
            datum: DatumRef::OriginPlane {
                plane: OriginPlane::Zx,
            },
        },
        entities: Vec::new(),
        constraints: Vec::new(),
        solved: Default::default(),
    };
    let pts = [(0.0, 30.0), (0.0, 0.0), (50.0, 0.0)];
    for (k, name) in ["up", "base"].iter().enumerate() {
        s.entities.push(SketchEntity {
            id: SketchEntityId::new(*name),
            geometry: SketchGeometry::Line {
                start: PointExpr::mm(pts[k].0, pts[k].1),
                end: PointExpr::mm(pts[k + 1].0, pts[k + 1].1),
            },
            construction: false,
        });
    }
    let named = |role: Role, key: &str, o: u32| {
        Reference::face(EntityName::Generated(GenerationName {
            feature: FeatureId::new("base"),
            role,
            tag: SourceTag::Key { key: key.into() },
            ordinal: Some(o),
        }))
    };
    for kind in [HemKind::Closed, HemKind::Open { gap: mm(1.0) }] {
        let mut d = PartDocument::new("hem.part");
        d.features.push(sketch("s_chain", s.clone()));
        d.features.push(Feature::new(
            "base",
            FeatureKind::SheetMetal(SheetMetalFeature::BaseFlange {
                profile: ProfileRef::sketch("s_chain"),
                depth: Some(mm(40.0)),
                side: MaterialSide::Left,
                rules: SheetMetalRules {
                    thickness: mm(2.0),
                    bend_radius: mm(3.0),
                    rule: BendRule::KFactor {
                        k: runmat_cad::value::ValueExpr::number(0.44),
                    },
                    gauge: None,
                    relief: ReliefKind::Tear,
                },
            }),
        ));
        d.features.push(Feature::new(
            "hem",
            FeatureKind::SheetMetal(SheetMetalFeature::Hem {
                edges: vec![Reference::edges_between(
                    named(Role::SheetFace, "base", 0),
                    named(Role::SheetEdge, "base", 1),
                )],
                kind: kind.clone(),
                length: mm(8.0),
            }),
        ));
        let env = Env::new();
        let r = env.regen(&d);
        assert!(r.status("hem").unwrap().ran(), "{:?}", r.diagnostics);
        let fake = FakeKernel::new();
        let ctx = RegenContext {
            kernel: &fake,
            ..env.ctx()
        };
        let reference = regenerate(
            &d,
            &ctx,
            &RegenRequest::default(),
            &mut RegenCache::default(),
        );
        let hem_names = |r: &runmat_cad::regen::RegenResult| {
            let mut v: Vec<String> = r
                .state
                .bodies
                .iter()
                .flat_map(|b| b.names.faces.iter())
                .flatten()
                .filter(|n| matches!(n, EntityName::Generated(g) if g.feature.0 == "hem"))
                .map(|n| format!("{n:?}"))
                .collect();
            v.sort();
            v
        };
        let names = hem_names(&r);
        assert!(!names.is_empty());
        if reference.status("hem").map(|s| s.ran()).unwrap_or(false) {
            assert_eq!(names, hem_names(&reference), "{kind:?}");
        }
        // The hem adds its half-turn fold and its flat: depth × t × (length + π (r + t/2)),
        // r the fold's inner radius (half the gap; a hundredth of t when closed).
        let k = KernelUnderTest::new();
        let vol = |r: &runmat_cad::regen::RegenResult| {
            volume(
                &k,
                &KernelBody {
                    handle: r.state.bodies[0].handle.clone(),
                    topology: r.state.bodies[0].topology.clone(),
                },
            )
            .0
        };
        let mut plain = d.clone();
        plain.features.pop();
        let before = vol(&env.regen(&plain));
        let t = 0.002;
        let inner = match kind {
            HemKind::Closed => t * 0.01,
            _ => 0.0005,
        };
        let want = 0.04 * t * (0.008 + std::f64::consts::PI * (inner + t / 2.0));
        let added = vol(&r) - before;
        assert!(close(added, want, 1e-6), "{kind:?}: {added} vs {want}");
    }
}
