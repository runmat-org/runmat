//! The naming corpus across feature kinds.
//!
//! Every feature family (extrudes of arc, circle and slanted profiles; revolves; revolve cuts;
//! sweeps; lofts; shells; drafts; simple, counterbore and countersink holes; linear and
//! circular patterns; mirrors; rotated body patterns; fillets; chamfers; splits; booleans) is
//! built from inputs and probed with references whose expected targets are stated as oracles
//! on the geometry. Each family is then put through the same edits:
//!
//! - parameter edits, several of which change topology counts;
//! - suppress, unsuppress, and suppression in a configuration;
//! - a configuration that switches parameters;
//! - a reorder of an independent feature around the family's feature;
//! - a rename through the lock (references keep the old id);
//! - a merge of two concurrent edits, rechecked for redirected or broken references;
//!
//! plus family-specific sketch edits (curves added and removed), pattern count changes,
//! splits, merges and cross-file renames. Every probe in every state must resolve to an entity
//! that satisfies its oracle, or fail with exactly the expected problem. Nothing resolves by
//! geometry, nothing changes silently, and every canonical reference the state offers resolves
//! back to its own entity.

mod common;
use common::*;
use runmat_cad::diagnostics::{DiagCode, Repair};
use runmat_cad::ids::{FeatureId, FilePath, InputName, PublishName};
use runmat_cad::model::*;
use runmat_cad::refs::merge::recheck_part_merge;
use runmat_cad::refs::resolve::BodyView;
use runmat_cad::refs::*;
use runmat_cad::regen::*;
use runmat_cad::value::{BinaryOp, InputDecl, ValueExpr};
use std::collections::BTreeMap;
use std::rc::Rc;

// ------------------------------------------------------------------------------- helpers

fn fid(s: &str) -> FeatureId {
    FeatureId::new(s)
}

fn add(a: ValueExpr, b: ValueExpr) -> ValueExpr {
    ValueExpr::binary(BinaryOp::Add, a, b)
}

fn sub(a: ValueExpr, b: ValueExpr) -> ValueExpr {
    ValueExpr::binary(BinaryOp::Sub, a, b)
}

fn mul(k: f64, b: ValueExpr) -> ValueExpr {
    ValueExpr::binary(BinaryOp::Mul, ValueExpr::number(k), b)
}

fn half(b: ValueExpr) -> ValueExpr {
    mul(0.5, b)
}

fn pt(x: ValueExpr, y: ValueExpr) -> PointExpr {
    PointExpr::Coords { x, y }
}

fn line(id: &str, a: (ValueExpr, ValueExpr), b: (ValueExpr, ValueExpr)) -> SketchEntity {
    SketchEntity {
        id: runmat_cad::ids::SketchEntityId::new(id),
        geometry: SketchGeometry::Line {
            start: pt(a.0, a.1),
            end: pt(b.0, b.1),
        },
        construction: false,
    }
}

fn arc(
    id: &str,
    c: (ValueExpr, ValueExpr),
    a: (ValueExpr, ValueExpr),
    b: (ValueExpr, ValueExpr),
) -> SketchEntity {
    SketchEntity {
        id: runmat_cad::ids::SketchEntityId::new(id),
        geometry: SketchGeometry::Arc {
            center: pt(c.0, c.1),
            start: pt(a.0, a.1),
            end: pt(b.0, b.1),
        },
        construction: false,
    }
}

fn circle(id: &str, c: (ValueExpr, ValueExpr), r: ValueExpr) -> SketchEntity {
    SketchEntity {
        id: runmat_cad::ids::SketchEntityId::new(id),
        geometry: SketchGeometry::Circle {
            center: pt(c.0, c.1),
            radius: r,
        },
        construction: false,
    }
}

fn sketch_of(plane: PlaneRef, entities: Vec<SketchEntity>) -> Sketch {
    Sketch {
        plane,
        entities,
        constraints: Vec::new(),
        solved: SolvedCache::default(),
    }
}

fn offset_plane(id: &str, base: PlaneRef, distance: ValueExpr) -> Feature {
    Feature::new(id, FeatureKind::Plane(PlaneDef::Offset { base, distance }))
}

fn origin(plane: OriginPlane) -> PlaneRef {
    PlaneRef::Datum {
        datum: DatumRef::OriginPlane { plane },
    }
}

fn top_plane() -> Feature {
    offset_plane("top_plane", PlaneRef::origin_xy(), input("thickness"))
}

fn vertical_axis(id: &str, x: f64, y: f64) -> Feature {
    let p = |z: f64| PointRef::Coords {
        x: mm(x),
        y: mm(y),
        z: mm(z),
    };
    Feature::new(
        id,
        FeatureKind::Axis(AxisDef::Through {
            a: p(0.0),
            b: p(10.0),
        }),
    )
}

fn axis_datum(id: &str) -> AxisRef {
    AxisRef::Datum {
        datum: DatumRef::Feature { feature: fid(id) },
    }
}

fn name_of(r: &Reference) -> EntityName {
    match r {
        Reference::Name { name, .. } => name.clone(),
        _ => panic!("not a name"),
    }
}

fn hole_feature(
    id: &str,
    x: f64,
    y: f64,
    dia: ValueExpr,
    depth: Option<ValueExpr>,
    form: HoleForm,
) -> Feature {
    Feature::new(
        id,
        FeatureKind::Hole(Hole {
            face: Reference::end_cap("body"),
            positions: vec![HolePosition {
                key: "a".into(),
                at: PointRef::Coords {
                    x: mm(x),
                    y: mm(y),
                    z: input("thickness"),
                },
            }],
            diameter: Some(dia),
            end: depth
                .map(|depth| HoleEnd::Blind { depth })
                .unwrap_or(HoleEnd::ThroughAll),
            form,
            spec: None,
            drill_point: None,
            series: None,
        }),
    )
}

fn hole_face(feature: &str, role: Role) -> Reference {
    Reference::gen_face(feature, role, SourceTag::Key { key: "a".into() })
}

// ------------------------------------------------------------------------------- oracles

/// Parameter values by input name (mm, degrees, counts).
type Pv = BTreeMap<&'static str, f64>;

fn pv(items: &[(&'static str, f64)]) -> Pv {
    items.iter().copied().collect()
}

#[derive(Clone)]
enum Geom {
    Face(FaceGeom),
    Edge(EdgeGeom),
    Body,
}

type Oracle = Rc<dyn Fn(&Geom) -> Result<(), String>>;

fn close(a: f64, b: f64, tol: f64, what: &str) -> Result<(), String> {
    if (a - b).abs() <= tol {
        Ok(())
    } else {
        Err(format!("{}: {} vs expected {} (tol {})", what, a, b, tol))
    }
}

fn face_of(g: &Geom) -> Result<&FaceGeom, String> {
    match g {
        Geom::Face(f) => Ok(f),
        _ => Err("not a face".into()),
    }
}

/// A plane whose normal is along `n` (either sense) and whose centroid lies at `coord` on
/// `axis` (all lengths in mm).
fn plane_at(n: [f64; 3], axis: usize, coord: f64) -> Oracle {
    Rc::new(move |g| {
        let f = face_of(g)?;
        match f.surface {
            SurfaceKind::Plane { normal } => {
                let d = normal[0] * n[0] + normal[1] * n[1] + normal[2] * n[2];
                close(d.abs(), 1.0, 1e-6, "normal alignment")?;
            }
            other => return Err(format!("not a plane: {:?}", other)),
        }
        close(f.centroid[axis] * 1000.0, coord, 1e-6, "plane position")
    })
}

/// A plane along `n` (either sense), anywhere.
fn plane_along(n: [f64; 3]) -> Oracle {
    Rc::new(move |g| {
        let f = face_of(g)?;
        match f.surface {
            SurfaceKind::Plane { normal } => {
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                let d = (normal[0] * n[0] + normal[1] * n[1] + normal[2] * n[2]) / l;
                close(d.abs(), 1.0, 1e-6, "normal alignment")
            }
            other => Err(format!("not a plane: {:?}", other)),
        }
    })
}

/// A cylinder of radius `r` about a vertical or horizontal axis through (`x`, `y`) in mm
/// (coordinates of the axis point on the two axes other than `along`).
fn cylinder(r: f64, along: usize, at: [f64; 2]) -> Oracle {
    Rc::new(move |g| {
        let f = face_of(g)?;
        match f.surface {
            SurfaceKind::Cylinder {
                axis_point,
                axis_direction,
                radius,
            } => {
                close(radius * 1000.0, r, 1e-6, "radius")?;
                close(axis_direction[along].abs(), 1.0, 1e-9, "axis direction")?;
                let others: Vec<usize> = (0..3).filter(|a| *a != along).collect();
                close(axis_point[others[0]] * 1000.0, at[0], 1e-6, "axis position")?;
                close(axis_point[others[1]] * 1000.0, at[1], 1e-6, "axis position")
            }
            other => Err(format!("not a cylinder: {:?}", other)),
        }
    })
}

/// The face's centroid lies within `tol` mm of `c` on the given axes.
fn centroid_near(c: [Option<f64>; 3], tol: f64) -> Oracle {
    Rc::new(move |g| {
        let f = face_of(g)?;
        for (a, want) in c.iter().enumerate() {
            if let Some(v) = want {
                close(f.centroid[a] * 1000.0, *v, tol, "centroid")?;
            }
        }
        Ok(())
    })
}

fn both(a: Oracle, b: Oracle) -> Oracle {
    Rc::new(move |g| {
        a(g)?;
        b(g)
    })
}

fn any() -> Oracle {
    Rc::new(|_| Ok(()))
}

// ------------------------------------------------------------------------------- probes

#[derive(Clone)]
enum Expect {
    /// Resolves to `count` entities, each satisfying the oracle.
    Ok(usize, Oracle),
    /// Fails with exactly this problem (the `kind` tag of `RefProblem`).
    Problem(&'static str),
    Waiting,
}

fn ok1(o: Oracle) -> Expect {
    Expect::Ok(1, o)
}

#[derive(Clone)]
struct Probe {
    name: &'static str,
    kind: TopoKind,
    reference: Reference,
    card: Cardinality,
    /// The feature whose suppression makes this probe wait.
    maker: &'static str,
    expect: Expect,
    /// What the probe resolves to while the family's feature is suppressed, when it does not
    /// simply wait (references to faces the feature only modifies).
    when_suppressed: Option<Expect>,
}

impl Probe {
    fn suppressed(mut self, e: Expect) -> Self {
        self.when_suppressed = Some(e);
        self
    }
}

fn probe(
    name: &'static str,
    kind: TopoKind,
    reference: Reference,
    card: Cardinality,
    maker: &'static str,
    expect: Expect,
) -> Probe {
    Probe {
        name,
        kind,
        reference,
        card,
        maker,
        expect,
        when_suppressed: None,
    }
}

fn face1(name: &'static str, r: Reference, maker: &'static str, o: Oracle) -> Probe {
    probe(
        name,
        TopoKind::Face,
        r,
        Cardinality::ExactlyOne,
        maker,
        ok1(o),
    )
}

// ------------------------------------------------------------------------------- families

struct Family {
    name: &'static str,
    feature: &'static str,
    /// Extra inputs (name, kind, default).
    inputs: Vec<(&'static str, &'static str, f64)>,
    build: fn(&mut PartDocument),
    probes: fn(&Pv) -> Vec<Probe>,
    edits: Vec<Pv>,
}

impl Family {
    fn defaults(&self) -> Pv {
        let mut p = pv(&[("width", 80.0), ("depth", 50.0), ("thickness", 10.0)]);
        for (n, _, v) in &self.inputs {
            p.insert(n, *v);
        }
        p
    }

    fn at(&self, edit: &Pv) -> Pv {
        let mut p = self.defaults();
        p.extend(edit.iter().map(|(k, v)| (*k, *v)));
        p
    }

    fn doc(&self, p: &Pv) -> PartDocument {
        let mut d = block();
        for (n, k, _) in &self.inputs {
            d.inputs.push(InputDecl::new(n, k, mm(0.0)));
        }
        (self.build)(&mut d);
        apply(&mut d, p);
        for pr in (self.probes)(p) {
            d.regions
                .push(region(pr.name, pr.kind, vec![pr.reference], pr.card));
        }
        d
    }
}

fn value_for(d: &PartDocument, name: &str, v: f64) -> ValueExpr {
    let kind = d
        .inputs
        .iter()
        .find(|i| i.name.0 == name)
        .unwrap_or_else(|| panic!("no input {}", name))
        .kind
        .0
        .clone();
    match kind.as_str() {
        "angle" => ValueExpr::deg(v),
        "count" => ValueExpr::number(v),
        _ => mm(v),
    }
}

fn apply(d: &mut PartDocument, p: &Pv) {
    for (k, v) in p {
        let e = value_for(d, k, *v);
        set_input(d, k, e);
    }
}

fn arc_boss() -> Family {
    Family {
        name: "arc_boss",
        feature: "boss",
        inputs: vec![
            ("len", "length", 15.0),
            ("rad", "length", 6.0),
            ("h", "length", 8.0),
        ],
        build: |d| {
            let (x0, x1) = (sub(mm(40.0), input("len")), add(mm(40.0), input("len")));
            let (y0, y1) = (sub(mm(25.0), input("rad")), add(mm(25.0), input("rad")));
            d.features.push(top_plane());
            d.features.push(sketch(
                "s_boss",
                sketch_of(
                    PlaneRef::datum("top_plane"),
                    vec![
                        line("lb", (x0.clone(), y0.clone()), (x1.clone(), y0.clone())),
                        arc(
                            "ar",
                            (x1.clone(), mm(25.0)),
                            (x1.clone(), y0.clone()),
                            (x1.clone(), y1.clone()),
                        ),
                        line("lt", (x1, y1.clone()), (x0.clone(), y1.clone())),
                        arc("al", (x0.clone(), mm(25.0)), (x0.clone(), y1), (x0, y0)),
                    ],
                ),
            ));
            d.features
                .push(extrude("boss", "s_boss", input("h"), BodyMode::add()));
        },
        probes: |p| {
            let side = |c| side("boss", "s_boss", c);
            vec![
                face1(
                    "boss_arc",
                    side("ar"),
                    "boss",
                    cylinder(p["rad"], 2, [40.0 + p["len"], 25.0]),
                ),
                face1(
                    "boss_flat",
                    side("lb"),
                    "boss",
                    plane_at([0.0, 1.0, 0.0], 1, 25.0 - p["rad"]),
                ),
                face1(
                    "boss_top",
                    Reference::end_cap("boss"),
                    "boss",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0 + p["h"]),
                ),
                probe(
                    "boss_rim",
                    TopoKind::Edge,
                    Reference::edges_between(side("ar"), Reference::end_cap("boss")),
                    Cardinality::AtLeastOne,
                    "boss",
                    Expect::Ok(1, any()),
                ),
                face1(
                    "blk_top",
                    Reference::end_cap("body"),
                    "body",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0),
                ),
            ]
        },
        edits: vec![
            pv(&[("h", 12.0)]),
            pv(&[("rad", 8.0)]),
            pv(&[("len", 10.0)]),
        ],
    }
}

fn circle_pocket() -> Family {
    Family {
        name: "circle_pocket",
        feature: "pocket",
        inputs: vec![("pr", "length", 10.0), ("pd", "length", 4.0)],
        build: |d| {
            d.features.push(top_plane());
            d.features.push(sketch(
                "s_pk",
                sketch_of(
                    PlaneRef::datum("top_plane"),
                    vec![circle("c1", (mm(40.0), mm(25.0)), input("pr"))],
                ),
            ));
            d.features.push(extrude_reverse(
                "pocket",
                "s_pk",
                input("pd"),
                BodyMode::cut(),
            ));
        },
        probes: |p| {
            let wall = side("pocket", "s_pk", "c1");
            vec![
                face1(
                    "pk_wall",
                    wall.clone(),
                    "pocket",
                    cylinder(p["pr"], 2, [40.0, 25.0]),
                ),
                face1(
                    "pk_floor",
                    Reference::end_cap("pocket"),
                    "pocket",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0 - p["pd"]),
                ),
                probe(
                    "pk_edge",
                    TopoKind::Edge,
                    Reference::edges_between(wall, Reference::end_cap("pocket")),
                    Cardinality::AtLeastOne,
                    "pocket",
                    Expect::Ok(1, any()),
                ),
                face1(
                    "blk_top",
                    Reference::end_cap("body"),
                    "body",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0),
                ),
            ]
        },
        edits: vec![pv(&[("pr", 14.0)]), pv(&[("pd", 7.0)])],
    }
}

fn hex_points(n: usize) -> Vec<(ValueExpr, ValueExpr)> {
    (0..n)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            (
                add(mm(40.0), mul(a.cos(), input("hr"))),
                add(mm(25.0), mul(a.sin(), input("hr"))),
            )
        })
        .collect()
}

fn hex_cut() -> Family {
    Family {
        name: "hex_cut",
        feature: "hex",
        inputs: vec![("hr", "length", 12.0)],
        build: |d| {
            d.features.push(top_plane());
            let names = ["h1", "h2", "h3", "h4", "h5", "h6"];
            d.features.push(sketch(
                "s_hex",
                polygon(PlaneRef::datum("top_plane"), &hex_points(6), Some(&names)),
            ));
            d.features
                .push(extrude_reverse("hex", "s_hex", mm(20.0), BodyMode::cut()));
        },
        probes: |p| {
            let a = 30f64.to_radians();
            let r = p["hr"] * a.cos();
            vec![
                face1(
                    "hex_side1",
                    side("hex", "s_hex", "h1"),
                    "hex",
                    both(
                        plane_along([a.cos(), a.sin(), 0.0]),
                        centroid_near(
                            [
                                Some(40.0 + r * a.cos()),
                                Some(25.0 + r * a.sin()),
                                Some(5.0),
                            ],
                            0.2 * p["hr"],
                        ),
                    ),
                ),
                face1(
                    "hex_side4",
                    side("hex", "s_hex", "h4"),
                    "hex",
                    plane_along([a.cos(), a.sin(), 0.0]),
                ),
                face1(
                    "blk_top",
                    Reference::end_cap("body"),
                    "body",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0),
                ),
            ]
        },
        edits: vec![pv(&[("hr", 16.0)]), pv(&[("hr", 9.0)])],
    }
}

fn revolve() -> Family {
    Family {
        name: "revolve",
        feature: "spin",
        inputs: vec![
            ("ri", "length", 8.0),
            ("ro", "length", 20.0),
            ("rh", "length", 15.0),
            ("ang", "angle", 270.0),
        ],
        build: |d| {
            // Zx plane: local x is world z, local y is world x.
            d.features.push(sketch(
                "s_rev",
                rect(
                    origin(OriginPlane::Zx),
                    mm(0.0),
                    input("ri"),
                    input("rh"),
                    input("ro"),
                ),
            ));
            d.features.push(Feature::new(
                "spin",
                FeatureKind::Revolve(Revolve {
                    profile: ProfileRef::sketch("s_rev"),
                    axis: AxisRef::origin(OriginAxis::Z),
                    angle: input("ang"),
                    mode: BodyMode::New,
                }),
            ));
        },
        probes: |p| {
            let side = |c| side("spin", "s_rev", c);
            let full = p["ang"] >= 360.0;
            let cap = |o: Oracle| {
                if full {
                    Expect::Problem("not_generated")
                } else {
                    ok1(o)
                }
            };
            vec![
                face1(
                    "rev_outer",
                    side("l3"),
                    "spin",
                    cylinder(p["ro"], 2, [0.0, 0.0]),
                ),
                face1(
                    "rev_inner",
                    side("l1"),
                    "spin",
                    cylinder(p["ri"], 2, [0.0, 0.0]),
                ),
                face1(
                    "rev_top",
                    side("l2"),
                    "spin",
                    plane_at([0.0, 0.0, 1.0], 2, p["rh"]),
                ),
                probe(
                    "rev_start",
                    TopoKind::Face,
                    Reference::start_cap("spin"),
                    Cardinality::ExactlyOne,
                    "spin",
                    cap(plane_at([0.0, 1.0, 0.0], 1, 0.0)),
                ),
                probe(
                    "rev_end",
                    TopoKind::Face,
                    Reference::end_cap("spin"),
                    Cardinality::ExactlyOne,
                    "spin",
                    cap(any()),
                ),
                probe(
                    "rev_body",
                    TopoKind::Body,
                    Reference::feature_body("spin"),
                    Cardinality::ExactlyOne,
                    "spin",
                    Expect::Ok(1, any()),
                ),
            ]
        },
        edits: vec![
            pv(&[("ang", 180.0)]),
            pv(&[("ro", 25.0)]),
            pv(&[("ang", 360.0)]),
        ],
    }
}

fn groove() -> Family {
    Family {
        name: "groove",
        feature: "groove",
        inputs: vec![
            ("gi", "length", 8.0),
            ("go", "length", 14.0),
            ("gdep", "length", 4.0),
        ],
        build: |d| {
            d.features
                .push(offset_plane("gplane", origin(OriginPlane::Zx), mm(25.0)));
            d.features.push(vertical_axis("gaxis", 40.0, 25.0));
            // Local x is world z, local y is world x.
            d.features.push(sketch(
                "s_gr",
                rect(
                    PlaneRef::datum("gplane"),
                    sub(input("thickness"), input("gdep")),
                    add(mm(40.0), input("gi")),
                    add(input("thickness"), mm(2.0)),
                    add(mm(40.0), input("go")),
                ),
            ));
            d.features.push(Feature::new(
                "groove",
                FeatureKind::Revolve(Revolve {
                    profile: ProfileRef::sketch("s_gr"),
                    axis: axis_datum("gaxis"),
                    angle: ValueExpr::deg(360.0),
                    mode: BodyMode::cut(),
                }),
            ));
        },
        probes: |p| {
            let side = |c| side("groove", "s_gr", c);
            vec![
                face1(
                    "g_out",
                    side("l3"),
                    "groove",
                    cylinder(p["go"], 2, [40.0, 25.0]),
                ),
                face1(
                    "g_in",
                    side("l1"),
                    "groove",
                    cylinder(p["gi"], 2, [40.0, 25.0]),
                ),
                face1(
                    "g_floor",
                    side("l4"),
                    "groove",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0 - p["gdep"]),
                ),
                // The ring splits the block's top into the island and the rest.
                probe(
                    "blk_top_pieces",
                    TopoKind::Face,
                    Reference::end_cap("body"),
                    Cardinality::AtLeastOne,
                    "body",
                    Expect::Ok(2, plane_at([0.0, 0.0, 1.0], 2, 10.0)),
                ),
                probe(
                    "blk_top_one",
                    TopoKind::Face,
                    Reference::end_cap("body"),
                    Cardinality::ExactlyOne,
                    "body",
                    Expect::Problem("ambiguous"),
                ),
            ]
        },
        edits: vec![
            pv(&[("go", 16.0)]),
            pv(&[("gdep", 6.0)]),
            pv(&[("gi", 5.0)]),
        ],
    }
}

fn sweep_doc(d: &mut PartDocument, third: bool) {
    d.features
        .push(offset_plane("swp", PlaneRef::origin_xy(), mm(20.0)));
    d.features.push(sketch(
        "s_swp",
        rect(
            PlaneRef::datum("swp"),
            mm(60.0),
            mm(20.0),
            mm(66.0),
            mm(26.0),
        ),
    ));
    d.features
        .push(offset_plane("pathp", origin(OriginPlane::Zx), mm(23.0)));
    // Local (u, v) = (world z, world x).
    let u1 = add(mm(20.0), input("rise"));
    let mut path = vec![
        line("p1", (mm(20.0), mm(63.0)), (u1.clone(), mm(63.0))),
        line(
            "p2",
            (u1.clone(), mm(63.0)),
            (add(u1.clone(), mm(10.0)), add(mm(63.0), input("run"))),
        ),
    ];
    if third {
        path.push(line(
            "p3",
            (add(u1.clone(), mm(10.0)), add(mm(63.0), input("run"))),
            (add(u1, mm(25.0)), add(mm(63.0), input("run"))),
        ));
    }
    d.features
        .push(sketch("s_path", sketch_of(PlaneRef::datum("pathp"), path)));
    d.features.push(Feature::new(
        "sw",
        FeatureKind::Sweep(Sweep {
            profile: ProfileRef::sketch("s_swp"),
            path: ProfileRef::sketch("s_path"),
            mode: BodyMode::New,
        }),
    ));
}

fn sweep_side(ordinal: Option<u32>) -> Reference {
    Reference::face(EntityName::Generated(GenerationName {
        feature: fid("sw"),
        role: Role::Side,
        tag: SourceTag::Curve {
            sketch: fid("s_swp"),
            entity: runmat_cad::ids::SketchEntityId::new("l2"),
        },
        ordinal,
    }))
}

fn sweep_probes(p: &Pv, segments: usize) -> Vec<Probe> {
    let end_z = 20.0 + p["rise"] + 10.0 + if segments == 3 { 15.0 } else { 0.0 };
    vec![
        face1(
            "sw_l2_0",
            sweep_side(Some(0)),
            "sw",
            plane_at([1.0, 0.0, 0.0], 0, 66.0),
        ),
        face1(
            "sw_l2_1",
            sweep_side(Some(1)),
            "sw",
            plane_along([10.0, 0.0, -p["run"]]),
        ),
        probe(
            "sw_l2",
            TopoKind::Face,
            sweep_side(None),
            Cardinality::AtLeastOne,
            "sw",
            Expect::Ok(segments, any()),
        ),
        face1(
            "sw_start",
            Reference::start_cap("sw"),
            "sw",
            plane_at([0.0, 0.0, 1.0], 2, 20.0),
        ),
        face1(
            "sw_end",
            Reference::end_cap("sw"),
            "sw",
            plane_at([0.0, 0.0, 1.0], 2, end_z),
        ),
    ]
}

fn sweep() -> Family {
    Family {
        name: "sweep",
        feature: "sw",
        inputs: vec![("rise", "length", 20.0), ("run", "length", 10.0)],
        build: |d| sweep_doc(d, false),
        probes: |p| sweep_probes(p, 2),
        edits: vec![pv(&[("rise", 28.0)]), pv(&[("run", 14.0)])],
    }
}

fn loft_doc(d: &mut PartDocument, third: bool) {
    let sq = |id: &str, plane: &str, side: ValueExpr| {
        sketch(
            id,
            rect(
                PlaneRef::datum(plane),
                sub(mm(20.0), half(side.clone())),
                sub(mm(25.0), half(side.clone())),
                add(mm(20.0), half(side.clone())),
                add(mm(25.0), half(side)),
            ),
        )
    };
    d.features
        .push(offset_plane("lp0", PlaneRef::origin_xy(), mm(20.0)));
    d.features.push(offset_plane(
        "lp1",
        PlaneRef::origin_xy(),
        add(mm(20.0), input("lz")),
    ));
    d.features.push(sq("s_l0", "lp0", input("la")));
    d.features.push(sq("s_l1", "lp1", input("lb")));
    let mut profiles = vec![ProfileRef::sketch("s_l0"), ProfileRef::sketch("s_l1")];
    if third {
        d.features.push(offset_plane(
            "lp2",
            PlaneRef::origin_xy(),
            add(mm(40.0), input("lz")),
        ));
        d.features.push(sq("s_l2", "lp2", input("la")));
        profiles.push(ProfileRef::sketch("s_l2"));
    }
    d.features.push(Feature::new(
        "lf",
        FeatureKind::Loft(Loft {
            profiles,
            guides: Vec::new(),
            mode: BodyMode::New,
        }),
    ));
}

fn loft_side(ordinal: Option<u32>) -> Reference {
    Reference::face(EntityName::Generated(GenerationName {
        feature: fid("lf"),
        role: Role::Side,
        tag: SourceTag::Curve {
            sketch: fid("s_l0"),
            entity: runmat_cad::ids::SketchEntityId::new("l1"),
        },
        ordinal,
    }))
}

fn loft() -> Family {
    Family {
        name: "loft",
        feature: "lf",
        inputs: vec![
            ("la", "length", 20.0),
            ("lb", "length", 12.0),
            ("lz", "length", 25.0),
        ],
        build: |d| loft_doc(d, false),
        probes: |p| {
            vec![
                face1(
                    "lf_side1",
                    loft_side(None),
                    "lf",
                    centroid_near([Some(20.0), None, Some(20.0 + p["lz"] / 2.0)], 2.5),
                ),
                face1(
                    "lf_top",
                    Reference::end_cap("lf"),
                    "lf",
                    plane_at([0.0, 0.0, 1.0], 2, 20.0 + p["lz"]),
                ),
                face1(
                    "lf_bottom",
                    Reference::start_cap("lf"),
                    "lf",
                    plane_at([0.0, 0.0, 1.0], 2, 20.0),
                ),
            ]
        },
        edits: vec![pv(&[("lb", 16.0)]), pv(&[("lz", 30.0)])],
    }
}

fn shell_inner(face: &Reference) -> Reference {
    Reference::gen_face("sh", Role::ShellInner, SourceTag::face_of(&name_of(face)))
}

fn shell() -> Family {
    Family {
        name: "shell",
        feature: "sh",
        inputs: vec![("wall", "length", 2.0)],
        build: |d| {
            d.features.push(Feature::new(
                "sh",
                FeatureKind::Shell(Shell {
                    body: Reference::feature_body("body"),
                    open_faces: vec![Reference::end_cap("body")],
                    thickness: input("wall"),
                    faces: Vec::new(),
                    outward: false,
                }),
            ));
        },
        probes: |p| {
            let front = side("body", "s1", "l1");
            vec![
                face1(
                    "sh_front_in",
                    shell_inner(&front),
                    "sh",
                    plane_at([0.0, 1.0, 0.0], 1, p["wall"]),
                ),
                face1(
                    "sh_floor",
                    shell_inner(&Reference::start_cap("body")),
                    "sh",
                    plane_at([0.0, 0.0, 1.0], 2, p["wall"]),
                ),
                face1(
                    "sh_rim",
                    Reference::gen_face(
                        "sh",
                        Role::ShellRim,
                        SourceTag::face_of(&name_of(&Reference::end_cap("body"))),
                    ),
                    "sh",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0),
                ),
                face1("sh_front", front, "body", plane_at([0.0, 1.0, 0.0], 1, 0.0)),
                probe(
                    "sh_opened",
                    TopoKind::Face,
                    Reference::end_cap("body"),
                    Cardinality::ExactlyOne,
                    "body",
                    Expect::Problem("consumed"),
                )
                .suppressed(ok1(plane_at([0.0, 0.0, 1.0], 2, 10.0))),
            ]
        },
        edits: vec![pv(&[("wall", 3.0)]), pv(&[("width", 60.0)])],
    }
}

fn draft() -> Family {
    Family {
        name: "draft",
        feature: "dr",
        inputs: vec![("da", "angle", 5.0)],
        build: |d| {
            d.features.push(Feature::new(
                "dr",
                FeatureKind::Draft(Draft {
                    faces: vec![side("body", "s1", "l1"), side("body", "s1", "l3")],
                    neutral: PlaneRef::origin_xy(),
                    angle: input("da"),
                    kind: Default::default(),
                    flip: false,
                }),
            ));
        },
        probes: |p| {
            let a = p["da"].to_radians();
            vec![
                face1(
                    "dr_front",
                    side("body", "s1", "l1"),
                    "body",
                    plane_along([0.0, -a.cos(), a.sin()]),
                )
                .suppressed(ok1(plane_at([0.0, 1.0, 0.0], 1, 0.0))),
                face1(
                    "dr_back",
                    side("body", "s1", "l3"),
                    "body",
                    plane_along([0.0, a.cos(), a.sin()]),
                )
                .suppressed(ok1(plane_at([0.0, 1.0, 0.0], 1, p["depth"]))),
                face1(
                    "dr_top",
                    Reference::end_cap("body"),
                    "body",
                    plane_at([0.0, 0.0, 1.0], 2, p["thickness"]),
                ),
                face1(
                    "dr_right",
                    side("body", "s1", "l2"),
                    "body",
                    plane_at([1.0, 0.0, 0.0], 0, p["width"]),
                ),
            ]
        },
        edits: vec![pv(&[("da", 8.0)]), pv(&[("thickness", 14.0)])],
    }
}

fn hole() -> Family {
    Family {
        name: "hole",
        feature: "h",
        inputs: vec![("hd", "length", 6.0), ("hdep", "length", 5.0)],
        build: |d| {
            d.features.push(hole_feature(
                "h",
                20.0,
                25.0,
                input("hd"),
                Some(input("hdep")),
                HoleForm::Simple,
            ))
        },
        probes: |p| {
            let wall = hole_face("h", Role::HoleWall);
            vec![
                face1(
                    "h_wall",
                    wall.clone(),
                    "h",
                    cylinder(p["hd"] / 2.0, 2, [20.0, 25.0]),
                ),
                face1(
                    "h_bottom",
                    hole_face("h", Role::HoleBottom),
                    "h",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0 - p["hdep"]),
                ),
                probe(
                    "h_edge",
                    TopoKind::Edge,
                    Reference::edges_between(wall, Reference::end_cap("body")),
                    Cardinality::ExactlyOne,
                    "h",
                    Expect::Ok(1, any()),
                ),
            ]
        },
        edits: vec![pv(&[("hd", 8.0)]), pv(&[("hdep", 7.0)])],
    }
}

fn counterbore() -> Family {
    Family {
        name: "counterbore",
        feature: "cb",
        inputs: vec![("cbd", "length", 12.0), ("cbh", "length", 3.0)],
        build: |d| {
            d.features.push(hole_feature(
                "cb",
                50.0,
                25.0,
                mm(6.0),
                Some(mm(8.0)),
                HoleForm::Counterbore {
                    diameter: Some(input("cbd")),
                    depth: Some(input("cbh")),
                },
            ))
        },
        probes: |p| {
            vec![
                face1(
                    "cb_floor",
                    hole_face("cb", Role::HoleCounterbore),
                    "cb",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0 - p["cbh"]),
                ),
                face1(
                    "cb_wall",
                    hole_face("cb", Role::HoleCounterboreWall),
                    "cb",
                    cylinder(p["cbd"] / 2.0, 2, [50.0, 25.0]),
                ),
                face1(
                    "cb_hole",
                    hole_face("cb", Role::HoleWall),
                    "cb",
                    cylinder(3.0, 2, [50.0, 25.0]),
                ),
                face1(
                    "cb_bottom",
                    hole_face("cb", Role::HoleBottom),
                    "cb",
                    plane_at([0.0, 0.0, 1.0], 2, 2.0),
                ),
            ]
        },
        edits: vec![pv(&[("cbd", 14.0)]), pv(&[("cbh", 4.0)])],
    }
}

fn countersink() -> Family {
    Family {
        name: "countersink",
        feature: "cs",
        inputs: vec![("csd", "length", 12.0), ("csa", "angle", 90.0)],
        build: |d| {
            d.features.push(hole_feature(
                "cs",
                60.0,
                25.0,
                mm(6.0),
                Some(mm(8.0)),
                HoleForm::Countersink {
                    diameter: Some(input("csd")),
                    angle: Some(input("csa")),
                },
            ))
        },
        probes: |_| {
            vec![
                face1(
                    "cs_cone",
                    hole_face("cs", Role::HoleCountersink),
                    "cs",
                    centroid_near([Some(60.0), Some(25.0), None], 0.6),
                ),
                face1(
                    "cs_hole",
                    hole_face("cs", Role::HoleWall),
                    "cs",
                    cylinder(3.0, 2, [60.0, 25.0]),
                ),
                face1(
                    "cs_bottom",
                    hole_face("cs", Role::HoleBottom),
                    "cs",
                    plane_at([0.0, 0.0, 1.0], 2, 2.0),
                ),
            ]
        },
        edits: vec![pv(&[("csd", 10.0)]), pv(&[("csa", 82.0)])],
    }
}

fn copy_face(base: &Reference, pattern: &str, instance: InstanceKey) -> Reference {
    Reference::face(EntityName::copy(&name_of(base), &fid(pattern), instance))
}

fn linear_pattern() -> Family {
    Family {
        name: "linear_pattern",
        feature: "row",
        inputs: vec![("sp", "length", 15.0), ("n", "count", 4.0)],
        build: |d| {
            d.features.push(hole_feature(
                "h",
                10.0,
                25.0,
                mm(4.0),
                None,
                HoleForm::Simple,
            ));
            d.features.push(Feature::new(
                "row",
                FeatureKind::Pattern(Pattern {
                    seeds: PatternSeeds::Features {
                        features: vec![fid("h")],
                    },
                    layout: PatternLayout::Linear {
                        direction: AxisRef::origin(OriginAxis::X),
                        spacing: input("sp"),
                        count: input("n"),
                    },
                    skip: Vec::new(),
                }),
            ));
        },
        probes: |p| {
            let wall = hole_face("h", Role::HoleWall);
            let inst = |k: u32| {
                let x = 10.0 + k as f64 * p["sp"];
                if (k as f64) < p["n"] {
                    ok1(cylinder(2.0, 2, [x, 25.0]))
                } else {
                    Expect::Problem("instance_removed")
                }
            };
            vec![
                probe(
                    "row_w1",
                    TopoKind::Face,
                    copy_face(&wall, "row", InstanceKey::Linear { step: 1 }),
                    Cardinality::ExactlyOne,
                    "row",
                    inst(1),
                ),
                probe(
                    "row_w2",
                    TopoKind::Face,
                    copy_face(&wall, "row", InstanceKey::Linear { step: 2 }),
                    Cardinality::ExactlyOne,
                    "row",
                    inst(2),
                ),
                face1("row_seed", wall, "h", cylinder(2.0, 2, [10.0, 25.0])),
            ]
        },
        edits: vec![pv(&[("n", 5.0)]), pv(&[("sp", 12.0)]), pv(&[("n", 2.0)])],
    }
}

fn rotate_about(x: f64, y: f64, deg: f64) -> (f64, f64) {
    let (s, c) = deg.to_radians().sin_cos();
    let (dx, dy) = (x - 40.0, y - 25.0);
    (40.0 + c * dx - s * dy, 25.0 + s * dx + c * dy)
}

fn circular_pattern() -> Family {
    Family {
        name: "circular_pattern",
        feature: "ring",
        inputs: vec![("n", "count", 4.0)],
        build: |d| {
            d.features.push(top_plane());
            d.features.push(sketch(
                "s_pin",
                rect(
                    PlaneRef::datum("top_plane"),
                    mm(55.0),
                    mm(23.0),
                    mm(59.0),
                    mm(27.0),
                ),
            ));
            d.features
                .push(extrude("pin", "s_pin", mm(5.0), BodyMode::add()));
            d.features.push(vertical_axis("cax", 40.0, 25.0));
            d.features.push(Feature::new(
                "ring",
                FeatureKind::Pattern(Pattern {
                    seeds: PatternSeeds::Features {
                        features: vec![fid("pin")],
                    },
                    layout: PatternLayout::Circular {
                        axis: axis_datum("cax"),
                        count: input("n"),
                        angle: ValueExpr::deg(360.0),
                        equal_spacing: true,
                    },
                    skip: Vec::new(),
                }),
            ));
        },
        probes: |p| {
            let top = Reference::end_cap("pin");
            let inst = |k: u32| {
                if (k as f64) < p["n"] {
                    let (x, y) = rotate_about(57.0, 25.0, 360.0 * k as f64 / p["n"]);
                    ok1(both(
                        plane_at([0.0, 0.0, 1.0], 2, 15.0),
                        centroid_near([Some(x), Some(y), None], 0.6),
                    ))
                } else {
                    Expect::Problem("instance_removed")
                }
            };
            vec![
                probe(
                    "ring_top1",
                    TopoKind::Face,
                    copy_face(&top, "ring", InstanceKey::Circular { step: 1 }),
                    Cardinality::ExactlyOne,
                    "ring",
                    inst(1),
                ),
                probe(
                    "ring_top2",
                    TopoKind::Face,
                    copy_face(&top, "ring", InstanceKey::Circular { step: 2 }),
                    Cardinality::ExactlyOne,
                    "ring",
                    inst(2),
                ),
                face1(
                    "pin_top",
                    top,
                    "pin",
                    both(
                        plane_at([0.0, 0.0, 1.0], 2, 15.0),
                        centroid_near([Some(57.0), Some(25.0), None], 1e-6),
                    ),
                ),
            ]
        },
        edits: vec![pv(&[("n", 6.0)]), pv(&[("n", 3.0)]), pv(&[("n", 2.0)])],
    }
}

fn mirror() -> Family {
    Family {
        name: "mirror",
        feature: "mir",
        inputs: vec![("md", "length", 3.0)],
        build: |d| {
            d.features.push(top_plane());
            d.features.push(sketch(
                "s_pk2",
                rect(
                    PlaneRef::datum("top_plane"),
                    mm(25.0),
                    mm(5.0),
                    mm(35.0),
                    mm(15.0),
                ),
            ));
            d.features.push(extrude_reverse(
                "pk2",
                "s_pk2",
                input("md"),
                BodyMode::cut(),
            ));
            // The plane x = y: the image swaps x and y.
            d.features.push(Feature::new(
                "mp",
                FeatureKind::Plane(PlaneDef::Angle {
                    base: origin(OriginPlane::Yz),
                    axis: AxisRef::origin(OriginAxis::Z),
                    angle: ValueExpr::deg(-45.0),
                }),
            ));
            d.features.push(Feature::new(
                "mir",
                FeatureKind::Mirror(Mirror {
                    seeds: PatternSeeds::Features {
                        features: vec![fid("pk2")],
                    },
                    plane: PlaneRef::datum("mp"),
                }),
            ));
        },
        probes: |p| {
            let floor = Reference::end_cap("pk2");
            let wall = side("pk2", "s_pk2", "l1");
            vec![
                face1(
                    "mir_floor",
                    copy_face(&floor, "mir", InstanceKey::Mirror),
                    "mir",
                    both(
                        plane_at([0.0, 0.0, 1.0], 2, 10.0 - p["md"]),
                        centroid_near([Some(10.0), Some(30.0), None], 1e-6),
                    ),
                ),
                face1(
                    "mir_wall",
                    copy_face(&wall, "mir", InstanceKey::Mirror),
                    "mir",
                    plane_at([1.0, 0.0, 0.0], 0, 5.0),
                ),
                face1(
                    "seed_floor",
                    floor,
                    "pk2",
                    centroid_near([Some(30.0), Some(10.0), Some(10.0 - p["md"])], 1e-6),
                ),
            ]
        },
        edits: vec![pv(&[("md", 5.0)])],
    }
}

fn body_pattern() -> Family {
    Family {
        name: "body_pattern",
        feature: "bring",
        inputs: vec![("n", "count", 3.0)],
        build: |d| {
            d.features
                .push(offset_plane("nubp", PlaneRef::origin_xy(), mm(20.0)));
            d.features.push(sketch(
                "s_nub",
                rect(
                    PlaneRef::datum("nubp"),
                    mm(60.0),
                    mm(23.0),
                    mm(64.0),
                    mm(27.0),
                ),
            ));
            d.features
                .push(extrude("nub", "s_nub", mm(4.0), BodyMode::New));
            d.features.push(vertical_axis("cax", 40.0, 25.0));
            d.features.push(Feature::new(
                "bring",
                FeatureKind::Pattern(Pattern {
                    seeds: PatternSeeds::Bodies {
                        bodies: vec![Reference::feature_body("nub")],
                    },
                    layout: PatternLayout::Circular {
                        axis: axis_datum("cax"),
                        count: input("n"),
                        angle: ValueExpr::deg(360.0),
                        equal_spacing: true,
                    },
                    skip: Vec::new(),
                }),
            ));
        },
        probes: |p| {
            let (x, y) = rotate_about(62.0, 25.0, 360.0 / p["n"]);
            vec![
                face1(
                    "bring_top1",
                    copy_face(
                        &Reference::end_cap("nub"),
                        "bring",
                        InstanceKey::Circular { step: 1 },
                    ),
                    "bring",
                    both(
                        plane_at([0.0, 0.0, 1.0], 2, 24.0),
                        centroid_near([Some(x), Some(y), None], 0.6),
                    ),
                ),
                probe(
                    "bring_body1",
                    TopoKind::Body,
                    Reference::body(EntityName::copy(
                        &name_of(&Reference::feature_body("nub")),
                        &fid("bring"),
                        InstanceKey::Circular { step: 1 },
                    )),
                    Cardinality::ExactlyOne,
                    "bring",
                    Expect::Ok(1, any()),
                ),
                face1(
                    "nub_top",
                    Reference::end_cap("nub"),
                    "nub",
                    plane_at([0.0, 0.0, 1.0], 2, 24.0),
                ),
            ]
        },
        edits: vec![pv(&[("n", 4.0)]), pv(&[("n", 5.0)])],
    }
}

fn blend_face(feature: &str, role: Role, a: &Reference, b: &Reference) -> Reference {
    Reference::gen_face(
        feature,
        role,
        SourceTag::edge_between(&name_of(a), &name_of(b)),
    )
}

fn fillet_family() -> Family {
    Family {
        name: "fillet",
        feature: "rnd",
        inputs: vec![("fr", "length", 2.0)],
        build: |d| {
            d.features.push(Feature::new(
                "rnd",
                FeatureKind::Fillet(Fillet {
                    edges: vec![Reference::edges_between(
                        side("body", "s1", "l1"),
                        Reference::end_cap("body"),
                    )],
                    radius: input("fr"),
                    variable: Vec::new(),
                    profile: Default::default(),
                    setbacks: Vec::new(),
                    propagate: true,
                }),
            ));
        },
        probes: |p| {
            let front = side("body", "s1", "l1");
            let top = Reference::end_cap("body");
            vec![
                face1(
                    "rnd_face",
                    blend_face("rnd", Role::Blend, &front, &top),
                    "rnd",
                    Rc::new({
                        let r = p["fr"];
                        move |g| match face_of(g)?.surface {
                            SurfaceKind::Cylinder { radius, .. } => {
                                close(radius * 1000.0, r, 1e-9, "blend radius")
                            }
                            other => Err(format!("{:?}", other)),
                        }
                    }),
                ),
                face1(
                    "rnd_front",
                    front,
                    "body",
                    plane_at([0.0, 1.0, 0.0], 1, 0.0),
                ),
                face1("rnd_top", top, "body", plane_at([0.0, 0.0, 1.0], 2, 10.0)),
            ]
        },
        edits: vec![pv(&[("fr", 3.0)]), pv(&[("width", 60.0)])],
    }
}

fn chamfer_family() -> Family {
    Family {
        name: "chamfer",
        feature: "chf",
        inputs: vec![("cd", "length", 1.5)],
        build: |d| {
            d.features.push(Feature::new(
                "chf",
                FeatureKind::Chamfer(Chamfer {
                    edges: vec![Reference::edges_between(
                        side("body", "s1", "l2"),
                        Reference::end_cap("body"),
                    )],
                    form: ChamferForm::Equal {
                        distance: input("cd"),
                    },
                    flip: false,
                }),
            ));
        },
        probes: |p| {
            let right = side("body", "s1", "l2");
            let top = Reference::end_cap("body");
            vec![
                face1(
                    "chf_face",
                    blend_face("chf", Role::Chamfer, &right, &top),
                    "chf",
                    centroid_near([Some(p["width"]), Some(p["depth"] / 2.0), Some(10.0)], 1e-6),
                ),
                face1(
                    "chf_right",
                    right,
                    "body",
                    plane_at([1.0, 0.0, 0.0], 0, p["width"]),
                ),
            ]
        },
        edits: vec![pv(&[("cd", 2.0)]), pv(&[("depth", 40.0)])],
    }
}

fn split_family() -> Family {
    Family {
        name: "split",
        feature: "halve",
        inputs: vec![("sx", "length", 15.0)],
        build: |d| {
            d.features
                .push(offset_plane("cutp", origin(OriginPlane::Yz), input("sx")));
            d.features.push(Feature::new(
                "halve",
                FeatureKind::Split(Split {
                    body: Reference::feature_body("body"),
                    tool: PlaneRef::datum("cutp"),
                }),
            ));
        },
        probes: |p| {
            let half_body = |side| {
                Reference::body(EntityName::generated(
                    "halve",
                    Role::Body,
                    SourceTag::Side { side },
                ))
            };
            let even = (p["sx"] - p["width"] / 2.0).abs() < 1e-9;
            vec![
                probe(
                    "half_pos",
                    TopoKind::Body,
                    half_body(PlaneSide::Positive),
                    Cardinality::ExactlyOne,
                    "halve",
                    Expect::Ok(1, any()),
                ),
                probe(
                    "half_neg",
                    TopoKind::Body,
                    half_body(PlaneSide::Negative),
                    Cardinality::ExactlyOne,
                    "halve",
                    Expect::Ok(1, any()),
                ),
                face1(
                    "split_pos",
                    Reference::gen_face(
                        "halve",
                        Role::SplitFace,
                        SourceTag::Side {
                            side: PlaneSide::Positive,
                        },
                    ),
                    "halve",
                    plane_at([1.0, 0.0, 0.0], 0, p["sx"]),
                ),
                // The original body's slot: the dominant piece, or a reported tie.
                probe(
                    "blk_body",
                    TopoKind::Body,
                    Reference::feature_body("body"),
                    Cardinality::ExactlyOne,
                    "body",
                    if even {
                        Expect::Problem("ambiguous")
                    } else {
                        Expect::Ok(1, any())
                    },
                ),
            ]
        },
        edits: vec![pv(&[("sx", 12.0)]), pv(&[("sx", 40.0)])],
    }
}

fn boolean_family() -> Family {
    Family {
        name: "boolean",
        feature: "uni",
        inputs: vec![("tz", "length", 10.0)],
        build: |d| {
            d.features.push(sketch(
                "s_t",
                rect(
                    PlaneRef::origin_xy(),
                    mm(70.0),
                    mm(10.0),
                    mm(90.0),
                    mm(20.0),
                ),
            ));
            d.features
                .push(extrude("tool", "s_t", input("tz"), BodyMode::New));
            d.features.push(Feature::new(
                "uni",
                FeatureKind::Boolean(Boolean {
                    kind: BooleanKind::Union,
                    target: Reference::feature_body("body"),
                    tools: vec![Reference::feature_body("tool")],
                    keep_tools: false,
                }),
            ));
        },
        probes: |p| {
            vec![
                face1(
                    "tool_right",
                    side("tool", "s_t", "l2"),
                    "tool",
                    plane_at([1.0, 0.0, 0.0], 0, 90.0),
                ),
                face1(
                    "tool_top",
                    Reference::end_cap("tool"),
                    "tool",
                    plane_at([0.0, 0.0, 1.0], 2, p["tz"]),
                ),
                face1(
                    "blk_top",
                    Reference::end_cap("body"),
                    "body",
                    plane_at([0.0, 0.0, 1.0], 2, 10.0),
                ),
                probe(
                    "blk_body",
                    TopoKind::Body,
                    Reference::feature_body("body"),
                    Cardinality::ExactlyOne,
                    "body",
                    Expect::Ok(1, any()),
                ),
            ]
        },
        edits: vec![pv(&[("tz", 14.0)])],
    }
}

fn outward_shell() -> Family {
    Family {
        name: "outward shell",
        feature: "osh",
        inputs: vec![("wall", "length", 2.0)],
        build: |d| {
            let mut s = Shell::new(
                Reference::feature_body("body"),
                vec![Reference::end_cap("body")],
                input("wall"),
            );
            s.outward = true;
            d.features.push(Feature::new("osh", FeatureKind::Shell(s)));
        },
        probes: |p| {
            let front = side("body", "s1", "l1");
            let top = Reference::end_cap("body");
            let outer = |f: &Reference| {
                Reference::gen_face("osh", Role::ShellOuter, SourceTag::face_of(&name_of(f)))
            };
            vec![
                face1(
                    "osh_front_out",
                    outer(&front),
                    "osh",
                    plane_at([0.0, 1.0, 0.0], 1, -p["wall"]),
                ),
                face1(
                    "osh_bottom_out",
                    outer(&Reference::start_cap("body")),
                    "osh",
                    plane_at([0.0, 0.0, 1.0], 2, -p["wall"]),
                ),
                face1(
                    "osh_rim",
                    Reference::gen_face("osh", Role::ShellRim, SourceTag::face_of(&name_of(&top))),
                    "osh",
                    plane_at([0.0, 0.0, 1.0], 2, p["thickness"]),
                ),
                // The body's own faces line the cavity.
                face1(
                    "osh_front_in",
                    front,
                    "body",
                    plane_at([0.0, 1.0, 0.0], 1, 0.0),
                ),
                probe(
                    "osh_opened",
                    TopoKind::Face,
                    top,
                    Cardinality::ExactlyOne,
                    "body",
                    Expect::Problem("consumed"),
                )
                .suppressed(ok1(plane_at([0.0, 0.0, 1.0], 2, p["thickness"]))),
            ]
        },
        edits: vec![
            pv(&[("wall", 3.0)]),
            pv(&[("width", 60.0)]),
            pv(&[("thickness", 14.0)]),
        ],
    }
}

fn vertex_chamfer() -> Family {
    Family {
        name: "vertex chamfer",
        feature: "vc",
        inputs: vec![("vd", "length", 6.0)],
        build: |d| {
            d.features.push(Feature::new(
                "vc",
                FeatureKind::Chamfer(Chamfer {
                    edges: vec![Reference::vertex_of(vec![
                        Reference::end_cap("body"),
                        side("body", "s1", "l1"),
                        side("body", "s1", "l2"),
                    ])],
                    form: ChamferForm::Vertex {
                        distances: [input("vd"), input("vd"), input("vd")],
                    },
                    flip: false,
                }),
            ));
        },
        probes: |p| {
            let faces = [
                name_of(&Reference::end_cap("body")),
                name_of(&side("body", "s1", "l1")),
                name_of(&side("body", "s1", "l2")),
            ];
            let v = p["vd"];
            vec![
                // The triangle's centroid, a third of the way in along each edge (the fake
                // kernel's staircase lands within a tenth of the size).
                face1(
                    "vc_face",
                    Reference::gen_face("vc", Role::Chamfer, SourceTag::vertex_of(&faces)),
                    "vc",
                    centroid_near(
                        [
                            Some(p["width"] - v / 3.0),
                            Some(v / 3.0),
                            Some(p["thickness"] - v / 3.0),
                        ],
                        v / 10.0,
                    ),
                ),
                face1(
                    "vc_front",
                    side("body", "s1", "l1"),
                    "body",
                    plane_at([0.0, 1.0, 0.0], 1, 0.0),
                ),
                face1(
                    "vc_top",
                    Reference::end_cap("body"),
                    "body",
                    plane_at([0.0, 0.0, 1.0], 2, p["thickness"]),
                ),
            ]
        },
        edits: vec![
            pv(&[("vd", 4.0)]),
            pv(&[("width", 60.0)]),
            pv(&[("thickness", 14.0)]),
        ],
    }
}

fn cut_with_surface() -> Family {
    Family {
        name: "cut with surface",
        feature: "trim",
        inputs: vec![("level", "length", 4.0)],
        build: |d| {
            d.features
                .push(offset_plane("lvl", origin(OriginPlane::Xy), input("level")));
            d.features.push(sketch(
                "s_sheet",
                rect_names(
                    PlaneRef::datum("lvl"),
                    -10.0,
                    -10.0,
                    100.0,
                    70.0,
                    ["sa", "sb", "sc", "sd"],
                ),
            ));
            d.features.push(Feature::new(
                "sheet",
                FeatureKind::Surface(SurfaceFeature::Planar {
                    profile: ProfileRef::sketch("s_sheet"),
                }),
            ));
            d.features.push(Feature::new(
                "trim",
                FeatureKind::Surface(SurfaceFeature::CutWithSurface {
                    body: Reference::feature_body("body"),
                    surface: Reference::feature_body("sheet"),
                    keep: PlaneSide::Negative,
                }),
            ));
        },
        probes: |p| {
            vec![
                face1(
                    "trim_cut",
                    Reference::gen_face(
                        "trim",
                        Role::SplitFace,
                        SourceTag::Side {
                            side: PlaneSide::Negative,
                        },
                    ),
                    "trim",
                    plane_at([0.0, 0.0, 1.0], 2, p["level"]),
                ),
                face1(
                    "trim_front",
                    side("body", "s1", "l1"),
                    "body",
                    plane_at([0.0, 1.0, 0.0], 1, 0.0),
                ),
                face1(
                    "trim_floor",
                    Reference::start_cap("body"),
                    "body",
                    plane_at([0.0, 0.0, 1.0], 2, 0.0),
                ),
                probe(
                    "trim_top",
                    TopoKind::Face,
                    Reference::end_cap("body"),
                    Cardinality::ExactlyOne,
                    "body",
                    Expect::Problem("consumed"),
                )
                .suppressed(ok1(plane_at([0.0, 0.0, 1.0], 2, p["thickness"]))),
            ]
        },
        edits: vec![pv(&[("level", 6.0)]), pv(&[("width", 60.0)])],
    }
}

fn families() -> Vec<Family> {
    vec![
        arc_boss(),
        circle_pocket(),
        hex_cut(),
        revolve(),
        groove(),
        sweep(),
        loft(),
        shell(),
        draft(),
        hole(),
        counterbore(),
        countersink(),
        linear_pattern(),
        circular_pattern(),
        mirror(),
        body_pattern(),
        fillet_family(),
        chamfer_family(),
        split_family(),
        boolean_family(),
        outward_shell(),
        vertex_chamfer(),
        cut_with_surface(),
    ]
}

// ------------------------------------------------------------------------------- checking

/// Geometry of each entity a region resolved, found by its names in the final state.
fn region_geoms(r: &RegenResult, name: &str, kind: TopoKind) -> Vec<Geom> {
    let o = &r.regions[&runmat_cad::ids::RegionName::new(name)];
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for names in &o.entities {
        match kind {
            TopoKind::Face => {
                for (bi, b) in r.state.bodies.iter().enumerate() {
                    for (fi, set) in b.names.faces.iter().enumerate() {
                        if set.len() == names.len()
                            && names.iter().all(|n| set.contains(n))
                            && seen.insert((bi, fi))
                        {
                            out.push(Geom::Face(b.topology.faces[fi].geom));
                        }
                    }
                }
            }
            TopoKind::Edge => {
                for b in &r.state.bodies {
                    for ei in 0..b.topology.edges.len() {
                        if let Some(pair) = edge_name(&b.names, &b.topology, ei as u32) {
                            if pair.to_vec() == *names {
                                out.push(Geom::Edge(b.topology.edges[ei].geom));
                            }
                        }
                    }
                }
            }
            _ => out.push(Geom::Body),
        }
    }
    out
}

fn problem_kind(p: &RefProblem) -> String {
    serde_json::to_value(p).unwrap()["kind"]
        .as_str()
        .unwrap()
        .to_owned()
}

fn check_probe(r: &RegenResult, p: &Probe, expect: &Expect) -> Result<(), String> {
    let o = r
        .regions
        .get(&runmat_cad::ids::RegionName::new(p.name))
        .ok_or("region missing")?;
    let outcome = &o.references.first().ok_or("no reference record")?.outcome;
    match (expect, outcome) {
        (Expect::Ok(n, oracle), ResolveOutcome::Resolved(res)) => {
            if res.class == ResolutionClass::Geometric {
                return Err("resolved by geometry".into());
            }
            if o.entities.len() != *n {
                return Err(format!("{} entities, expected {}", o.entities.len(), n));
            }
            let gs = region_geoms(r, p.name, p.kind);
            if gs.len() != *n {
                return Err(format!("{} geometries for {} entities", gs.len(), n));
            }
            for g in &gs {
                if let Geom::Edge(e) = g {
                    if e.length <= 0.0 {
                        return Err("an edge of no length".into());
                    }
                }
                oracle(g)?;
            }
            Ok(())
        }
        (Expect::Problem("ambiguous"), ResolveOutcome::Resolved(res))
            if p.card == Cardinality::ExactlyOne && res.len() > 1 =>
        {
            // The item checks its cardinality: it must fail and say so.
            if o.ok {
                return Err("several entities accepted where one is needed".into());
            }
            // A region reports its expectation with one candidate per entity.
            let told = r.diagnostics.diagnostics.iter().any(|x| {
                x.code == DiagCode::RefAmbiguous
                    || (x.code == DiagCode::RegionExpectation
                        && x.repairs.iter().any(|rp| {
                            matches!(rp, Repair::Disambiguate { candidates, .. }
                                if candidates.len() == res.len())
                        }))
            });
            if !told {
                return Err("no ambiguity diagnostic with a candidate per entity".into());
            }
            Ok(())
        }
        (Expect::Problem(k), ResolveOutcome::Failed(prob)) => {
            let got = problem_kind(prob);
            if got == *k {
                Ok(())
            } else {
                Err(format!("problem {} (expected {}): {:?}", got, k, prob))
            }
        }
        (Expect::Waiting, ResolveOutcome::Waiting(_)) => Ok(()),
        (e, o) => Err(format!(
            "expected {}, got {:?}",
            match e {
                Expect::Ok(n, _) => format!("ok({})", n),
                Expect::Problem(k) => format!("problem {}", k),
                Expect::Waiting => "waiting".into(),
            },
            o
        )),
    }
}

/// Invariants of every regenerated state.
fn check_state(d: &PartDocument, r: &RegenResult) -> Result<(), String> {
    for dg in &r.diagnostics.diagnostics {
        if matches!(
            dg.code,
            DiagCode::KernelNotSupported
                | DiagCode::KernelFailed
                | DiagCode::KernelHistoryIncomplete
                | DiagCode::RefTokenCollision
                | DiagCode::RefFallbackUsed
        ) {
            return Err(format!("{}: {}", dg.code, dg.message));
        }
    }
    if r.stats.geometric != 0 {
        return Err("a reference resolved by geometry".into());
    }
    // Every canonical reference the state offers selects its own entity; a qualified piece
    // selects it alone.
    let st = &r.state;
    let views: Vec<BodyView<'_>> = st.views();
    let resolver = Resolver {
        file: &d.path,
        bodies: views,
        features: &st.table,
        consumer: None,
        burials: &st.burials,
        publishes: &d.publishes,
        lock: None,
        catalog: None,
        policy: ResolutionPolicy::default(),
        names: Some(&st.names),
    };
    for (bi, b) in st.bodies.iter().enumerate() {
        for fi in 0..b.names.faces.len() {
            let h = EntityHandle {
                body: bi as u32,
                kind: TopoKind::Face,
                index: fi as u32,
            };
            let Some(cr) = resolver.canonical_reference(&h) else {
                return Err(format!("face {:?} has no canonical reference", h));
            };
            match resolver.resolve(&cr, Some(TopoKind::Face), Cardinality::AtLeastOne) {
                ResolveOutcome::Resolved(res) => {
                    let set = res.entities();
                    if !set.contains(&h) {
                        return Err(format!("`{}` does not select its face", cr));
                    }
                    if b.names.qualifiers.contains_key(&(fi as u32)) && set.len() != 1 {
                        return Err(format!("qualified `{}` selects {}", cr, set.len()));
                    }
                }
                other => return Err(format!("`{}` does not resolve: {:?}", cr, other)),
            }
        }
    }
    Ok(())
}

fn check_all(
    d: &PartDocument,
    r: &RegenResult,
    probes: &[Probe],
    expect: &dyn Fn(&Probe) -> Option<Expect>,
) -> Vec<String> {
    let mut errs = Vec::new();
    if let Err(e) = check_state(d, r) {
        errs.push(format!("state: {}", e));
    }
    for p in probes {
        if let Some(e) = expect(p) {
            if let Err(msg) = check_probe(r, p, &e) {
                errs.push(format!("{}: {}", p.name, msg));
            }
        }
    }
    errs
}

/// A scenario: a document before and after an edit, each checked.
struct Outcome {
    name: String,
    errors: Vec<String>,
}

fn notch(d: &mut PartDocument, at_start: bool) {
    let s = sketch(
        "s_notch",
        rect_names(
            PlaneRef::origin_xy(),
            -1.0,
            46.0,
            4.0,
            51.0,
            ["n1", "n2", "n3", "n4"],
        ),
    );
    let f = extrude("notch", "s_notch", mm(12.0), BodyMode::cut());
    if at_start {
        let i = d.feature_index(&fid("body")).unwrap() + 1;
        d.features.splice(i..i, [s, f]);
    } else {
        d.features.extend([s, f]);
    }
}

fn run_family(f: &Family) -> Vec<Outcome> {
    let env = Env::new();
    let mut out = Vec::new();
    let p0 = f.defaults();
    let probes0 = (f.probes)(&p0);
    let d0 = f.doc(&p0);
    let r0 = env.regen(&d0);
    let base_errs = check_all(&d0, &r0, &probes0, &|p| Some(p.expect.clone()));
    let mine = |p: &Probe| p.maker == f.feature;

    // Parameter edits.
    for e in &f.edits {
        let p1 = f.at(e);
        let probes1 = (f.probes)(&p1);
        let mut d1 = d0.clone();
        apply(&mut d1, &p1);
        let r1 = env.regen(&d1);
        let mut errs = base_errs.clone();
        errs.extend(check_all(&d1, &r1, &probes1, &|p| Some(p.expect.clone())));
        out.push(Outcome {
            name: format!("{}: edit {:?}", f.name, e),
            errors: errs,
        });

        // The same edit through a configuration.
        let mut dc = d0.clone();
        dc.configurations.push(Configuration {
            name: "alt".into(),
            overrides: e
                .iter()
                .map(|(k, v)| (InputName::new(*k), value_for(&d0, k, *v)))
                .collect(),
            suppress: Default::default(),
        });
        let rc = env.regen_config(&dc, "alt");
        out.push(Outcome {
            name: format!("{}: configuration {:?}", f.name, e),
            errors: check_all(&dc, &rc, &probes1, &|p| Some(p.expect.clone())),
        });
    }

    // Suppress, unsuppress, suppress in a configuration.
    let mut ds = d0.clone();
    ds.feature_mut(f.feature).unwrap().suppressed = true;
    let rs = env.regen(&ds);
    let waiting = |p: &Probe| {
        p.when_suppressed
            .clone()
            .or_else(|| mine(p).then_some(Expect::Waiting))
    };
    let any_mine = probes0.iter().any(mine);
    out.push(Outcome {
        name: format!("{}: suppress", f.name),
        errors: check_all(&ds, &rs, &probes0, &waiting),
    });
    let mut errs = check_all(&ds, &rs, &probes0, &waiting);
    errs.extend(base_errs.clone());
    out.push(Outcome {
        name: format!("{}: unsuppress", f.name),
        errors: errs,
    });
    let mut dcs = d0.clone();
    dcs.configurations.push(Configuration {
        name: "lite".into(),
        overrides: Default::default(),
        suppress: [fid(f.feature)].into_iter().collect(),
    });
    let rcs = env.regen_config(&dcs, "lite");
    let mut errs = check_all(&dcs, &rcs, &probes0, &waiting);
    if any_mine
        && !rcs
            .diagnostics
            .diagnostics
            .iter()
            .any(|x| x.code == DiagCode::RefWaiting && x.message.contains("configuration"))
    {
        errs.push("no waiting diagnostic naming the configuration".into());
    }
    out.push(Outcome {
        name: format!("{}: suppressed in configuration", f.name),
        errors: errs,
    });

    // Reorder an independent feature around this one.
    let mut da = d0.clone();
    notch(&mut da, false);
    let mut db = d0.clone();
    notch(&mut db, true);
    let (ra, rb) = (env.regen(&da), env.regen(&db));
    let mut errs = check_all(&da, &ra, &probes0, &|p| Some(p.expect.clone()));
    errs.extend(check_all(&db, &rb, &probes0, &|p| Some(p.expect.clone())));
    out.push(Outcome {
        name: format!("{}: reorder", f.name),
        errors: errs,
    });

    // Rename through the lock: references keep the old id.
    {
        let mut env = Env::new();
        let mut d = d0.clone();
        let mut lock = LockSection::default();
        let mut minter = SeededMinter {
            seed: "corpus".into(),
        };
        for ft in &d.features {
            lock.ensure(&ItemPath::feature(&d.path, &ft.id), &mut minter);
        }
        let new = fid(&format!("{}_renamed", f.feature));
        lock.rename(
            &ItemPath::feature(&d.path, &fid(f.feature)),
            &ItemPath::feature(&d.path, &new),
        )
        .unwrap();
        d.feature_mut(f.feature).unwrap().id = new;
        env.lock = Some(lock);
        let r = env.regen(&d);
        let mut errs = check_all(&d, &r, &probes0, &|p| Some(p.expect.clone()));
        if any_mine && r.stats.via_alias == 0 {
            errs.push("nothing resolved through the alias".into());
        }
        out.push(Outcome {
            name: format!("{}: rename", f.name),
            errors: errs,
        });
    }

    // A merge of concurrent edits: one side edits a parameter, the other adds a feature.
    if let Some(e) = f.edits.first() {
        let p1 = f.at(e);
        let probes1 = (f.probes)(&p1);
        let mut ours = d0.clone();
        apply(&mut ours, &p1);
        let mut theirs = d0.clone();
        notch(&mut theirs, false);
        let mut merged = ours.clone();
        notch(&mut merged, false);
        let (ro, rt, rm) = (env.regen(&ours), env.regen(&theirs), env.regen(&merged));
        let rep = recheck_part_merge(Some(&r0), &ro, &rt, &rm);
        let mut errs = check_all(&merged, &rm, &probes1, &|p| Some(p.expect.clone()));
        if !rep.findings.is_empty() {
            errs.push(format!("merge findings: {:?}", rep.findings));
        }
        if rep.checked == 0 {
            errs.push("merge checked nothing".into());
        }
        out.push(Outcome {
            name: format!("{}: merge", f.name),
            errors: errs,
        });
    }
    out
}

// ------------------------------------------------------------------------------- specials

fn special(name: &str, errors: Vec<String>) -> Outcome {
    Outcome {
        name: name.into(),
        errors,
    }
}

fn specials() -> Vec<Outcome> {
    let env = Env::new();
    let mut out = Vec::new();

    // Sweep: a third path segment adds a side piece; the first two keep their ordinals.
    {
        let f = sweep();
        let p = f.defaults();
        let mut d = block();
        for (n, k, _) in &f.inputs {
            d.inputs.push(InputDecl::new(n, k, mm(0.0)));
        }
        sweep_doc(&mut d, true);
        apply(&mut d, &p);
        let probes = sweep_probes(&p, 3);
        for pr in &probes {
            d.regions.push(region(
                pr.name,
                pr.kind,
                vec![pr.reference.clone()],
                pr.card,
            ));
        }
        let r = env.regen(&d);
        out.push(special(
            "sweep: path curve added",
            check_all(&d, &r, &probes, &|p| Some(p.expect.clone())),
        ));
    }

    // Loft: a third section makes two spans; the ordinal-free side is now two pieces.
    {
        let f = loft();
        let p = f.defaults();
        let mut d = block();
        for (n, k, _) in &f.inputs {
            d.inputs.push(InputDecl::new(n, k, mm(0.0)));
        }
        loft_doc(&mut d, true);
        apply(&mut d, &p);
        let probes = vec![
            probe(
                "lf_side_one",
                TopoKind::Face,
                loft_side(None),
                Cardinality::ExactlyOne,
                "lf",
                Expect::Problem("ambiguous"),
            ),
            probe(
                "lf_side_all",
                TopoKind::Face,
                loft_side(None),
                Cardinality::AtLeastOne,
                "lf",
                Expect::Ok(2, any()),
            ),
            face1(
                "lf_span0",
                loft_side(Some(0)),
                "lf",
                centroid_near([None, None, Some(20.0 + p["lz"] / 2.0)], 2.5),
            ),
            face1(
                "lf_span1",
                loft_side(Some(1)),
                "lf",
                centroid_near([None, None, Some(30.0 + p["lz"])], 2.5),
            ),
        ];
        for pr in &probes {
            d.regions.push(region(
                pr.name,
                pr.kind,
                vec![pr.reference.clone()],
                pr.card,
            ));
        }
        let r = env.regen(&d);
        out.push(special(
            "loft: section added",
            check_all(&d, &r, &probes, &|p| Some(p.expect.clone())),
        ));
    }

    // Arc boss: splitting the bottom line into two keeps the arcs; replacing an arc with lines
    // reports its side as not generated, with candidates.
    {
        let f = arc_boss();
        let p = f.defaults();
        let base = f.doc(&p);
        let edit = |d: &mut PartDocument, replace_arc: bool| {
            let Some(Feature {
                kind: FeatureKind::Sketch(s),
                ..
            }) = d.features.iter_mut().find(|x| x.id == fid("s_boss"))
            else {
                unreachable!()
            };
            let (x0, x1) = (sub(mm(40.0), input("len")), add(mm(40.0), input("len")));
            let (y0, y1) = (sub(mm(25.0), input("rad")), add(mm(25.0), input("rad")));
            if replace_arc {
                s.entities.retain(|e| e.id.0 != "ar");
                s.entities.push(line(
                    "ar1",
                    (x1.clone(), y0.clone()),
                    (add(x1.clone(), input("rad")), mm(25.0)),
                ));
                s.entities.push(line(
                    "ar2",
                    (add(x1.clone(), input("rad")), mm(25.0)),
                    (x1, y1),
                ));
            } else {
                s.entities.retain(|e| e.id.0 != "lb");
                s.entities
                    .push(line("lb", (x0, y0.clone()), (mm(40.0), y0.clone())));
                s.entities
                    .push(line("lb2", (mm(40.0), y0.clone()), (x1, y0)));
            }
        };
        let mut d1 = base.clone();
        edit(&mut d1, false);
        let r1 = env.regen(&d1);
        let probes = (f.probes)(&p);
        let mut errs = check_all(&d1, &r1, &probes, &|pr| {
            Some(if pr.name == "boss_flat" {
                // Two collinear lines make two coplanar sides, kept apart (the profile does
                // not ask them to merge): `lb` is the half from x0 to the split point. The
                // sampled corner arc adds staircase facets near x0.
                ok1(both(
                    plane_at([0.0, 1.0, 0.0], 1, 25.0 - p["rad"]),
                    centroid_near([Some(40.0 - p["len"] / 2.0), None, None], 1.5),
                ))
            } else {
                pr.expect.clone()
            })
        });
        out.push(special("arc_boss: profile curve added", errs.split_off(0)));
        let mut d2 = base.clone();
        edit(&mut d2, true);
        let r2 = env.regen(&d2);
        errs = check_all(&d2, &r2, &probes, &|pr| {
            Some(match pr.name {
                "boss_arc" => Expect::Problem("not_generated"),
                "boss_rim" => Expect::Problem("not_generated"),
                _ => pr.expect.clone(),
            })
        });
        let has_candidates = r2.diagnostics.diagnostics.iter().any(|x| {
            x.code == DiagCode::RefNotGenerated
                && x.repairs.iter().any(
                    |rp| matches!(rp, Repair::RePick { candidates, .. } if !candidates.is_empty()),
                )
        });
        if !has_candidates {
            errs.push("no same-feature candidates offered".into());
        }
        out.push(special("arc_boss: profile curve removed", errs));
    }

    // Hex cut becomes a pentagon: h6 is gone, h1 stays (rotated by the new vertex angles).
    {
        let f = hex_cut();
        let p = f.defaults();
        let mut d = f.doc(&p);
        let Some(Feature {
            kind: FeatureKind::Sketch(s),
            ..
        }) = d.features.iter_mut().find(|x| x.id == fid("s_hex"))
        else {
            unreachable!()
        };
        *s = polygon(
            PlaneRef::datum("top_plane"),
            &hex_points(5),
            Some(&["h1", "h2", "h3", "h4", "h5"]),
        );
        d.regions.push(region(
            "hex_side6",
            TopoKind::Face,
            vec![side("hex", "s_hex", "h6")],
            Cardinality::ExactlyOne,
        ));
        let r = env.regen(&d);
        let a = 36f64.to_radians();
        let mut probes = (f.probes)(&p);
        probes.push(probe(
            "hex_side6",
            TopoKind::Face,
            side("hex", "s_hex", "h6"),
            Cardinality::ExactlyOne,
            "hex",
            Expect::Problem("not_generated"),
        ));
        out.push(special(
            "hex_cut: profile curve removed",
            check_all(&d, &r, &probes, &|pr| {
                Some(match pr.name {
                    "hex_side1" => ok1(plane_along([a.cos(), a.sin(), 0.0])),
                    "hex_side4" => ok1(any()),
                    _ => pr.expect.clone(),
                })
            }),
        ));
    }

    // Linear pattern: skipping instance 2 is reported by key; instance 1 is untouched.
    {
        let f = linear_pattern();
        let p = f.defaults();
        let mut d = f.doc(&p);
        if let FeatureKind::Pattern(pt) = &mut d.feature_mut("row").unwrap().kind {
            pt.skip.push(InstanceKey::Linear { step: 2 });
        }
        let r = env.regen(&d);
        out.push(special(
            "linear_pattern: skip instance",
            check_all(&d, &r, &(f.probes)(&p), &|pr| {
                Some(if pr.name == "row_w2" {
                    Expect::Problem("instance_removed")
                } else {
                    pr.expect.clone()
                })
            }),
        ));
    }

    // Hole form changes: simple → counterbore keeps the wall and bottom and adds the floor;
    // blind → through reports the bottom as not generated.
    {
        let f = hole();
        let p = f.defaults();
        let mut d = f.doc(&p);
        if let FeatureKind::Hole(h) = &mut d.feature_mut("h").unwrap().kind {
            h.form = HoleForm::Counterbore {
                diameter: Some(mm(10.0)),
                depth: Some(mm(2.0)),
            };
        }
        let floor = face1(
            "h_cb_floor",
            hole_face("h", Role::HoleCounterbore),
            "h",
            plane_at([0.0, 0.0, 1.0], 2, 8.0),
        );
        d.regions.push(region(
            floor.name,
            floor.kind,
            vec![floor.reference.clone()],
            floor.card,
        ));
        let mut probes = (f.probes)(&p);
        probes.push(floor);
        let r = env.regen(&d);
        out.push(special(
            "hole: simple to counterbore",
            check_all(&d, &r, &probes, &|pr| {
                Some(if pr.name == "h_edge" {
                    // The wall now meets the counterbore floor, not the top.
                    Expect::Problem("not_adjacent")
                } else {
                    pr.expect.clone()
                })
            }),
        ));
        let mut d2 = f.doc(&p);
        if let FeatureKind::Hole(h) = &mut d2.feature_mut("h").unwrap().kind {
            h.end = HoleEnd::ThroughAll;
        }
        let r2 = env.regen(&d2);
        out.push(special(
            "hole: blind to through",
            check_all(&d2, &r2, &(f.probes)(&p), &|pr| {
                Some(if pr.name == "h_bottom" {
                    Expect::Problem("not_generated")
                } else {
                    pr.expect.clone()
                })
            }),
        ));
    }

    // Shell: opening the front too removes its inner wall and consumes the front.
    {
        let f = shell();
        let p = f.defaults();
        let mut d = f.doc(&p);
        if let FeatureKind::Shell(s) = &mut d.feature_mut("sh").unwrap().kind {
            s.open_faces.push(side("body", "s1", "l1"));
        }
        let r = env.regen(&d);
        out.push(special(
            "shell: face opened",
            check_all(&d, &r, &(f.probes)(&p), &|pr| {
                Some(match pr.name {
                    "sh_front_in" => Expect::Problem("not_generated"),
                    "sh_front" => Expect::Problem("consumed"),
                    _ => pr.expect.clone(),
                })
            }),
        ));
    }

    // Draft: drafting the right face too tilts it; the others are unchanged.
    {
        let f = draft();
        let p = f.defaults();
        let mut d = f.doc(&p);
        if let FeatureKind::Draft(dr) = &mut d.feature_mut("dr").unwrap().kind {
            dr.faces.push(side("body", "s1", "l2"));
        }
        let a = p["da"].to_radians();
        let r = env.regen(&d);
        out.push(special(
            "draft: face added",
            check_all(&d, &r, &(f.probes)(&p), &|pr| {
                Some(if pr.name == "dr_right" {
                    ok1(plane_along([a.cos(), 0.0, a.sin()]))
                } else {
                    pr.expect.clone()
                })
            }),
        ));
    }

    // Split exactly in half: the slot ties, is reported with candidates that each select one
    // piece, and the part output (which names the original body) reports the ambiguity.
    {
        let f = split_family();
        let p = f.at(&pv(&[("sx", 40.0)]));
        let d = f.doc(&p);
        let r = env.regen(&d);
        let mut errs = check_all(&d, &r, &(f.probes)(&p), &|pr| Some(pr.expect.clone()));
        match r
            .diagnostics
            .diagnostics
            .iter()
            .find(|x| x.code == DiagCode::RefBodySlotTie)
        {
            None => errs.push("no slot-tie warning".into()),
            Some(w) => {
                let cands: Vec<Reference> = w
                    .repairs
                    .iter()
                    .flat_map(|rp| match rp {
                        Repair::Disambiguate { candidates, .. } => candidates.clone(),
                        _ => Vec::new(),
                    })
                    .collect();
                if cands.len() != 2 {
                    errs.push(format!("{} candidates", cands.len()));
                }
                let st = &r.state;
                let resolver = Resolver {
                    file: &d.path,
                    bodies: st.views(),
                    features: &st.table,
                    consumer: None,
                    burials: &st.burials,
                    publishes: &d.publishes,
                    lock: None,
                    catalog: None,
                    policy: ResolutionPolicy::default(),
                    names: Some(&st.names),
                };
                let mut seen = std::collections::BTreeSet::new();
                for c in &cands {
                    match resolver.resolve(c, Some(TopoKind::Body), Cardinality::ExactlyOne) {
                        ResolveOutcome::Resolved(res) => {
                            seen.extend(res.entities());
                        }
                        other => errs.push(format!("`{}`: {:?}", c, other)),
                    }
                }
                if seen.len() != 2 {
                    errs.push("candidates do not select both pieces".into());
                }
            }
        }
        if r.parts.values().any(|o| o.ok) {
            errs.push("the part output resolved a tied slot".into());
        }
        out.push(special("split: even split ties", errs));
    }

    // Fillet: renaming the block's extrude reaches the blend face's nested names through the
    // name table, whatever the depth.
    {
        let f = fillet_family();
        let p = f.defaults();
        let mut d = f.doc(&p);
        let mut env = Env::new();
        let mut lock = LockSection::default();
        let mut minter = SeededMinter {
            seed: "corpus".into(),
        };
        for ft in &d.features {
            lock.ensure(&ItemPath::feature(&d.path, &ft.id), &mut minter);
        }
        lock.rename(
            &ItemPath::feature(&d.path, &fid("body")),
            &ItemPath::feature(&d.path, &fid("plate")),
        )
        .unwrap();
        d.feature_mut("body").unwrap().id = fid("plate");
        env.lock = Some(lock);
        let r = env.regen(&d);
        let mut errs = check_all(&d, &r, &(f.probes)(&p), &|pr| Some(pr.expect.clone()));
        let blend = &r.regions[&runmat_cad::ids::RegionName::new("rnd_face")];
        match &blend.references[0].outcome {
            ResolveOutcome::Resolved(res) if res.class == ResolutionClass::ViaAlias => {
                if !res
                    .notes
                    .iter()
                    .any(|n| matches!(n, ResolveNote::ViaAlias { .. }))
                {
                    errs.push("no alias note".into());
                }
            }
            other => errs.push(format!("blend not via alias: {:?}", other)),
        }
        out.push(special("fillet: nested names follow a rename", errs));
    }

    // Groove: the two top pieces carry qualifiers, and each qualified reference selects one.
    {
        let f = groove();
        let p = f.defaults();
        let d = f.doc(&p);
        let r = env.regen(&d);
        let q: usize = r
            .state
            .bodies
            .iter()
            .map(|b| b.names.qualifiers.len())
            .sum();
        let mut errs = Vec::new();
        if q < 2 {
            errs.push(format!("{} qualified pieces", q));
        }
        out.push(special("groove: split top pieces are qualified", errs));
    }

    // Cross-file: another file reads each kind's face through a published name, and keeps
    // reading it after the publisher's file is renamed.
    for (fam, publish) in [
        (revolve(), side("spin", "s_rev", "l3")),
        (hole(), hole_face("h", Role::HoleWall)),
        (shell(), shell_inner(&side("body", "s1", "l1"))),
        (sweep(), sweep_side(Some(0))),
        (loft(), Reference::end_cap("lf")),
        (draft(), side("body", "s1", "l1")),
        (
            circular_pattern(),
            copy_face(
                &Reference::end_cap("pin"),
                "ring",
                InstanceKey::Circular { step: 1 },
            ),
        ),
        (counterbore(), hole_face("cb", Role::HoleCounterbore)),
    ] {
        out.push(cross_file_rename(&fam, publish));
    }
    out
}

fn cross_file_rename(f: &Family, published: Reference) -> Outcome {
    let p = f.defaults();
    let mut d = f.doc(&p);
    d.path = FilePath::new("parts/source.part");
    d.publishes.push(PublishEntry {
        name: PublishName::new("mount"),
        target: PublishTarget::Entities {
            kind: TopoKind::Face,
            references: vec![published],
        },
        auto: false,
    });
    let env = Env::new();
    let r = env.regen(&d);
    let mut errs = Vec::new();
    if let Err(e) = check_state(&d, &r) {
        errs.push(e);
    }
    let mut catalog = MemoryCatalog::default();
    let new_path = FilePath::new("parts/renamed.part");
    catalog.set_live(&new_path, r.publishes.clone());
    let mut lock = LockSection::default();
    let mut minter = SeededMinter {
        seed: "corpus".into(),
    };
    lock.rename_file(&d.path, &new_path, &mut minter).unwrap();
    let mut consumer = block();
    consumer.path = FilePath::new("parts/user.part");
    consumer.regions.push(region(
        "mount",
        TopoKind::Face,
        vec![Reference::published(
            Some("parts/source.part"),
            "mount",
            XrefContext::Live,
        )],
        Cardinality::ExactlyOne,
    ));
    let mut cenv = Env::new();
    cenv.catalog = Some(catalog);
    cenv.lock = Some(lock);
    let rc = cenv.regen(&consumer);
    let o = &rc.regions[&runmat_cad::ids::RegionName::new("mount")];
    match &o.references[0].outcome {
        ResolveOutcome::Resolved(res) => {
            if !res
                .notes
                .iter()
                .any(|n| matches!(n, ResolveNote::XrefRenamed { .. }))
            {
                errs.push("no rename note".into());
            }
            if res.len() != 1 {
                errs.push(format!("{} entities", res.len()));
            }
        }
        other => errs.push(format!("{:?}", other)),
    }
    Outcome {
        name: format!("{}: cross-file rename", f.name),
        errors: errs,
    }
}

// ------------------------------------------------------------------------------- the gate

fn report(outcomes: &[Outcome]) {
    let failed: Vec<String> = outcomes
        .iter()
        .filter(|o| !o.errors.is_empty())
        .map(|o| format!("{}\n    {}", o.name, o.errors.join("\n    ")))
        .collect();
    assert!(
        failed.is_empty(),
        "{} of {} scenarios failed:\n{}",
        failed.len(),
        outcomes.len(),
        failed.join("\n")
    );
}

#[test]
fn kinds_corpus_every_family_under_every_edit() {
    let mut all = Vec::new();
    for f in families() {
        all.extend(run_family(&f));
    }
    all.extend(specials());
    report(&all);
    assert!(all.len() >= 150, "the corpus has {} scenarios", all.len());
}
