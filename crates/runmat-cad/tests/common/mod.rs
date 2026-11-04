//! Shared builders for the integration tests.
#![allow(dead_code)]

use runmat_cad::diagnostics::{DiagCode, Diagnostic};
use runmat_cad::ids::{FeatureId, RegionName, SketchEntityId};
use runmat_cad::model::*;
use runmat_cad::refs::*;
use runmat_cad::regen::*;
use runmat_cad::sketch_solver::ExplicitSketchSolver;
use runmat_cad::units::StubUnits;
use runmat_cad::value::{InputDecl, NoDynamicValues, ValueExpr};

pub fn mm(v: f64) -> ValueExpr {
    ValueExpr::mm(v)
}

pub fn input(name: &str) -> ValueExpr {
    ValueExpr::input(name)
}

/// A closed polygon sketch with lines named `l1..ln` (or the given names).
pub fn polygon(
    plane: PlaneRef,
    points: &[(ValueExpr, ValueExpr)],
    names: Option<&[&str]>,
) -> Sketch {
    let n = points.len();
    let name = |i: usize| -> String {
        match names {
            Some(ns) => ns[i].to_owned(),
            None => format!("l{}", i + 1),
        }
    };
    let entities = (0..n)
        .map(|i| {
            let (x0, y0) = points[i].clone();
            let (x1, y1) = points[(i + 1) % n].clone();
            SketchEntity {
                id: SketchEntityId::new(name(i)),
                geometry: SketchGeometry::Line {
                    start: PointExpr::Coords { x: x0, y: y0 },
                    end: PointExpr::Coords { x: x1, y: y1 },
                },
                construction: false,
            }
        })
        .collect();
    Sketch {
        plane,
        entities,
        constraints: Vec::new(),
        solved: SolvedCache::default(),
    }
}

/// Rectangle `x0..x1 × y0..y1`: l1 bottom (y0), l2 right (x1), l3 top (y1), l4 left (x0).
pub fn rect(plane: PlaneRef, x0: ValueExpr, y0: ValueExpr, x1: ValueExpr, y1: ValueExpr) -> Sketch {
    polygon(
        plane,
        &[
            (x0.clone(), y0.clone()),
            (x1.clone(), y0),
            (x1, y1.clone()),
            (x0, y1),
        ],
        None,
    )
}

pub fn rect_names(plane: PlaneRef, x0: f64, y0: f64, x1: f64, y1: f64, names: [&str; 4]) -> Sketch {
    polygon(
        plane,
        &[
            (mm(x0), mm(y0)),
            (mm(x1), mm(y0)),
            (mm(x1), mm(y1)),
            (mm(x0), mm(y1)),
        ],
        Some(&names),
    )
}

pub fn sketch(id: &str, s: Sketch) -> Feature {
    Feature::new(id, FeatureKind::Sketch(s))
}

pub fn extrude(id: &str, sketch: &str, distance: ValueExpr, mode: BodyMode) -> Feature {
    Feature::new(
        id,
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch(sketch),
            extent: Extent::Blind { distance },
            reverse: false,
            mode,
            from: Default::default(),
            second: None,
            draft: None,
            thin: None,
        }),
    )
}

pub fn extrude_reverse(id: &str, sketch: &str, distance: ValueExpr, mode: BodyMode) -> Feature {
    Feature::new(
        id,
        FeatureKind::Extrude(Extrude {
            profile: ProfileRef::sketch(sketch),
            extent: Extent::Blind { distance },
            reverse: true,
            mode,
            from: Default::default(),
            second: None,
            draft: None,
            thin: None,
        }),
    )
}

pub fn fillet(id: &str, edges: Vec<Reference>, r: f64) -> Feature {
    Feature::new(
        id,
        FeatureKind::Fillet(Fillet {
            edges,
            radius: mm(r),
            variable: Vec::new(),
            profile: Default::default(),
            setbacks: Vec::new(),
            propagate: true,
        }),
    )
}

pub fn region(name: &str, kind: TopoKind, select: Vec<Reference>, expect: Cardinality) -> Region {
    Region {
        name: RegionName::new(name),
        kind,
        select,
        expect,
    }
}

pub fn side(feature: &str, sketch: &str, curve: &str) -> Reference {
    Reference::side(feature, sketch, curve)
}

/// The base block: `width × depth × thickness` (80 × 50 × 10 mm), sketch `s1`, extrude `body`.
pub fn block() -> PartDocument {
    let mut d = PartDocument::new("bracket.part");
    d.inputs.push(InputDecl::new("width", "length", mm(80.0)));
    d.inputs.push(InputDecl::new("depth", "length", mm(50.0)));
    d.inputs
        .push(InputDecl::new("thickness", "length", mm(10.0)));
    d.features.push(sketch(
        "s1",
        rect(
            PlaneRef::origin_xy(),
            mm(0.0),
            mm(0.0),
            input("width"),
            input("depth"),
        ),
    ));
    d.features
        .push(extrude("body", "s1", input("thickness"), BodyMode::New));
    d.parts.push(PartOutput {
        name: "bracket".into(),
        body: Reference::feature_body("body"),
        part_number: Some("BR-100".into()),
        material: Some("material.aluminum".into()),
        properties: Vec::new(),
    });
    d
}

pub fn set_input(d: &mut PartDocument, name: &str, v: ValueExpr) {
    d.inputs
        .iter_mut()
        .find(|i| i.name.0 == name)
        .expect("input")
        .default = v;
}

/// The kernel the integration tests run on: the grid test kernel, or (when these files are
/// built into `runmat-geometry-io`'s `cad_kernel_corpus` test) the OCCT kernel under test.
#[cfg(not(runmat_cad_kernel_under_test))]
pub type TestKernel = FakeKernel;
#[cfg(runmat_cad_kernel_under_test)]
pub type TestKernel = crate::kernel_under_test::KernelUnderTest;

pub struct Env {
    pub kernel: TestKernel,
    pub solver: ExplicitSketchSolver,
    pub units: StubUnits,
    pub dynamic: NoDynamicValues,
    pub lock: Option<LockSection>,
    pub catalog: Option<MemoryCatalog>,
    pub policy: ResolutionPolicy,
}

impl Env {
    pub fn new() -> Self {
        Env {
            kernel: TestKernel::new(),
            solver: ExplicitSketchSolver,
            units: StubUnits,
            dynamic: NoDynamicValues,
            lock: None,
            catalog: None,
            policy: ResolutionPolicy::default(),
        }
    }

    pub fn ctx(&self) -> RegenContext<'_> {
        RegenContext {
            kernel: &self.kernel,
            solver: &self.solver,
            units: &self.units,
            dynamic: &self.dynamic,
            lock: self.lock.as_ref(),
            catalog: self.catalog.as_ref().map(|c| c as &dyn PublishedCatalog),
            policy: self.policy,
            slots: None,
        }
    }

    pub fn regen(&self, d: &PartDocument) -> RegenResult {
        let mut cache = RegenCache::default();
        regenerate(d, &self.ctx(), &RegenRequest::default(), &mut cache)
    }

    pub fn regen_config(&self, d: &PartDocument, config: &str) -> RegenResult {
        let mut cache = RegenCache::default();
        regenerate(
            d,
            &self.ctx(),
            &RegenRequest {
                configuration: Some(config.into()),
            },
            &mut cache,
        )
    }
}

pub fn has(r: &RegenResult, code: DiagCode) -> bool {
    r.diagnostics.diagnostics.iter().any(|d| d.code == code)
}

pub fn diag(r: &RegenResult, code: DiagCode) -> &Diagnostic {
    r.diagnostics
        .diagnostics
        .iter()
        .find(|d| d.code == code)
        .unwrap_or_else(|| panic!("no {:?} in {:#?}", code, r.diagnostics.diagnostics))
}

pub fn errors(r: &RegenResult) -> Vec<&Diagnostic> {
    r.diagnostics.errors().collect()
}

pub fn assert_clean(r: &RegenResult) {
    let e = errors(r);
    assert!(e.is_empty(), "unexpected errors: {:#?}", e);
    for f in &r.features {
        assert!(
            f.status.ran()
                || matches!(
                    f.status,
                    FeatureStatus::Suppressed
                        | FeatureStatus::RolledBack
                        | FeatureStatus::SuppressedInConfiguration
                ),
            "feature {} is {:?}",
            f.id,
            f.status
        );
    }
}

/// Face geometry for a region's entities (faces only), via the final state: the faces its
/// references resolved to (two pieces of a split face share their names, so the names alone do
/// not tell them apart), else the faces bearing its resolved names.
pub fn region_faces(r: &RegenResult, name: &str) -> Vec<FaceGeom> {
    let o = r
        .regions
        .get(&RegionName::new(name))
        .unwrap_or_else(|| panic!("no region {}", name));
    let mut handles = std::collections::BTreeSet::new();
    for rec in &o.references {
        if let ResolveOutcome::Resolved(res) = &rec.outcome {
            handles.extend(
                res.entities()
                    .into_iter()
                    .filter(|h| h.kind == TopoKind::Face),
            );
        }
    }
    if !handles.is_empty() {
        return handles
            .iter()
            .map(|h| r.state.bodies[h.body as usize].topology.faces[h.index as usize].geom)
            .collect();
    }
    let mut out = Vec::new();
    for names in &o.entities {
        for b in &r.state.bodies {
            for (fi, set) in b.names.faces.iter().enumerate() {
                if names.iter().all(|n| set.contains(n)) && set.len() == names.len() {
                    out.push(b.topology.faces[fi].geom);
                }
            }
        }
    }
    out
}

pub fn region_ok(r: &RegenResult, name: &str) -> bool {
    r.regions
        .get(&RegionName::new(name))
        .map(|o| o.ok)
        .unwrap_or(false)
}

pub fn region_count(r: &RegenResult, name: &str) -> usize {
    r.regions
        .get(&RegionName::new(name))
        .map(|o| o.entities.len())
        .unwrap_or(0)
}

pub fn normal(g: &FaceGeom) -> [f64; 3] {
    match g.surface {
        SurfaceKind::Plane { normal } => normal,
        _ => [0.0; 3],
    }
}

pub fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

pub fn fid(s: &str) -> FeatureId {
    FeatureId::new(s)
}

pub fn total_faces(r: &RegenResult) -> usize {
    r.state.bodies.iter().map(|b| b.topology.faces.len()).sum()
}
