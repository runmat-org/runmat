//! K1 sheet metal on exact geometry: form tools stamped into a flat plate.
//!
//! The plate is 80 × 40 mm and 2 mm thick; each form is stamped on its top face at
//! (40, 20) mm. A louver keeps the part's volume (its hood is the lance's material); a dimple
//! or bead adds its wall's footprint times its height. The flat pattern keeps the plate.

use crate::common::{fid, mm, Env};
use crate::families::{plate_doc, plate_top};
use crate::k1_ops::{close, volume};
use runmat_cad::model::datum::PointRef;
use runmat_cad::model::sheet_metal::*;
use runmat_cad::model::*;
use runmat_cad::regen::kernel::KernelBody;
use runmat_cad::regen::RegenResult;
use runmat_cad::value::ValueExpr;
use std::f64::consts::PI;

use crate::kernel_under_test::KernelUnderTest;

const T: f64 = 0.002;

fn with_flat(mut d: PartDocument) -> PartDocument {
    d.features.push(Feature::new(
        "flat",
        FeatureKind::SheetMetal(SheetMetalFeature::FlatPattern {
            fixed_face: plate_top(),
            grain: None,
        }),
    ));
    d
}

/// The part's volume and top (largest z), and the flat pattern's volume.
fn measure(out: &RegenResult) -> (f64, f64, f64) {
    let of = |flat: bool| {
        out.state
            .bodies
            .iter()
            .find(|b| b.names.body.iter().any(|n| n.maker() == &fid("flat")) == flat)
            .expect("a body")
    };
    let v = |b: &runmat_cad::regen::engine::LiveBody| {
        volume(
            &KernelUnderTest::new(),
            &KernelBody {
                handle: b.handle.clone(),
                topology: b.topology.clone(),
            },
        )
        .0
    };
    let part = of(false);
    let top = part
        .topology
        .faces
        .iter()
        .map(|f| f.geom.bbox_max[2])
        .fold(f64::MIN, f64::max);
    (v(part), top, v(of(true)))
}

fn stamped(tool: FormToolShape) -> (f64, f64, f64) {
    let env = Env::new();
    let plain = env.regen(&with_flat(plate_doc()));
    let mut d = plate_doc();
    d.features.push(Feature::new(
        "form",
        FeatureKind::SheetMetal(SheetMetalFeature::FormTool {
            tool,
            face: plate_top(),
            at: PointRef::Coords {
                x: mm(40.0),
                y: mm(20.0),
                z: mm(2.0),
            },
            rotation: ValueExpr::deg(0.0),
        }),
    ));
    let out = env.regen(&with_flat(d));
    assert!(out.status("form").unwrap().ran(), "{:?}", out.diagnostics);
    assert!(out.status("flat").unwrap().ran(), "{:?}", out.diagnostics);
    let (p0, _, f0) = measure(&plain);
    let (p1, top, f1) = measure(&out);
    assert!(
        close(f1, f0, 1e-12),
        "the flat pattern keeps the plate: {f1} vs {f0}"
    );
    let mut seen = std::collections::BTreeSet::new();
    for n in out
        .state
        .bodies
        .iter()
        .flat_map(|b| b.names.faces.iter().flatten())
    {
        assert!(seen.insert(n.clone()), "{n} on two faces");
    }
    (p1 - p0, top, p0)
}

#[test]
fn k1_a_louver_keeps_the_parts_volume_and_rises_to_its_height() {
    let (change, top, base) = stamped(FormToolShape::Louver {
        length: mm(20.0),
        width: mm(6.0),
        height: mm(3.0),
    });
    assert!(change.abs() <= 1e-9 * base, "{change}");
    assert!(close(top, T + 0.003, 1e-9), "{top}");
}

#[test]
fn k1_a_dimple_adds_its_walls_footprint_times_its_height() {
    let (r, h) = (0.005, 0.003);
    let (change, top, _) = stamped(FormToolShape::Dimple {
        diameter: mm(10.0),
        height: mm(3.0),
    });
    let want = PI * (r * r - (r - T) * (r - T)) * h;
    assert!(close(change, want, 1e-9), "{change} vs {want}");
    assert!(close(top, T + h, 1e-9), "{top}");
}

#[test]
fn k1_a_bead_adds_its_walls_footprint_times_its_height() {
    let (l, w, h) = (0.03, 0.008, 0.003);
    let (change, top, _) = stamped(FormToolShape::Bead {
        length: mm(30.0),
        width: mm(8.0),
        height: mm(3.0),
    });
    let want = (l * w - (l - 2.0 * T) * (w - 2.0 * T)) * h;
    assert!(close(change, want, 1e-9), "{change} vs {want}");
    assert!(close(top, T + h, 1e-9), "{top}");
}
