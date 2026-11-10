//! 3MF export through `design.export`: the package reads back through RunMat's reader,
//! conforms to the core specification's rules and the production extension, validates against
//! the 3MF core and production XSDs, and keeps sizes, materials and assembly placements.

use base64::Engine;
use runmat_design_tools::formats::three_mf::{self, Content};
use runmat_design_tools::{call, Files};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

const PLATE: &str = r#"runmat 0.7
std 1

part plate {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    output plate = { body = feature.body.body, part_number = "PL-80", material = "material.aluminum_6061" }
}
"#;

const PAIR: &str = r#"runmat 0.7
std 1

assembly pair {
    use part plate as lower
    use part plate as upper {
        placement = { placement = offset, offset = { translation = [0 mm, 0 mm, 10 mm] } }
    }
}
"#;

fn files() -> Files {
    [("/p/plate.part", PLATE), ("/p/pair.assembly", PAIR)]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

fn export(path: &str) -> Vec<u8> {
    let out: Value = call(
        "design.export",
        json!({ "path": path, "format": "3mf" }),
        &files(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let f = &out["files"][0];
    assert_eq!(f["mimeType"], "model/3mf");
    base64::engine::general_purpose::STANDARD
        .decode(f["content"].as_str().unwrap())
        .unwrap()
}

fn extents(tris: &[[[f64; 3]; 3]]) -> [f64; 3] {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in tris.iter().flatten() {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
}

/// Validate against the vendored XSDs with libxml2 through Python's cross-platform lxml wheel.
fn xsd_valid(package: &[u8], name: &str) {
    let entries = runmat_design_tools::formats::opc::unzip(package).unwrap();
    let model = &entries
        .iter()
        .find(|(n, _)| n == "3D/3dmodel.model")
        .unwrap()
        .1;
    let dir = std::env::temp_dir().join(format!("runmat-3mf-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("3dmodel.model");
    std::fs::write(&file, model).unwrap();
    let schema = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/3mf/schema.xsd");
    let out = Command::new(if cfg!(windows) { "python" } else { "python3" })
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/3mf/validate_xsd.py"))
        .arg(&schema)
        .arg(&file)
        .output()
        .expect(
            "3MF schema validation requires Python with lxml (python -m pip install 'lxml>=6,<7')",
        );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn a_part_exports_as_a_conforming_3mf_with_its_material_and_part_number() {
    let z = export("/p/plate.part");
    xsd_valid(&z, "part");
    let m = three_mf::read(&z).expect("reads back");
    assert_eq!(m.unit, "millimeter");
    assert_eq!(three_mf::check(&m), Vec::<String>::new());
    assert_eq!(m.items.len(), 1);
    let o = m.object(m.items[0].object).unwrap();
    assert_eq!(o.part_number.as_deref(), Some("PL-80"));
    let (pid, pindex) = o.material.unwrap();
    let group = &m.materials.iter().find(|(id, _)| *id == pid).unwrap().1;
    assert_eq!(group[pindex as usize].name, "aluminum_6061");
    assert_eq!(group[pindex as usize].color, "#C7CCD4FF");
    let e = extents(&m.world_triangles());
    for (got, want) in e.iter().zip([80.0, 50.0, 10.0]) {
        assert!((got - want).abs() < 1e-6, "extents {e:?}");
    }
    // The same model writes the same bytes (UUIDs derive from the path).
    assert_eq!(z, export("/p/plate.part"));
}

#[test]
fn an_assembly_exports_one_shared_object_and_a_build_item_per_placement() {
    let z = export("/p/pair.assembly");
    xsd_valid(&z, "assembly");
    let m = three_mf::read(&z).expect("reads back");
    assert_eq!(three_mf::check(&m), Vec::<String>::new());
    assert_eq!(m.items.len(), 2);
    assert_eq!(
        m.items[0].object, m.items[1].object,
        "both plates use one object"
    );
    let meshes = m
        .objects
        .iter()
        .filter(|o| matches!(o.content, Content::Mesh { .. }))
        .count();
    assert_eq!(meshes, 1);
    let dz: Vec<f64> = m
        .items
        .iter()
        .map(|i| i.transform.unwrap_or(three_mf::IDENTITY)[11])
        .collect();
    assert!(
        dz.iter().any(|z| z.abs() < 1e-9) && dz.iter().any(|z| (z - 10.0).abs() < 1e-9),
        "{dz:?}"
    );
    let e = extents(&m.world_triangles());
    assert!(
        (e[2] - 20.0).abs() < 1e-6,
        "the pair stacks to 20 mm: {e:?}"
    );
}

#[test]
fn the_checker_finds_open_meshes_bad_references_and_missing_uuids() {
    let z = export("/p/plate.part");
    let mut m = three_mf::read(&z).unwrap();
    if let Content::Mesh { triangles, .. } = &mut m.objects[0].content {
        triangles.pop();
    }
    m.items[0].uuid = None;
    m.items.push(three_mf::Item {
        object: 99,
        transform: None,
        part_number: None,
        uuid: Some("x".into()),
    });
    let p = three_mf::check(&m).join("\n");
    assert!(p.contains("not closed"), "{p}");
    assert!(p.contains("no p:UUID"), "{p}");
    assert!(p.contains("object 99"), "{p}");
    assert!(p.contains("not a UUID"), "{p}");
}
