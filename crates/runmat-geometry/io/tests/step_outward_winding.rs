#![cfg(feature = "occt-native")]
//! A STEP solid's preview mesh is wound outward on every face: each edge two faces share is
//! walked once in each direction, and the enclosed volume is positive. Sections cap a cut by
//! walking that winding, and shading and inside/outside tests read the same normals.

use std::collections::HashMap;

use runmat_geometry_io::{import_geometry, GeometryImportOptions};

fn example(name: &str) -> runmat_geometry_core::SurfaceMesh {
    let path = format!(
        "{}/../../../docs/simulation/examples/geometry/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap();
    let imported = import_geometry(&path, &bytes, GeometryImportOptions::default()).unwrap();
    assert_eq!(imported.asset.surface_meshes.len(), 1);
    imported.asset.surface_meshes.into_iter().next().unwrap()
}

/// Edges walked twice in the same direction, and the mesh's signed volume.
fn winding(mesh: &runmat_geometry_core::SurfaceMesh) -> (usize, usize, f64) {
    let key = |i: u32| mesh.vertices[i as usize].map(|c| (c * 1.0e9).round() as i64);
    let mut uses: HashMap<([i64; 3], [i64; 3]), i32> = HashMap::new();
    let mut volume = 0.0;
    for t in &mesh.triangles {
        for k in 0..3 {
            let (a, b) = (key(t[k]), key(t[(k + 1) % 3]));
            if a == b {
                continue;
            }
            let (lo, hi, sign) = if a < b { (a, b, 1) } else { (b, a, -1) };
            *uses.entry((lo, hi)).or_default() += sign;
        }
        let [a, b, c] = t.map(|i| mesh.vertices[i as usize]);
        volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))
            / 6.0;
    }
    let unbalanced = uses.values().filter(|n| **n != 0).count();
    (unbalanced, uses.len(), volume)
}

#[test]
fn an_imported_step_solid_is_wound_outward_on_every_face() {
    let mesh = example("bracket.step");
    let (unbalanced, edges, volume) = winding(&mesh);
    println!("bracket.step: {unbalanced} of {edges} edges unbalanced, volume {volume}");
    assert_eq!(unbalanced, 0, "every shared edge is walked once each way");
    assert!(volume > 0.0, "the faces face outward (volume {volume})");
}
