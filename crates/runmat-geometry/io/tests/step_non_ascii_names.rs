#![cfg(feature = "occt-native")]
//! A product name outside ASCII imports as UTF-8. A RunMat assembly names a part placed in a
//! configuration of its inputs with their values (`tilted (depth 12 mm, tilt -20°)`). OCCT wrote
//! the degree sign as raw UTF-8, and the import turned it into the single byte B0, which failed
//! the whole import with "data for rust::String is not utf-8".
//! Fixtures: `fixtures/configured-degree-name.step`, from runmat-geometry-wasm-host's
//! `a_configured_part_with_an_angle_input_exports_and_imports_back`, and the same file with the
//! name written as Part 21's `\X2\00B0\X0\`.

use runmat_geometry_io::{import_geometry, GeometryImportOptions};

const RAW: &[u8] = include_bytes!("fixtures/configured-degree-name.step");
const ESCAPED: &[u8] = include_bytes!("fixtures/configured-degree-name-escaped.step");

#[test]
fn a_product_name_with_a_degree_sign_imports_as_utf8() {
    for (file, bytes) in [("raw.step", RAW), ("escaped.step", ESCAPED)] {
        let imported = import_geometry(file, bytes, GeometryImportOptions::default())
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        let bodies: Vec<&str> = imported
            .asset
            .regions
            .iter()
            .filter(|r| r.tag.as_deref() == Some("cad_body"))
            .map(|r| r.name.as_str())
            .collect();
        assert!(
            bodies.contains(&"tilted (depth 12 mm, tilt -20°)"),
            "{file}: bodies {bodies:?}"
        );
    }
}
