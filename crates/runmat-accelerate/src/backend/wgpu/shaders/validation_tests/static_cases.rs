//! Test-only inventory rows for unmodified production F32/F64 templates.

use super::{validate_generated, RequiredCaps};
use std::collections::BTreeSet;

pub(super) struct PrecisionPair {
    pub name: &'static str,
    pub single: &'static str,
    pub double: &'static str,
}

fn assert_inventory<'a>(labels: impl IntoIterator<Item = &'a str>, expected_rows: usize) {
    let mut seen = BTreeSet::new();
    for label in labels {
        assert!(seen.insert(label), "duplicate static shader case: {label}");
    }
    assert_eq!(
        seen.len(),
        expected_rows,
        "static shader case count changed"
    );
}

pub(super) fn validate_pairs(
    rows: &[PrecisionPair],
    expected_rows: usize,
    geometries: &[(u32, u32)],
) {
    assert_inventory(rows.iter().map(|row| row.name), expected_rows);
    for row in rows {
        for (source, caps, precision) in [
            (row.single, RequiredCaps::None, "single"),
            (row.double, RequiredCaps::Float64, "double"),
        ] {
            for &(workgroup, tile) in geometries {
                validate_generated(
                    &format!("{}-{precision}-wg{workgroup}-tile{tile}", row.name),
                    source,
                    workgroup,
                    tile,
                    caps,
                );
            }
        }
    }
}

pub(super) fn validate_singles(
    rows: &[(&str, &str)],
    expected_rows: usize,
    geometries: &[(u32, u32)],
) {
    assert_inventory(rows.iter().map(|(name, _)| *name), expected_rows);
    for &(name, source) in rows {
        for &(workgroup, tile) in geometries {
            validate_generated(
                &format!("{name}-wg{workgroup}-tile{tile}"),
                source,
                workgroup,
                tile,
                RequiredCaps::None,
            );
        }
    }
}
