#![cfg(feature = "occt-native")]
//! Import survey for a large exported assembly named by `RUNMAT_LARGE_STEP` (ignored by default).
//!
//! Run: `RUNMAT_LARGE_STEP=/path/pack.step cargo test -p runmat-geometry-io --features occt-native
//! --test hero_pack_import -- --ignored --nocapture`. It imports the file, then samples every
//! curve/pcurve/surface incidence densely and reports the largest gaps next to the written edge
//! tolerance, naming the faces involved.

use runmat_geometry_core::{
    ExactCurveEvaluator, ExactPcurveEvaluator, ExactSurfaceEvaluator, GeometryEvaluationControl,
    GeometryEvaluationError, LengthUnit,
};
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
    OcctExactEvaluator,
};

struct Unlimited;

impl GeometryEvaluationControl for Unlimited {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

#[test]
#[ignore = "needs RUNMAT_LARGE_STEP pointing at a large exported STEP assembly"]
fn large_step_assembly_imports_and_reports_incidence_gaps() {
    let path = std::env::var("RUNMAT_LARGE_STEP").expect("RUNMAT_LARGE_STEP");
    let bytes = std::fs::read(&path).unwrap();
    let started = std::time::Instant::now();
    let imported = import_exact_cad(
        &path,
        &bytes,
        GeometryFormat::Step,
        &ExactCadImportOptions {
            source_units: Some(LengthUnit::Millimeter),
            ..ExactCadImportOptions::default()
        },
        &GeometryImportContext::new(),
    );
    let elapsed = started.elapsed();
    let imported = match imported {
        Ok(imported) => imported,
        Err(error) => panic!("import failed after {elapsed:?}: {error}"),
    };
    let topology = &imported.topology;
    println!(
        "imported in {elapsed:?}: {} bodies, {} faces, {} edges, {} coedges, {} named entities",
        topology.bodies.len(),
        topology.faces.len(),
        topology.edges.len(),
        topology.coedges.len(),
        imported.entity_names.len()
    );
    let evaluator = OcctExactEvaluator::new(&imported).unwrap();
    let mut solid_of_face = std::collections::BTreeMap::new();
    for solid in &topology.solids {
        let name = imported
            .entity_names
            .get(&solid.id)
            .and_then(|names| names.first().cloned())
            .unwrap_or_default();
        for shell_id in std::iter::once(&solid.outer_shell_id).chain(&solid.void_shell_ids) {
            if let Some(shell) = topology.shells.iter().find(|shell| &shell.id == shell_id) {
                for face_use in &shell.face_uses {
                    solid_of_face.insert(face_use.entity_id.clone(), name.clone());
                }
            }
        }
    }
    let mut gaps = Vec::new();
    for coedge in &topology.coedges {
        let edge = topology
            .edges
            .iter()
            .find(|edge| edge.id == coedge.edge_id)
            .unwrap();
        let face = topology
            .faces
            .iter()
            .find(|face| face.id == coedge.face_id)
            .unwrap();
        let range =
            ExactCurveEvaluator::parameter_range(&evaluator, &edge.curve_evaluator_id).unwrap();
        let mut worst = 0.0_f64;
        for index in 0..=256 {
            let parameter = range.start + (range.end - range.start) * index as f64 / 256.0;
            let curve = ExactCurveEvaluator::point(
                &evaluator,
                &edge.curve_evaluator_id,
                parameter,
                &Unlimited,
            )
            .unwrap()
            .canonical_value();
            let uv = ExactPcurveEvaluator::point(
                &evaluator,
                &coedge.pcurve_evaluator_id,
                parameter,
                &Unlimited,
            )
            .unwrap()
            .map(|coordinate| coordinate.with_native_coordinate(|value, _axis| value));
            let mapped =
                ExactSurfaceEvaluator::point(&evaluator, &face.surface_evaluator_id, uv, &Unlimited)
                    .unwrap()
                    .canonical_value();
            let gap = (0..3)
                .map(|axis| (curve[axis] - mapped[axis]).powi(2))
                .sum::<f64>()
                .sqrt();
            worst = worst.max(gap);
        }
        gaps.push((worst, edge.tolerance_m.canonical_value(), face.id.clone()));
    }
    gaps.sort_by(|left, right| right.0.total_cmp(&left.0));
    let over_written = gaps
        .iter()
        .filter(|(gap, tolerance, _)| *gap > tolerance * 1.1)
        .count();
    let over_resolution = gaps.iter().filter(|(gap, _, _)| *gap > 1.0e-8).count();
    println!(
        "coedges sampled: {}; over 1.1 x written edge tolerance: {over_written}; over 10 nm: {over_resolution}",
        gaps.len()
    );
    let mut by_solid = std::collections::BTreeMap::<String, (usize, f64)>::new();
    for (gap, _, face) in gaps.iter().filter(|(gap, _, _)| *gap > 1.0e-8) {
        let entry = by_solid
            .entry(solid_of_face.get(face).cloned().unwrap_or_default())
            .or_default();
        entry.0 += 1;
        entry.1 = entry.1.max(*gap);
    }
    let mut by_solid: Vec<_> = by_solid.into_iter().collect();
    by_solid.sort_by(|left, right| right.1 .1.total_cmp(&left.1 .1));
    for (solid, (count, worst)) in by_solid.iter().take(25) {
        println!("solid {solid:?}: {count} edge uses over 10 nm, largest gap {worst:.3e} m");
    }
}
