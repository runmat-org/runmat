//! Trimmed admission and located/periodic evaluation must survive session reconstruction.
use super::*;
use runmat_geometry_core::ExactSurfaceEvaluator;
use std::sync::atomic::{AtomicU64, Ordering};

struct AllocationLimit(AtomicU64);

impl GeometryEvaluationControl for AllocationLimit {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, count: u64) -> Result<(), GeometryEvaluationError> {
        self.0
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(count)
            })
            .map(|_| ())
            .map_err(|_| {
                GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::AllocationBudgetExceeded,
                    "surface-session test allocation ceiling",
                )
            })
    }
}

#[test]
fn session_surface_inventory_is_admitted_before_native_reconstruction() {
    let imported = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &meter_brep_options(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    let representation_bytes = imported.representation.len() as u64;
    let face_count = imported.topology.faces.len() as u64;
    for ceiling in [representation_bytes, representation_bytes + 8 * face_count] {
        let control = AllocationLimit(AtomicU64::new(ceiling));
        let error = OcctExactEvaluator::from_closure_controlled(
            &imported.representation,
            imported.kernel_units.meters_per_unit(),
            &imported.topology,
            &imported.evaluators,
            &control,
        )
        .err()
        .expect("surface storage must be admitted");
        assert_eq!(
            error.kind,
            GeometryEvaluationErrorKind::AllocationBudgetExceeded
        );
    }
    let control = AllocationLimit(AtomicU64::new(representation_bytes + 48 * face_count));
    let evaluator = OcctExactEvaluator::from_closure_controlled(
        &imported.representation,
        imported.kernel_units.meters_per_unit(),
        &imported.topology,
        &imported.evaluators,
        &control,
    )
    .unwrap();
    assert_eq!(control.0.load(Ordering::SeqCst), 0);
    let surface = &imported.topology.faces[0].surface_evaluator_id;
    assert!(ExactSurfaceEvaluator::parameter_bounds(&evaluator, surface).is_ok());
}

#[test]
fn session_surface_bounds_preserve_trim_admission_and_original_geometry() {
    let fixtures: &[(&str, &[u8])] = &[
        ("box.brep", BOX),
        (
            "cylinder.brep",
            include_bytes!("../../../../tests/fixtures/meshing_cylinder_translated.brep"),
        ),
        (
            "sphere.brep",
            include_bytes!("../../../../tests/fixtures/meshing_sphere.brep"),
        ),
    ];
    for (name, bytes) in fixtures {
        let imported = import_exact_cad(
            name,
            bytes,
            GeometryFormat::Brep,
            &meter_brep_options(),
            &GeometryImportContext::new(),
        )
        .unwrap();
        let direct = OcctExactEvaluator::new(&imported).unwrap();
        let cold = OcctExactEvaluator::from_closure(
            &imported.representation,
            imported.kernel_units.meters_per_unit(),
            &imported.topology,
            &imported.evaluators,
        )
        .unwrap();
        for face in &imported.topology.faces {
            let id = &face.surface_evaluator_id;
            let bounds = ExactSurfaceEvaluator::parameter_bounds(&direct, id).unwrap();
            assert_eq!(
                bounds,
                ExactSurfaceEvaluator::parameter_bounds(&cold, id).unwrap()
            );
            let periods = ExactSurfaceEvaluator::periodicity(&direct, id).unwrap();
            let uv = bounds.map(|range| (range.start + range.end) * 0.5);
            let expected = ExactSurfaceEvaluator::derivatives(&direct, id, uv, &Unlimited).unwrap();
            assert_eq!(
                expected,
                ExactSurfaceEvaluator::derivatives(&cold, id, uv, &Unlimited).unwrap()
            );
            // Independent inverse projection uses the bounded face adaptor.
            let projected = ExactSurfaceEvaluator::closest_point(
                &cold,
                id,
                expected.point_m,
                runmat_types::physical::Length::new(1e-8),
                &Unlimited,
            )
            .unwrap();
            assert!(projected.distance_m.canonical_value() < 1e-8, "{name}");
            for axis in 0..2 {
                let mut outside = uv;
                if let Some(period) = periods[axis] {
                    outside[axis] += period;
                    let wrapped =
                        ExactSurfaceEvaluator::point(&cold, id, outside, &Unlimited).unwrap();
                    assert!(wrapped
                        .canonical_value()
                        .into_iter()
                        .zip(expected.point_m.canonical_value())
                        .all(|(a, b)| (a - b).abs() < 1e-8));
                } else {
                    outside[axis] =
                        bounds[axis].end + (bounds[axis].end - bounds[axis].start).abs().max(1.0);
                    assert!(
                        ExactSurfaceEvaluator::point(&cold, id, outside, &Unlimited).is_err(),
                        "{name}"
                    );
                }
            }
        }
    }
}
