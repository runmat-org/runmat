//! Independent rectangular-cavity references: f = c/2 sqrt((l/Lx)² + (m/Ly)² + (n/Lz)²).
use super::*;
use crate::topology::test_meshes::{quadratic_slab_mesh, slab_mesh};
use runmat_analysis_core::{
    AcousticConstraint, PrescribedValue, Pressure, ProcedureStep, PropertyRegionTarget,
    RegionTarget,
};

fn cavity_model(count: usize) -> AnalysisModel {
    let mut model = super::super::tests::model();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.sequence.steps = vec![ProcedureStep::new(
        "acoustic",
        PhysicsFamily::Acoustic,
        AnalysisProcedure::ModalExtraction,
        SolutionSpace::Direct,
        IndependentDomain::Eigenvalue {
            requested_count: count,
        },
        Vec::new(),
    )];
    model
}

fn release(model: &mut AnalysisModel, face: &str) {
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: format!("release_{face}"),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                ),
                value: PrescribedValue::Constant(Pressure::new(0.0)),
            },
        }));
}

/// Unit cube, c = 2 m/s: f = sqrt(l² + m² + n²) Hz.
fn reference(released_x: bool, count: usize) -> Vec<f64> {
    let mut values = Vec::new();
    for l in 0..6_i32 {
        for m in 0..6_i32 {
            for n in 0..6_i32 {
                if released_x && l == 0 {
                    continue;
                }
                values.push(f64::from(l * l + m * m + n * n).sqrt());
            }
        }
    }
    values.sort_by(f64::total_cmp);
    values.truncate(count);
    values
}

// witness: FEA-ACOUSTIC-MODAL-001/rigid_box_modes @ acoustic_modal/tet10/numerical_kernel;
//   coverage: partial;
//   check: rigid-cavity-modes; boundary: Eight modes of a rigid quadratic-mesh cavity on 2, 3 and 4 cells match the
//   rectangular-cavity reference including degenerate triplets, with a zero-frequency uniform mode,
//   residuals at most 1e-8, mass orthogonality below 1e-8, h^4 frequency convergence and 4e-3 error
//   at four cells; the first ten modes within 0.5 percent on a cube are not asserted.
#[test]
fn rigid_cube_modes_match_rectangular_cavity_including_degenerate_triplets() {
    let count = 8;
    let mut previous_error = f64::INFINITY;
    for cells in [2, 3, 4] {
        let result = solve_canonical_modal(
            &cavity_model(count),
            &quadratic_slab_mesh(cells),
            "acoustic",
            AcousticModalPolicy::default(),
        )
        .unwrap();
        let expected = reference(false, count);
        assert_eq!(result.frequencies_hz.len(), count);
        // Uniform-pressure mode of the closed cavity.
        assert!(
            result.frequencies_hz[0].abs() < 1e-5,
            "{:?}",
            result.frequencies_hz
        );
        let error = result.frequencies_hz[1..]
            .iter()
            .zip(&expected[1..])
            .map(|(value, reference)| ((value - reference) / reference).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            previous_error.is_infinite() || previous_error / error > 2.5,
            "cells={cells}: {error} vs {previous_error}"
        );
        previous_error = error;
        assert!(result.relative_residuals.iter().all(|value| *value <= 1e-8));
        assert!(result.mass_orthogonality_error < 1e-8);
        if cells == 4 {
            // Quadratic elements: frequency error falls as h^4 (about 3.2x from 3 to 4 cells).
            assert!(error < 4e-3, "{:?} vs {expected:?}", result.frequencies_hz);
        }
    }
}

// witness: FEA-ACOUSTIC-MODAL-001/impedance_walls_and_damping_in_modes @
//   acoustic_modal/tet10/numerical_kernel; coverage: partial;
//   check: open-open-duct-modes; boundary: Pressure-release ends remove the rigid mode and five modes match the open-open duct
//   within 4e-3 with exactly zero modal pressure on released nodes; impedance walls and damping are
//   not covered.
#[test]
fn pressure_release_ends_remove_the_rigid_mode_and_match_open_open_duct() {
    let count = 5;
    let mut model = cavity_model(count);
    release(&mut model, "slab_left");
    release(&mut model, "slab_right");
    let result = solve_canonical_modal(
        &model,
        &quadratic_slab_mesh(4),
        "acoustic",
        AcousticModalPolicy::default(),
    )
    .unwrap();
    for (value, reference) in result.frequencies_hz.iter().zip(reference(true, count)) {
        assert!(
            ((value - reference) / reference).abs() < 4e-3,
            "{value} vs {reference}"
        );
    }
    // Released nodes carry exactly zero modal pressure.
    let projection = &result.projection;
    for mode in &result.mode_pressures {
        for (node, value) in projection.nodes.iter().zip(mode) {
            let x = node.coordinates_m.canonical_value()[0];
            if x.abs() < 1e-12 || (x - 1.0).abs() < 1e-12 {
                assert_eq!(*value, 0.0);
            }
        }
    }
}

// witness: FEA-ACOUSTIC-MODAL-001/rigid_box_modes @ acoustic_modal/tet4/numerical_kernel; coverage:
//   partial;
//   check: rigid-cavity-modes; boundary: On a linear mesh the first nonzero cavity mode converges with an error ratio above 3
//   from 2 to 4 cells (second order); only the first nonzero mode is compared.
#[test]
fn linear_cube_converges_at_second_order() {
    let errors = [2, 4]
        .into_iter()
        .map(|cells| {
            let result = solve_canonical_modal(
                &cavity_model(2),
                &slab_mesh(cells),
                "acoustic",
                AcousticModalPolicy::default(),
            )
            .unwrap();
            (result.frequencies_hz[1] - 1.0).abs()
        })
        .collect::<Vec<_>>();
    assert!(errors[0] / errors[1] > 3.0, "{errors:?}");
}

#[test]
fn lossy_boundaries_and_sources_are_rejected() {
    let mut model = cavity_model(2);
    model
        .boundaries
        .push(runmat_analysis_core::AnalysisBoundary {
            boundary_id: "absorber".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
            operator: BoundaryOperator::AcousticRadiation,
        });
    let error = solve_canonical_modal(
        &model,
        &slab_mesh(2),
        "acoustic",
        AcousticModalPolicy::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("complex modes"), "{error}");
}
