//! Full original canonical operators, not RHS-dependent Krylov diagnostics.
use super::*;
use crate::algebra::{RightInverseCheck, RightInverseVerdict, RightInverseWorkEnvelope};
use crate::physics::acoustic::canonical::tests as acoustic_tests;
use crate::solve::krylov::right_ilu::SparseRightIlu;
use crate::topology::test_meshes::{quadratic_slab_mesh, slab_mesh};
use runmat_analysis_core::{
    AnalysisBoundary, BoundaryOperator, RegionTarget, SpecificAcousticImpedance,
};

fn check_operator(prepared: &super::super::PreparedReduced) -> Option<RightInverseVerdict> {
    let a = &prepared.operator;
    let ilu = SparseRightIlu::factor(a, || false).ok()?;
    let envelope = RightInverseWorkEnvelope::derive(a.rows(), a.values().len()).unwrap();
    let mut check = RightInverseCheck::new(a, envelope).unwrap();
    let mut basis = vec![0.0; a.rows()];
    for index in 0..basis.len() {
        basis[index] = 1.0;
        let column = ilu.apply(&basis, || false).unwrap();
        check.push_column(index, &column).unwrap();
        basis[index] = 0.0;
    }
    Some(check.finish().unwrap())
}

fn check_corrected_operator(
    prepared: &super::super::PreparedReduced,
) -> crate::solve::uniqueness::SparseUniquenessVerdict {
    use crate::solve::uniqueness::{
        certify_sparse_uniqueness, SparseUniquenessEnvelope, SparseUniquenessPolicy,
    };
    let a = &prepared.operator;
    let policy = SparseUniquenessPolicy {
        candidate_relative_tolerance: 1.0 / (16.0 * a.rows() as f64),
        restart: 64,
        maximum_iterations_per_column: 128,
    };
    let envelope = SparseUniquenessEnvelope::derive(a, policy).unwrap();
    eprintln!(
        "original uniqueness dimension={} admitted_work={} requested_bytes={}",
        a.rows(),
        envelope.scalar_operations,
        envelope.resident_bytes
    );
    certify_sparse_uniqueness(a, policy, envelope).unwrap()
}

#[test]
fn original_canonical_lossless_discrete_pole_is_not_proved_by_zero_rhs() {
    let model = acoustic_tests::model();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let properties = crate::properties::compile_solid_property_map(&model, &mesh).unwrap();
    // Unit Tet4: gradients and consistent simplex mass give the antisymmetric
    // mode [0,1,-1,0] with omega²=20*c²=80, independently of density.
    let assembly = AcousticAssembly::assemble(
        AnalysisProjection::derive(&mesh).unwrap(),
        &properties,
        80.0_f64.sqrt(),
    )
    .unwrap();
    let policy = AcousticHarmonicPolicy {
        preconditioner: crate::contracts::SolverPreconditioner::None,
        ..Default::default()
    };
    let prepared = super::super::PreparedReduced::new(&assembly, policy).unwrap();
    let mode = [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let action = crate::algebra::spmv(&prepared.operator, &mode).unwrap();
    assert!(action.iter().all(|v| v.abs() < 1e-14));
    assert!(prepared.rhs.iter().all(|v| *v == 0.0));
    let solution = prepared.solve(&assembly, policy, None).unwrap();
    assert_eq!(solution.iterations, 0);
    assert_eq!(solution.relative_residual, 0.0);
    // Floating representation can shift an exact mathematical pole. No rank
    // classification follows from either a tiny action or unsuccessful ILU.
    let verdict = check_operator(&prepared);
    eprintln!("canonical Tet4 independent discrete-pole uniqueness={verdict:?}");
    assert!(!matches!(verdict, Some(RightInverseVerdict::Verified(_))));
    assert!(matches!(
        check_corrected_operator(&prepared),
        crate::solve::uniqueness::SparseUniquenessVerdict::Indeterminate { .. }
    ));
    let frequency = assembly.omega / std::f64::consts::TAU;
    let response = super::super::super::solve(assembly, frequency, policy, 1e-8).unwrap();
    assert_eq!(response.relative_residual, 0.0);
    assert_eq!(response.power_relative_error, 0.0);
    assert!(response
        .pressure_pa
        .iter()
        .all(|p| p.real == 0.0 && p.imaginary == 0.0));
}

#[test]
fn original_matched_duct_ilu_certificate_discriminator() {
    let mut verified = 0;
    let mut indeterminate = 0;
    for quadratic in [false, true] {
        for cells in [1, 2, 4] {
            let assembly = matched_assembly(cells, quadratic);
            let prepared =
                super::super::PreparedReduced::new(&assembly, AcousticHarmonicPolicy::default())
                    .unwrap();
            let verdict = check_operator(&prepared);
            eprintln!("original matched duct Tet10={quadratic} cells={cells} dimension={} nnz={} uniqueness={verdict:?}", prepared.operator.rows(), prepared.operator.values().len());
            match verdict {
                Some(RightInverseVerdict::Verified(_)) => verified += 1,
                _ => indeterminate += 1,
            }
        }
    }
    assert_eq!((verified, indeterminate), (0, 6));
    // This is a discriminator, not six qualified continuum or execution cases.
    eprintln!(
        "original matched duct ILU candidate verified={verified} indeterminate={indeterminate}"
    );
}

#[test]
fn original_matched_duct_corrected_inverse_certificate() {
    for quadratic in [false, true] {
        for cells in [1, 2, 4] {
            let assembly = matched_assembly(cells, quadratic);
            let prepared =
                super::super::PreparedReduced::new(&assembly, AcousticHarmonicPolicy::default())
                    .unwrap();
            let verdict = check_corrected_operator(&prepared);
            let crate::solve::uniqueness::SparseUniquenessVerdict::Verified {
                certificate,
                iterations,
            } = verdict
            else {
                panic!("original matched duct Tet10={quadratic} cells={cells}: {verdict:?}");
            };
            eprintln!("original matched duct corrected Tet10={quadratic} cells={cells} dimension={} actual_iterations={iterations} outward_defect={}", prepared.operator.rows(), certificate.defect_infinity_upper());
        }
    }
}

fn matched_assembly(cells: usize, quadratic: bool) -> AcousticAssembly {
    let mesh = if quadratic {
        quadratic_slab_mesh(cells)
    } else {
        slab_mesh(cells)
    };
    let mut model = acoustic_tests::model();
    model.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::solid("slab");
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "matched".into(),
        target: RegionTarget::boundary("slab_right"),
        operator: BoundaryOperator::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(6.0),
        },
    });
    let properties = crate::properties::compile_solid_property_map(&model, &mesh).unwrap();
    let step = &model.sequence.steps[0];
    let sample = crate::excitation::ExcitationSamplePoint::frequency(0.2);
    let sources = crate::excitation::realize_step_excitations(&model, step, sample).unwrap();
    AcousticAssembly::assemble(
        AnalysisProjection::derive(&mesh).unwrap(),
        &properties,
        std::f64::consts::TAU * 0.2,
    )
    .unwrap()
    .with_boundaries(&model, step, &sources, sample)
    .unwrap()
}
