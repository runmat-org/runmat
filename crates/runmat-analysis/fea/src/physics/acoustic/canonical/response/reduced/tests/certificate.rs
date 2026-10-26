//! The factorization that solves also certifies: a lossless discrete pole of the full original
//! canonical operator fails closed, even though its physical right-hand side is zero.
use super::*;
use crate::physics::acoustic::canonical::tests as acoustic_tests;

// witness: FEA-ACOUSTIC-NUMERICS-001/uniqueness @ acoustic_harmonic/tet4/numerical_kernel;
//   boundary: Independent Tet4 antisymmetric discrete pole fails the factorization certificate
//   despite a zero right-hand side; the same operator off the pole is certified.
#[test]
fn original_canonical_lossless_discrete_pole_fails_closed_despite_zero_rhs() {
    let model = acoustic_tests::model();
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let properties = crate::properties::compile_solid_property_map(&model, &mesh).unwrap();
    // Unit Tet4: gradients and consistent simplex mass give the antisymmetric
    // mode [0,1,-1,0] with omega²=20*c²=80, independently of density.
    let pole = AcousticAssembly::assemble(
        AnalysisProjection::derive(&mesh).unwrap(),
        &properties,
        runmat_types::physical::Quantity::new(80.0_f64.sqrt()),
    )
    .unwrap();
    let error = super::super::PreparedReduced::new(&pole, AcousticHarmonicPolicy::default())
        .err()
        .expect("a discrete pole is never certified");
    assert!(matches!(
        error,
        super::super::AcousticHarmonicError::Certificate(_)
    ));
    // Off the pole, the same original operator is certified and solved.
    let off = AcousticAssembly::assemble(
        AnalysisProjection::derive(&mesh).unwrap(),
        &properties,
        runmat_types::physical::Quantity::new(0.9 * 80.0_f64.sqrt()),
    )
    .unwrap();
    let prepared =
        super::super::PreparedReduced::new(&off, AcousticHarmonicPolicy::default()).unwrap();
    assert_eq!(prepared.certificate.dimension, 8);
    assert!(prepared.certificate.contraction_bound < prepared.certificate.limit);
    assert_eq!(
        prepared.certificate.operator_identity,
        prepared
            .operator
            .with_canonical_storage(|operator| operator.numerical_identity())
            .unwrap()
    );
}
