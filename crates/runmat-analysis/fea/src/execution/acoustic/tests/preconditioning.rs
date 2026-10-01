use super::*;

#[test]
fn pressure_preconditioner_is_explicit_identity_bound_and_unsupported_policies_reject() {
    let mesh = slab_mesh(2);
    let ilu = workload();
    let mut none = ilu.clone();
    none.solver.preconditioner = SolverPreconditioner::None;
    assert_ne!(ilu.identity().unwrap(), none.identity().unwrap());
    let result = ilu.execute(&mesh).unwrap();
    let imported =
        AcousticHarmonicSolverResult::from_canonical_bytes(&result.canonical_bytes().unwrap())
            .unwrap();
    assert!(imported.validate_for(&none, &mesh).is_err());
    assert!(imported.validate_for(&ilu, &mesh).is_ok());
    let none_result = none.execute(&mesh).unwrap();
    assert!(none_result.relative_residual() <= none.solver.relative_tolerance);
    for bad in [
        SolverPreconditioner::Jacobi,
        SolverPreconditioner::BlockJacobi,
    ] {
        let mut invalid = ilu.clone();
        invalid.solver.preconditioner = bad;
        assert!(invalid.validate().is_err());
    }
    let mut value = serde_json::to_value(ilu.solver).unwrap();
    value.as_object_mut().unwrap().remove("preconditioner");
    assert!(serde_json::from_value::<AcousticHarmonicPolicy>(value).is_err());
}
