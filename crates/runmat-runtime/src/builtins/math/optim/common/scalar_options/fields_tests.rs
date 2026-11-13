use super::*;
#[test]
fn solver_view_uses_canonical_scoped_aliases_without_changing_ordinary_first_bindings() {
    let mut builder = runmat_value::UnboundedStructValueBuilder::new();
    builder.insert("TolFun", Value::Num(1.));
    builder.insert("FunctionTolerance", Value::Num(2.));
    builder.insert("Generations", Value::Num(3.));
    builder.insert("StallTimeLimit", Value::Num(4.));
    let options = builder.finish();
    let ordinary = ScalarOptionFields::from_struct(Some(&options));
    assert!(matches!(ordinary.exact(OptimOptionField::TolFun),Some(Value::Num(v))if *v==1.));
    assert!(
        matches!(ordinary.exact(OptimOptionField::FunctionTolerance),Some(Value::Num(v))if *v==2.)
    );
    let ga = ScalarOptionFields::from_struct_for_solver(
        Some(&options),
        runmat_builtins::OptimSolver::Ga,
    );
    assert!(matches!(ga.exact(OptimOptionField::FunctionTolerance),Some(Value::Num(v))if *v==1.));
    assert!(ga.exact(OptimOptionField::TolFun).is_none());
    assert_eq!(ga.count("ga", OptimOptionField::MaxIter, 99).unwrap(), 3);
    assert_eq!(
        ga.real("ga", OptimOptionField::MaxStallTime, 99.).unwrap(),
        4.
    );
}
