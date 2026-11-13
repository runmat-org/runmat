use super::*;
#[test]
fn augmented_controls_are_read_only_for_continuous_augmented_nonlinear_problems() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation =
            ScalarCooperation::for_builtin(super::super::IDENTITY, Some(&runtime));
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("InitialPenalty", Value::from("inactive"));
        fields.insert("PenaltyFactor", Value::from("inactive"));
        let controls = Value::Struct(fields.finish());
        for activity in [
            Activity::Linear,
            Activity::IntegerLinear,
            Activity::IntegerNonlinear,
        ] {
            assert!(Prepared::new(controls.clone(), activity, &mut cooperation)
                .await
                .is_ok());
        }
        assert!(
            Prepared::new(controls, Activity::Nonlinear, &mut cooperation)
                .await
                .is_err()
        );
    });
}
#[test]
fn operator_grammar_keeps_constructor_and_solver_error_owners_distinct() {
    let value = Value::from("not-a-genetic-mutation");
    let constructor =
        admission::validate(Owner::Constructor, Field::MutationFcn, &value).unwrap_err();
    assert_eq!(
        constructor.identifier(),
        runmat_builtins::OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE.identifier
    );
    let solver = admission::validate(Owner::Solver, Field::MutationFcn, &value).unwrap_err();
    assert_eq!(
        solver.identifier(),
        runmat_builtins::GA_ERROR_INVALID_ARGUMENT.identifier
    );
}
