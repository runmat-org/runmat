#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::Value;
use test_helpers::execute_source;

#[test]
fn typed_study_sequence_compiles_and_executes_through_the_vm() {
    let values = execute_source(
        r#"
        object = fea.StepObjectRef.boundary("fixed");
        lifecycle = fea.ActivationLifecycle.activate();
        activation = fea.StepObjectActivation.new(object, lifecycle);

        kind = fea.StateKind.structuralPrimary();
        completeness = fea.StateCompleteness.complete();
        persistence = fea.StatePersistence.canonical();
        state = fea.StateOutput.new("equilibrium", kind, completeness, persistence);

        step = fea.StructuralStep.static("static");
        domain = fea.IndependentDomain.staticDomain();
        accuracy = fea.AccuracyPlan.baseline("acceptance");
        restart = fea.RestartPolicy.fresh();
        step = fea.ProcedureStep.configure( ...
            step, domain, {activation}, cell(0, 0), {state}, cell(0, 0), accuracy, restart);
        sequence = fea.AnalysisSequence.new( ...
            "production", {step}, cell(0, 0), cell(0, 0), cell(0, 0));
        backend = fea.ComputeBackend.cpu();
        solverOptions = fea.SolverOptions.defaults(step);
        executionStep = fea.ExecutionStep.new(step, backend, solverOptions);
        execution = fea.ExecutionPlan.new(sequence, {executionStep});
        "#,
    )
    .expect("typed sequence should compile and execute");

    for class_name in [
        "fea.StepObjectActivation",
        "fea.StateOutput",
        "fea.StructuralStep",
        "fea.AnalysisSequence",
        "fea.ComputeBackend",
        "fea.SolverOptions",
        "fea.ExecutionStep",
        "fea.ExecutionPlan",
    ] {
        assert!(
            values.iter().any(
                |value| matches!(value, Value::Object(object) if object.class_name().display_name() == class_name)
            ),
            "VM result should retain {class_name}"
        );
    }
}
