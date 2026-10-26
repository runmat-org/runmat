use super::*;
use crate::OptimizationProblemProperty;
use runmat_types::*;
#[test]
fn problem_class_properties_and_results_keep_native_value_roles() {
    let request = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let facts = inference::constructor(&request, &OPTIMPROBLEM_CATALOG_ENTRY);
    assert!(facts.diagnostics.is_empty());
    assert_eq!(facts.outputs[0].shape, ShapeFact::Scalar);
    let ValueKindFact::Object(object) = &facts.outputs[0].kind else {
        panic!("native problem object")
    };
    assert_eq!(
        object.runtime_class,
        Some(standard::OPTIMIZATION_PROBLEM.owned())
    );
    assert_eq!(object.handle_semantics, Some(false));
    let class = crate::standard_class_declaration(&standard::OPTIMIZATION_PROBLEM.owned()).unwrap();
    assert_eq!(class.kind, ClassKind::Value);
    assert_eq!(
        class.properties.len(),
        OptimizationProblemProperty::ALL.len()
    );
    for property in OptimizationProblemProperty::ALL {
        let declaration = class
            .properties
            .iter()
            .find(|p| p.name.display_name() == property.name())
            .unwrap();
        assert_eq!(declaration.attributes.get_access, MemberAccess::Public);
        assert_eq!(
            declaration.attributes.set_access,
            if property == OptimizationProblemProperty::Variables {
                MemberAccess::Private
            } else {
                MemberAccess::Public
            }
        );
        assert!(declaration.attributes.is_dependent);
    }
    assert!(class
        .methods
        .iter()
        .all(|method| matches!(method.callable, CallableIdentity::Builtin(_))));
    assert!(matches!(
        inference::read(&request, &OPTIMPROBLEM_SUBSREF_CATALOG_ENTRY).outputs[0].kind,
        ValueKindFact::Unknown
    ));
    let mut solve_request = CallRequest {
        arguments: vec![facts.outputs[0].clone()],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(5)),
    };
    let entry = &OPTIMPROBLEM_SOLVE_CATALOG_ENTRY;
    let crate::BuiltinInferenceRule::Identity(rule) = entry.contract.inference_rule else {
        panic!("canonical solve inference");
    };
    let solved = rule.infer(&solve_request, entry);
    assert!(solved.diagnostics.is_empty());
    assert_eq!(solved.outputs.len(), 5);
    let ValueKindFact::Object(flag) = &solved.outputs[2].kind else {
        panic!("termination enum fact");
    };
    assert_eq!(
        flag.runtime_class,
        Some(standard::OPTIMIZATION_EXITFLAG.owned())
    );
    assert_eq!(solved.outputs[1].shape, ShapeFact::Unknown);
    // The objective DAG is not encoded in this object fact: default selection
    // can be LSQ and its observed solver information consumes original Clock.
    assert!(solved.effects.0.contains(&EffectKind::Clock));
    assert!(solved.effects.0.contains(&EffectKind::HostCallback));
    let mut lsq_request = solve_request.clone();
    lsq_request.arguments.extend([
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
            Default::default(),
            false,
        ))),
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::String),
    ]);
    lsq_request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Solver".into()),
        LiteralValue::String("lsqlin".into()),
    ]);
    let selected = rule.infer(&lsq_request, entry);
    assert!(selected.diagnostics.is_empty());
    let ValueKindFact::Struct(output) = &selected.outputs[3].kind else {
        panic!("solver information");
    };
    assert!(!output.fields_complete && output.fields.contains_key("wallclocktime"));
    assert!(
        output.fields.contains_key("iterations") && output.fields.contains_key("firstorderopt")
    );
    for name in [
        "dualitygap",
        "constrviolation",
        "linearsolver",
        "cgiterations",
    ] {
        assert!(!output.fields.contains_key(name));
    }
    assert!(!output.fields.contains_key("primalfeasibility"));
    let mut qp_request = lsq_request.clone();
    qp_request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Solver".into()),
        LiteralValue::String("quadprog".into()),
    ]);
    let qp = rule.infer(&qp_request, entry);
    assert!(qp.diagnostics.is_empty());
    let ValueKindFact::Struct(info) = &qp.outputs[3].kind else {
        panic!("QP information")
    };
    assert!(info.fields_complete && info.fields.contains_key("wallclocktime"));
    assert!(!info.fields.contains_key("linearsolver"));
    assert!(qp.effects.0.contains(&EffectKind::Clock));
    qp_request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(3));
    assert!(!rule
        .infer(&qp_request, entry)
        .effects
        .0
        .contains(&EffectKind::Clock));
    assert_eq!(OptimizationExitflag::Nonconvex.numeric(), -6);
    let mut nonlinear_request = qp_request.clone();
    nonlinear_request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(5));
    nonlinear_request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Solver".into()),
        LiteralValue::String("fmincon".into()),
    ]);
    let nonlinear = rule.infer(&nonlinear_request, entry);
    assert!(nonlinear.diagnostics.is_empty());
    let ValueKindFact::Struct(info) = &nonlinear.outputs[3].kind else {
        panic!("nonlinear information");
    };
    assert!(info.fields_complete);
    assert!(info.fields.contains_key("funcCount") && info.fields.contains_key("wallclocktime"));
    assert!(!info.fields.contains_key("dualitygap"));
    assert_eq!(info.fields["constrviolation"].shape, ShapeFact::Unknown);
    assert_eq!(OptimizationExitflag::StepSizeBelowTolerance.numeric(), 2);
    assert_eq!(
        OptimizationExitflag::FunctionChangeBelowTolerance.numeric(),
        3
    );
    assert_eq!(OptimizationExitflag::OutputFcnStop.numeric(), -1);
    let mut residual_request = nonlinear_request.clone();
    residual_request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Solver".into()),
        LiteralValue::String("lsqnonlin".into()),
    ]);
    let residual = rule.infer(&residual_request, entry);
    assert!(residual.diagnostics.is_empty());
    let ValueKindFact::Struct(info) = &residual.outputs[3].kind else {
        panic!("nonlinear residual information");
    };
    assert!(info.fields_complete);
    for name in [
        "iterations",
        "funcCount",
        "firstorderopt",
        "stepsize",
        "cgiterations",
        "lssteplength",
        "wallclocktime",
    ] {
        assert!(info.fields.contains_key(name));
    }
    assert!(!info.fields.contains_key("dualitygap"));
    assert!(!info.fields.contains_key("constrviolation"));
    assert!(residual.effects.0.contains(&EffectKind::Clock));
    residual_request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(3));
    assert!(!rule
        .infer(&residual_request, entry)
        .effects
        .0
        .contains(&EffectKind::Clock));
    let mut unconstrained_request = nonlinear_request.clone();
    unconstrained_request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Solver".into()),
        LiteralValue::String("fminunc".into()),
    ]);
    let unconstrained = rule.infer(&unconstrained_request, entry);
    assert!(unconstrained.diagnostics.is_empty());
    let ValueKindFact::Struct(info) = &unconstrained.outputs[3].kind else {
        panic!("unconstrained information");
    };
    assert!(info.fields_complete);
    for name in [
        "funcCount",
        "firstorderopt",
        "stepsize",
        "objectivederivative",
    ] {
        assert!(info.fields.contains_key(name));
    }
    for name in ["wallclocktime", "constrviolation", "constraintCallCount"] {
        assert!(!info.fields.contains_key(name));
    }
    assert_eq!(unconstrained.outputs[4].shape.element_count(), Some(0));
    assert!(!unconstrained.effects.0.contains(&EffectKind::Clock));
    let mut integer_request = qp_request.clone();
    integer_request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(5));
    integer_request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Solver".into()),
        LiteralValue::String("intlinprog".into()),
    ]);
    let integer = rule.infer(&integer_request, entry);
    assert!(integer.diagnostics.is_empty());
    let ValueKindFact::Struct(info) = &integer.outputs[3].kind else {
        panic!("integer information");
    };
    assert!(info.fields_complete);
    for name in [
        "numnodes",
        "numfeaspoints",
        "relativegap",
        "absolutegap",
        "certifiedobjective",
        "lowerbound",
    ] {
        assert!(info.fields.contains_key(name));
    }
    assert!(!info.fields.contains_key("iterations"));
    assert_eq!(integer.outputs[4].shape.element_count(), Some(0));
    integer_request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(1));
    assert!(rule
        .infer(&integer_request, entry)
        .effects
        .0
        .contains(&EffectKind::Clock));
    assert_eq!(OptimizationExitflag::IntegerFeasible.numeric(), 2);
    assert_ne!(
        OptimizationExitflag::IntegerFeasible,
        OptimizationExitflag::StepSizeBelowTolerance
    );
    lsq_request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(3));
    assert!(!rule
        .infer(&lsq_request, entry)
        .effects
        .0
        .contains(&EffectKind::Clock));
    solve_request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(6));
    assert!(!rule.infer(&solve_request, entry).diagnostics.is_empty());
    let flag_class =
        crate::standard_class_declaration(&standard::OPTIMIZATION_EXITFLAG.owned()).unwrap();
    for method in ["double", "string", "eq", "ne"] {
        assert!(flag_class
            .methods
            .iter()
            .any(|candidate| candidate.name.display_name() == method));
    }
    assert_eq!(OptimizationExitflag::EquationSolved.numeric(), 1);
    assert_eq!(
        OptimizationExitflag::EquationSolved.name(),
        "EquationSolved"
    );
    assert_ne!(
        OptimizationExitflag::EquationSolved,
        OptimizationExitflag::OptimalSolution
    );
    for flag in OptimizationExitflag::ALL {
        assert_eq!(OptimizationExitflag::parse_member(flag.name()), Some(flag));
    }
}
