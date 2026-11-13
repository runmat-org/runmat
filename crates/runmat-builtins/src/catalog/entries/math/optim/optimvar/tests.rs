use super::*;
use runmat_types::*;
#[test]
fn variable_facts_keep_symbolic_class_alias_and_known_dimensions_without_numeric_placeholder() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            scalar.clone(),
            scalar,
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("x".into()),
            LiteralValue::Number(3.),
            LiteralValue::Number(2.),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let output = inference::constructor(&request, &OPTIMVAR_CATALOG_ENTRY);
    assert!(output.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &output.outputs[0].kind else {
        panic!("symbolic variable, not double")
    };
    assert_eq!(
        object.runtime_class.as_ref(),
        Some(&standard::OPTIMIZATION_VARIABLE.owned())
    );
    assert_eq!(object.handle_semantics, Some(true));
    assert!(!object.properties_complete);
    assert_eq!(
        output.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(2)])
    );
    assert_eq!(output.outputs[0].alias, AliasFact::Identity);
    // Text dimensions are not constructor-property boundaries. Named/mixed
    // and vector dimensions without an established literal geometry stay unknown.
    for dimensions in [
        vec![
            LiteralValue::String("north".into()),
            LiteralValue::Number(3.),
        ],
        vec![LiteralValue::String("north".into())],
        vec![LiteralValue::Unknown],
    ] {
        let mut mixed = request.clone();
        mixed.arguments.truncate(1);
        mixed
            .arguments
            .extend(dimensions.iter().map(|literal| match literal {
                LiteralValue::String(_) => ValueFact::scalar(ValueKindFact::String),
                _ => ValueFact::unknown(DynamicReason::RuntimeValue),
            }));
        let mut literals = vec![LiteralValue::String("x".into())];
        literals.extend(dimensions);
        mixed.literals = LiteralContext::new(literals);
        assert_eq!(
            inference::constructor(&mixed, &OPTIMVAR_CATALOG_ENTRY).outputs[0].shape,
            ShapeFact::Unknown
        );
    }
    let mut bounded = request.clone();
    bounded.arguments.truncate(1);
    bounded.arguments.extend([
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ]);
    bounded.literals = LiteralContext::new(vec![
        LiteralValue::String("x".into()),
        LiteralValue::String("lowerbound".into()),
        LiteralValue::Number(0.),
    ]);
    assert_eq!(
        inference::constructor(&bounded, &OPTIMVAR_CATALOG_ENTRY).outputs[0].shape,
        ShapeFact::Scalar
    );
    assert_eq!(
        OptimizationVariableProperty::parse_name_value("upperbound"),
        Some(OptimizationVariableProperty::UpperBound)
    );
    let declaration =
        crate::standard_class_declaration(&standard::OPTIMIZATION_VARIABLE.owned()).unwrap();
    assert_eq!(declaration.properties.len(), 5);
    let expression =
        crate::standard_class_declaration(&standard::OPTIMIZATION_EXPRESSION.owned()).unwrap();
    assert_eq!(
        declaration.methods.len() + 1,
        expression
            .methods
            .iter()
            .filter(|method| !method.is_static)
            .count()
    );
    for method in expression.methods.iter().filter(|method| {
        !method.is_static
            && !["subsref", "subsasgn", "evaluate"].contains(&method.name.display_name())
    }) {
        assert!(declaration
            .methods
            .iter()
            .any(|variable| variable.name == method.name && variable.callable == method.callable));
    }
    assert!(declaration
        .methods
        .iter()
        .all(|method| matches!(&method.callable, CallableIdentity::Builtin(_))));
}
