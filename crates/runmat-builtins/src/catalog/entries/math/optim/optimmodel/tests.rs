use super::*;
use runmat_types::*;
#[test]
fn symbolic_class_shape_and_comparison_roles_never_become_numeric_placeholders() {
    let number = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut request = CallRequest {
        arguments: vec![number.clone(), number],
        literals: LiteralContext::new(vec![LiteralValue::Number(2.), LiteralValue::Number(3.)]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let nullary = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let initial = inference::constructor(&nullary, &OPTIMEXPR_CATALOG_ENTRY);
    assert!(initial.diagnostics.is_empty());
    assert_eq!(
        initial.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    let result = inference::constructor(&request, &OPTIMEXPR_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("symbolic expression")
    };
    assert_eq!(
        object.runtime_class,
        Some(standard::OPTIMIZATION_EXPRESSION.owned())
    );
    assert_eq!(object.handle_semantics, Some(false));
    request.literals = LiteralContext::new(vec![
        LiteralValue::String("north".into()),
        LiteralValue::Number(3.),
    ]);
    assert_eq!(
        inference::constructor(&request, &OPTIMEXPR_CATALOG_ENTRY).outputs[0].shape,
        ShapeFact::Unknown
    );
    for (entry, class) in [
        (&OPTIMEXPR_EQ_CATALOG_ENTRY, standard::OPTIMIZATION_EQUALITY),
        (
            &OPTIMEXPR_LE_CATALOG_ENTRY,
            standard::OPTIMIZATION_INEQUALITY,
        ),
        (
            &OPTIMEXPR_GE_CATALOG_ENTRY,
            standard::OPTIMIZATION_INEQUALITY,
        ),
    ] {
        let output = crate::infer_catalog_call(entry, &request);
        let ValueKindFact::Object(object) = &output.outputs[0].kind else {
            panic!("typed comparison")
        };
        assert_eq!(object.runtime_class, Some(class.owned()));
        let declaration = crate::standard_class_declaration(&class.owned()).unwrap();
        assert_eq!(
            declaration.parent,
            Some(standard::OPTIMIZATION_CONSTRAINT.owned().qualified_name())
        );
    }
    let declaration =
        crate::standard_class_declaration(&standard::OPTIMIZATION_EXPRESSION.owned()).unwrap();
    let variables = declaration
        .properties
        .iter()
        .find(|p| p.name.display_name() == "Variables")
        .unwrap();
    assert_eq!(variables.attributes.set_access, MemberAccess::Private);
    assert!(declaration
        .methods
        .iter()
        .any(|method| method.name.display_name() == "sum"));
    assert!(declaration
        .methods
        .iter()
        .any(|method| method.name.display_name() == "plus"));
}

#[test]
fn numerical_method_declarations_derive_from_actual_canonical_entries() {
    let declaration =
        crate::standard_class_declaration(&standard::OPTIMIZATION_EXPRESSION.owned()).unwrap();
    for (name, entry) in OPTIMMODEL_NUMERICAL_METHODS {
        assert!(declaration
            .methods
            .iter()
            .any(|method| name.is(&method.name)
                && method.callable
                    == CallableIdentity::Builtin(BuiltinId(entry.identity.name.into()))));
    }
    let zero = declaration
        .methods
        .iter()
        .find(|method| OPTIMEXPR_ZEROS_METHOD.is(&method.name))
        .unwrap();
    assert!(zero.is_static);
    let like = declaration
        .methods
        .iter()
        .find(|method| OPTIMEXPR_ZEROS_LIKE_METHOD.is(&method.name))
        .unwrap();
    assert!(!like.is_static);
    assert!(like.attributes.is_hidden);
}
