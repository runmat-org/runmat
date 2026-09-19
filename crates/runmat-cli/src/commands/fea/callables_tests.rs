use runmat_types::{
    NumericClass, NumericDomain, NumericFact, OperationParameterArity, OperationTypePattern,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use runmat_analysis_language::SourceSymbol;
use runmat_package::ProjectSymbolDefinition;

use super::{facts::exact_definition, operation_pattern, source_parameters};

#[test]
fn source_parameters_preserve_required_optional_and_variadic_arity() {
    let source = r#"
function y = pulse(t, width, varargin)
arguments
    t
    width = 0.1
end
y = t + width;
end
"#;
    let parameters = source_parameters(source, "pulse", runmat_parser::CompatMode::Matlab).unwrap();
    assert_eq!(parameters.len(), 3);
    assert_eq!(parameters[0].name, "t");
    assert_eq!(parameters[0].arity, OperationParameterArity::Required);
    assert_eq!(parameters[1].name, "width");
    assert_eq!(parameters[1].arity, OperationParameterArity::Optional);
    assert_eq!(parameters[2].name, "varargin");
    assert_eq!(parameters[2].arity, OperationParameterArity::Variadic);
}

#[test]
fn imported_value_facts_map_to_dependency_neutral_operation_types() {
    let numeric = NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    };
    assert_eq!(
        operation_pattern(&ValueFact::scalar(ValueKindFact::Numeric(numeric))),
        OperationTypePattern::NumericScalar
    );
    assert_eq!(
        operation_pattern(&ValueFact::proven(
            ValueKindFact::Numeric(numeric),
            ShapeFact::Ranked { rank: 2 },
            StorageFact::Dense
        )),
        OperationTypePattern::NumericArray
    );
    assert_eq!(
        operation_pattern(&ValueFact::scalar(ValueKindFact::String)),
        OperationTypePattern::StringScalar
    );
}

#[test]
fn imported_function_resolution_is_scoped_to_the_exact_module() {
    let first = definition("project/first/pulse.m");
    let second = definition("project/second/pulse.m");
    let definitions = vec![first, second.clone()];
    let selected = exact_definition(
        &definitions,
        std::path::Path::new("project/second/pulse.m"),
        &SourceSymbol::new("pulse").unwrap(),
    )
    .unwrap();
    assert_eq!(selected.source_path, second.source_path);
}

fn definition(path: &str) -> ProjectSymbolDefinition {
    ProjectSymbolDefinition {
        name: "pulse".into(),
        qualified_name: "pulse".into(),
        source_path: path.into(),
        package_name: "project".into(),
        dependency_alias: None,
        package_instance: None,
        source_id: None,
        is_private: false,
    }
}
