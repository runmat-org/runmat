use runmat_analysis_language::*;
use runmat_design_syntax::{parse_fea, FeaModuleItem, FeaStatementKind};
use runmat_types::{
    OperationDistribution, OperationEffectFacts, OperationFacts, OperationFrontend,
    OperationIdentity, OperationLoweringId, OperationOverloadFacts, OperationOverloadIdentity,
    OperationParameter, OperationParameterArity, OperationQuantityRule, OperationSignature,
    OperationTypePattern, TopologyApplicability,
};

struct Catalog {
    operation: OperationFacts,
}

impl AnalysisOperationCatalog for Catalog {
    fn fingerprint(&self) -> [u8; 32] {
        [4; 32]
    }

    fn resolve(
        &self,
        request: &OperationResolutionRequest,
    ) -> Result<ResolvedOperationFacts, OperationDiagnostic> {
        if request.operation == self.operation.identity {
            Ok(ResolvedOperationFacts {
                operation: self.operation.clone(),
            })
        } else {
            Err(OperationDiagnostic {
                code: "fea.operation.unknown".to_string(),
                message: format!("unknown `{}`", request.operation),
            })
        }
    }
}

// witness: FEA-AUTHORING-OUTPUT-002/overload_contract @
//   shared_infrastructure/mesh_independent/source_check; coverage: partial;
//   boundary: Expression checking preserves dimensions and resolves catalog overloads; availability
//   and documentation fields of the overload contract are not asserted.
#[test]
fn expression_checking_preserves_dimensions_and_resolves_catalog_overloads() {
    let source =
        "study s {\noutput scaled = 2 * input.load\noutput peak = maximum(input.load)\n}\n";
    let syntax = parse_fea(source);
    let FeaModuleItem::Declaration(declaration) = &syntax.items[0] else {
        panic!("expected study declaration");
    };
    let expressions = declaration
        .body
        .statements
        .iter()
        .map(|statement| match &statement.kind {
            FeaStatementKind::Output { value, .. } => value,
            _ => panic!("expected output"),
        })
        .collect::<Vec<_>>();
    let mut scope = ExpressionScope::new();
    scope.insert_reference(
        ["input".to_string(), "load".to_string()],
        FeaType::Quantity((PhysicalDimension::FORCE).into()),
    );
    let catalog = Catalog {
        operation: OperationFacts {
            identity: OperationIdentity::new("maximum").unwrap(),
            category: "reduction".to_string(),
            summary: "Maximum value".to_string(),
            overloads: vec![OperationOverloadFacts {
                identity: OperationOverloadIdentity::new("maximum:fea-scalar").unwrap(),
                frontends: vec![OperationFrontend::Fea],
                signature: OperationSignature {
                    inputs: vec![parameter("values", OperationTypePattern::NumericScalar)],
                    outputs: vec![parameter("value", OperationTypePattern::NumericScalar)],
                },
                quantity: OperationQuantityRule::PreserveInput { input: 0 },
                topology: TopologyApplicability::NotApplicable,
                effects: OperationEffectFacts::default(),
                distribution: OperationDistribution::DeterministicReduction,
                lowering: OperationLoweringId::new("fea.maximum").unwrap(),
            }],
        },
    };
    for expression in expressions {
        let (checked, diagnostics) = check_expression(
            ExpressionCheck {
                path: &LogicalSourcePath::new("project/main.fea").unwrap(),
                source,
                scope: &scope,
                catalog: &catalog,
            },
            expression,
        );
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        assert_eq!(
            checked.value_type,
            FeaType::Quantity((PhysicalDimension::FORCE).into())
        );
    }
}

#[test]
fn expression_checking_rejects_dimensionally_invalid_addition() {
    let source = "study s {\noutput bad = 1 m + 1 s\n}\n";
    let syntax = parse_fea(source);
    let FeaModuleItem::Declaration(declaration) = &syntax.items[0] else {
        panic!("expected study declaration");
    };
    let FeaStatementKind::Output { value, .. } = &declaration.body.statements[0].kind else {
        panic!("expected output");
    };
    let catalog = Catalog {
        operation: OperationFacts {
            identity: OperationIdentity::new("unused").unwrap(),
            category: "test".to_string(),
            summary: "Unused".to_string(),
            overloads: Vec::new(),
        },
    };
    let (checked, diagnostics) = check_expression(
        ExpressionCheck {
            path: &LogicalSourcePath::new("project/main.fea").unwrap(),
            source,
            scope: &ExpressionScope::new(),
            catalog: &catalog,
        },
        value,
    );
    assert_eq!(checked.value_type, FeaType::Error);
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "fea.semantic.incompatible_addition"));
}

#[test]
fn vector_quantities_retain_shape_and_dimension() {
    let source = "study s {\noutput force = [0, 1, 0] kN\n}\n";
    let syntax = parse_fea(source);
    let FeaModuleItem::Declaration(declaration) = &syntax.items[0] else {
        panic!("expected study declaration");
    };
    let FeaStatementKind::Output { value, .. } = &declaration.body.statements[0].kind else {
        panic!("expected output");
    };
    let catalog = Catalog {
        operation: OperationFacts {
            identity: OperationIdentity::new("unused").unwrap(),
            category: "test".to_string(),
            summary: "Unused".to_string(),
            overloads: Vec::new(),
        },
    };
    let (checked, diagnostics) = check_expression(
        ExpressionCheck {
            path: &LogicalSourcePath::new("project/main.fea").unwrap(),
            source,
            scope: &ExpressionScope::new(),
            catalog: &catalog,
        },
        value,
    );
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");
    assert_eq!(
        checked.value_type,
        FeaType::Vector {
            element: Box::new(FeaType::Quantity((PhysicalDimension::FORCE).into())),
            length: Some(3),
        }
    );
}

fn parameter(name: &str, value: OperationTypePattern) -> OperationParameter {
    OperationParameter {
        name: name.to_string(),
        value,
        arity: OperationParameterArity::Required,
        default: None,
    }
}
