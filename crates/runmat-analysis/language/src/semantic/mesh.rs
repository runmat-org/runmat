use super::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaType, PhysicalDimension,
};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey,
    FeaSymbolKind, FeaSymbolOrigin, LogicalSourcePath,
};
use runmat_geometry_core::GeometryDefinitionRef;
use runmat_meshing_core::{ElementOrder, MeshDefinition, MeshingRequestSettings};
use std::collections::{BTreeMap, BTreeSet};

pub fn lower_mesh_definitions(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), MeshDefinition> {
    let mut result = BTreeMap::new();
    for (identity, declaration) in declarations {
        if identity.1.kind != FeaSymbolKind::Mesh {
            continue;
        }
        let initial_errors = diagnostics.len();
        let definition = definition(bound, declaration, diagnostics);
        if diagnostics.len() != initial_errors {
            continue;
        }
        let Some(definition) = definition else {
            continue;
        };
        match definition.validate() {
            Ok(()) => {
                result.insert(identity.clone(), definition);
            }
            Err(error) => diagnostic(
                declaration,
                declaration.span,
                error.to_string(),
                diagnostics,
            ),
        }
    }
    result
}

fn definition(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<MeshDefinition> {
    let geometry = geometry_reference(bound, declaration, diagnostics)?;
    let tetrahedra = unique_block(declaration, "tetrahedra", diagnostics)?;
    reject_unsupported_top_level(declaration, diagnostics);
    reject_unsupported_settings(declaration, tetrahedra, diagnostics);

    let order =
        required_enum(declaration, tetrahedra, "order", diagnostics).and_then(
            |value| match value {
                "linear" => Some(ElementOrder::Tet4),
                "quadratic" => Some(ElementOrder::Tet10),
                _ => {
                    diagnostic(
                        declaration,
                        declaration.span,
                        "mesh `order` must be `linear` or `quadratic`",
                        diagnostics,
                    );
                    None
                }
            },
        );
    let target_edge_length_m = required_length(declaration, tetrahedra, "target_size", diagnostics);
    let maximum_chordal_deviation_m =
        required_length(declaration, tetrahedra, "maximum_deviation", diagnostics);

    let defaults = MeshingRequestSettings::default();
    let maximum_grading_ratio = optional_scalar(
        declaration,
        tetrahedra,
        "maximum_grading_ratio",
        diagnostics,
    )
    .unwrap_or(defaults.maximum_grading_ratio);
    let deterministic_seed =
        optional_integer(declaration, tetrahedra, "deterministic_seed", diagnostics)
            .unwrap_or(defaults.deterministic_seed);
    let settings = MeshingRequestSettings {
        element_order: order?,
        deterministic_seed,
        target_edge_length_m: target_edge_length_m?,
        maximum_chordal_deviation_m: maximum_chordal_deviation_m?,
        maximum_grading_ratio,
        ..defaults
    };

    Some(MeshDefinition {
        module_path: declaration.module.as_str().to_string(),
        mesh_id: declaration.key.name.clone(),
        geometry,
        settings,
    })
}

fn geometry_reference(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryDefinitionRef> {
    let Some([namespace, name]) = declaration.target.as_deref() else {
        diagnostic(
            declaration,
            declaration.span,
            "mesh requires `on geometry.<name>`",
            diagnostics,
        );
        return None;
    };
    if namespace != "geometry" {
        diagnostic(
            declaration,
            declaration.span,
            "mesh target must use the `geometry` namespace",
            diagnostics,
        );
        return None;
    }
    let key = FeaSymbolKey {
        kind: FeaSymbolKind::Geometry,
        name: name.clone(),
    };
    let Some(symbol) = bound
        .symbols
        .get(&declaration.module)
        .and_then(|symbols| symbols.get(&key))
    else {
        diagnostic(
            declaration,
            declaration.span,
            format!("mesh references unknown geometry `{name}`"),
            diagnostics,
        );
        return None;
    };
    match &symbol.origin {
        FeaSymbolOrigin::Local => Some(GeometryDefinitionRef {
            module_path: declaration.module.as_str().to_string(),
            geometry_id: name.clone(),
        }),
        FeaSymbolOrigin::Import {
            target_module,
            target,
        } => Some(GeometryDefinitionRef {
            module_path: target_module.as_str().to_string(),
            geometry_id: target.name.clone(),
        }),
        FeaSymbolOrigin::Function => {
            diagnostic(
                declaration,
                declaration.span,
                "mesh target resolved to a function",
                diagnostics,
            );
            None
        }
    }
}

fn unique_block<'a>(
    declaration: &'a CheckedFeaDeclaration,
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a [CheckedFeaStatement]> {
    let matching = declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::Block {
                header, statements, ..
            } = &statement.kind
            else {
                return None;
            };
            (header.trim() == name).then_some(statements.as_slice())
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        diagnostic(
            declaration,
            declaration.span,
            format!("mesh requires exactly one `{name}` block"),
            diagnostics,
        );
        None
    } else {
        matching.into_iter().next()
    }
}

fn reject_unsupported_top_level(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    for statement in &declaration.statements {
        if matches!(
            &statement.kind,
            CheckedFeaStatementKind::Block { header, .. } if header.trim() == "tetrahedra"
        ) {
            continue;
        }
        diagnostic(
            declaration,
            statement.span,
            "mesh currently accepts only its `tetrahedra` request block",
            diagnostics,
        );
    }
}

fn reject_unsupported_settings(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let supported = BTreeSet::from([
        "order",
        "target_size",
        "maximum_deviation",
        "maximum_grading_ratio",
        "deterministic_seed",
    ]);
    let mut seen = BTreeSet::new();
    for statement in statements {
        let CheckedFeaStatementKind::Assignment {
            target, relation, ..
        } = &statement.kind
        else {
            diagnostic(
                declaration,
                statement.span,
                "tetrahedra settings must be assignments",
                diagnostics,
            );
            continue;
        };
        let Some(name) = target.as_slice().first().filter(|_| target.len() == 1) else {
            diagnostic(
                declaration,
                statement.span,
                "mesh setting names cannot be qualified",
                diagnostics,
            );
            continue;
        };
        if *relation != runmat_parser::fea::FeaAssignmentRelation::Equal {
            diagnostic(
                declaration,
                statement.span,
                "mesh settings require `=`",
                diagnostics,
            );
        }
        if !supported.contains(name.as_str()) {
            diagnostic(
                declaration,
                statement.span,
                format!("unsupported tetrahedra setting `{name}`"),
                diagnostics,
            );
        } else if !seen.insert(name.as_str()) {
            diagnostic(
                declaration,
                statement.span,
                format!("duplicate tetrahedra setting `{name}`"),
                diagnostics,
            );
        }
    }
}

fn assignment<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a CheckedExpression> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        (target.as_slice() == [name]).then_some(value)
    })
}

fn required_enum<'a>(
    declaration: &CheckedFeaDeclaration,
    statements: &'a [CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a str> {
    let Some(value) = assignment(statements, name) else {
        missing(declaration, name, diagnostics);
        return None;
    };
    let CheckedExpressionKind::Enum(value) = &value.kind else {
        invalid(declaration, value, name, "enum value", diagnostics);
        return None;
    };
    Some(value)
}

fn required_length(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let Some(value) = assignment(statements, name) else {
        missing(declaration, name, diagnostics);
        return None;
    };
    if value.value_type != FeaType::Quantity(PhysicalDimension::LENGTH) {
        invalid(declaration, value, name, "length quantity", diagnostics);
        return None;
    }
    scalar(value).or_else(|| {
        invalid(
            declaration,
            value,
            name,
            "constant length quantity",
            diagnostics,
        );
        None
    })
}

fn optional_scalar(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let value = assignment(statements, name)?;
    if !matches!(value.value_type, FeaType::Integer | FeaType::Real) {
        invalid(
            declaration,
            value,
            name,
            "dimensionless scalar",
            diagnostics,
        );
        return None;
    }
    scalar(value).or_else(|| {
        invalid(declaration, value, name, "constant scalar", diagnostics);
        None
    })
}

fn optional_integer(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<u64> {
    let value = assignment(statements, name)?;
    if value.value_type != FeaType::Integer {
        invalid(
            declaration,
            value,
            name,
            "non-negative integer",
            diagnostics,
        );
        return None;
    }
    let CheckedExpressionKind::Number(source) = &value.kind else {
        invalid(
            declaration,
            value,
            name,
            "constant non-negative integer",
            diagnostics,
        );
        return None;
    };
    source.parse().ok().or_else(|| {
        invalid(
            declaration,
            value,
            name,
            "non-negative 64-bit integer",
            diagnostics,
        );
        None
    })
}

fn scalar(value: &CheckedExpression) -> Option<f64> {
    match &value.kind {
        CheckedExpressionKind::Number(source) => source.parse().ok(),
        CheckedExpressionKind::Quantity {
            value,
            scale_bits,
            offset_bits,
        } => Some(scalar(value)? * f64::from_bits(*scale_bits) + f64::from_bits(*offset_bits)),
        CheckedExpressionKind::Unary { operator, operand } => {
            let scalar = scalar(operand)?;
            Some(match operator {
                runmat_parser::fea::FeaUnaryOperator::Positive => scalar,
                runmat_parser::fea::FeaUnaryOperator::Negative => -scalar,
            })
        }
        _ => None,
    }
}

fn missing(declaration: &CheckedFeaDeclaration, name: &str, diagnostics: &mut Vec<FeaDiagnostic>) {
    diagnostic(
        declaration,
        declaration.span,
        format!("tetrahedra mesh requires `{name}`"),
        diagnostics,
    );
}

fn invalid(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    name: &str,
    expected: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostic(
        declaration,
        value.span,
        format!("mesh setting `{name}` requires a {expected}"),
        diagnostics,
    );
}

fn diagnostic(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.mesh",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}
