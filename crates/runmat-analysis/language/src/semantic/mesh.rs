use super::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaType, PhysicalDimension,
};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey,
    FeaSymbolKind, FeaSymbolOrigin, LogicalSourcePath,
};
use runmat_geometry_core::GeometryDefinitionRef;
use runmat_meshing_core::{ElementOrder, MeshDefinition, MeshDimension, MeshingRequestSettings};
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
    let (block, dimension) = element_block(declaration, diagnostics)?;
    let (header, settings_block) = block;
    reject_unsupported_top_level(declaration, diagnostics);
    reject_unsupported_settings(declaration, header, settings_block, diagnostics);

    // Shell triangles and beam segments are linear; solids choose their tetrahedral order.
    let order = if dimension == MeshDimension::Solid {
        required_enum(declaration, settings_block, "order", diagnostics).and_then(|value| match value
        {
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
        })
    } else {
        Some(ElementOrder::Tet4)
    };
    let target_edge_length_m = required_length(declaration, settings_block, "target_size", diagnostics);
    let maximum_chordal_deviation_m =
        required_length(declaration, settings_block, "maximum_deviation", diagnostics);

    let defaults = MeshingRequestSettings::default();
    let maximum_grading_ratio = optional_scalar(
        declaration,
        settings_block,
        "maximum_grading_ratio",
        diagnostics,
    )
    .unwrap_or(defaults.maximum_grading_ratio);
    let deterministic_seed =
        optional_integer(declaration, settings_block, "deterministic_seed", diagnostics)
            .unwrap_or(defaults.deterministic_seed);
    let settings = MeshingRequestSettings {
        element_order: order?,
        dimension,
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

/// The element block names the solver elements: `tetrahedra` for solids, `triangles` for
/// sheet faces meshed as shells (with their edges as segments), `lines` for edges only.
const ELEMENT_BLOCKS: [(&str, MeshDimension); 3] = [
    ("tetrahedra", MeshDimension::Solid),
    ("triangles", MeshDimension::Surface),
    ("lines", MeshDimension::Curve),
];

type ElementBlock<'a> = ((&'static str, &'a [CheckedFeaStatement]), MeshDimension);

fn element_block<'a>(
    declaration: &'a CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ElementBlock<'a>> {
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
            ELEMENT_BLOCKS
                .iter()
                .find(|(name, _)| header.trim() == *name)
                .map(|(name, dimension)| ((*name, statements.as_slice()), *dimension))
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        diagnostic(
            declaration,
            declaration.span,
            "mesh requires exactly one `tetrahedra`, `triangles`, or `lines` block",
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
            CheckedFeaStatementKind::Block { header, .. }
                if ELEMENT_BLOCKS.iter().any(|(name, _)| header.trim() == *name)
        ) {
            continue;
        }
        diagnostic(
            declaration,
            statement.span,
            "mesh accepts only its `tetrahedra`, `triangles`, or `lines` request block",
            diagnostics,
        );
    }
}

fn reject_unsupported_settings(
    declaration: &CheckedFeaDeclaration,
    block: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let mut supported = BTreeSet::from([
        "target_size",
        "maximum_deviation",
        "maximum_grading_ratio",
        "deterministic_seed",
    ]);
    if block == "tetrahedra" {
        supported.insert("order");
    }
    let mut seen = BTreeSet::new();
    for statement in statements {
        let CheckedFeaStatementKind::Assignment {
            target, relation, ..
        } = &statement.kind
        else {
            diagnostic(
                declaration,
                statement.span,
                format!("{block} settings must be assignments"),
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
                format!("unsupported {block} setting `{name}`"),
                diagnostics,
            );
        } else if !seen.insert(name.as_str()) {
            diagnostic(
                declaration,
                statement.span,
                format!("duplicate {block} setting `{name}`"),
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
        if let Some((message, label)) =
            super::units::unit_mismatch(name, value, PhysicalDimension::LENGTH)
        {
            diagnostics.push(
                FeaDiagnostic::error(
                    "fea.semantic.mesh",
                    message,
                    crate::FeaDiagnosticPhase::Lowering,
                    Some(crate::FeaDiagnosticLocation {
                        path: declaration.module.clone(),
                        span: value.span,
                    }),
                )
                .with_label(label),
            );
            return None;
        }
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
        CheckedExpressionKind::CadParameterValue(source) => Some(source.value_si()),
        CheckedExpressionKind::Quantity {
            value,
            scale_bits,
            offset_bits,
            ..
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
        format!("mesh element block requires `{name}`"),
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
