use super::{
    parse_type_syntax, DeclarationInput, DeclarationSignature, ExpressionScope, FeaTopologyRole,
    FeaType, FunctionSignature,
};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKind,
    FeaSymbolOrigin, LogicalSourcePath,
};
use runmat_design_syntax::{
    FeaBlock, FeaDeclaration, FeaExpression, FeaStatementKind, FeaTokenRange,
};
use runmat_types::Span;

pub(super) struct InputSyntax {
    pub name: String,
    pub value_type: FeaType,
    pub default: Option<FeaExpression>,
    pub minimum: Option<FeaExpression>,
    pub maximum: Option<FeaExpression>,
    pub span: Span,
}

pub(super) fn extend_module_scope(
    scope: &mut ExpressionScope,
    bound: &BoundFeaProgram,
    path: &LogicalSourcePath,
) {
    if let Some(symbols) = bound.symbols.get(path) {
        for symbol in symbols.values() {
            scope.insert_reference(
                [
                    symbol.key.kind.namespace().to_string(),
                    symbol.key.name.clone(),
                ],
                FeaType::Declaration(symbol.key.kind),
            );
            if symbol.key.kind == FeaSymbolKind::Geometry {
                let FeaSymbolOrigin::Import {
                    target_module,
                    target,
                } = &symbol.origin
                else {
                    continue;
                };
                let Some(target_syntax) = bound
                    .graph
                    .modules
                    .get(target_module)
                    .and_then(|module| module.syntax.as_ref())
                else {
                    continue;
                };
                let Some(target_declaration) = target_syntax.items.iter().find_map(|item| {
                    let runmat_design_syntax::FeaModuleItem::Declaration(declaration) = item else {
                        return None;
                    };
                    (declaration.kind == runmat_design_syntax::FeaDeclarationKind::Geometry
                        && declaration.name.text == target.name)
                        .then_some(declaration)
                }) else {
                    continue;
                };
                extend_geometry_members(
                    scope,
                    target_syntax.source(),
                    target_declaration,
                    &symbol.key.name,
                );
            }
        }
    }
    for ((module, name), facts) in &bound.functions {
        if module == path {
            scope.insert_function(
                ["function".to_string(), name.clone()],
                FunctionSignature {
                    inputs: facts.inputs.clone(),
                    outputs: facts.outputs.clone(),
                },
            );
        }
    }
    let Some(module) = bound.graph.modules.get(path) else {
        return;
    };
    let Some(syntax) = &module.syntax else {
        return;
    };
    for item in &syntax.items {
        let runmat_design_syntax::FeaModuleItem::Declaration(declaration) = item else {
            continue;
        };
        if declaration.kind != runmat_design_syntax::FeaDeclarationKind::Geometry {
            continue;
        }
        extend_geometry_members(scope, syntax.source(), declaration, &declaration.name.text);
    }
}

pub(super) fn declaration_inputs(
    source: &str,
    declaration: &FeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
    path: &LogicalSourcePath,
) -> Vec<InputSyntax> {
    let mut inputs = model_header_inputs(source, declaration, diagnostics, path);
    for statement in &declaration.body.statements {
        match &statement.kind {
            FeaStatementKind::Input {
                name,
                type_syntax,
                default,
            } => {
                if let Some(value_type) = parse_type_range(source, *type_syntax, path, diagnostics)
                {
                    inputs.push(InputSyntax {
                        name: name.text.clone(),
                        value_type,
                        default: default.clone(),
                        minimum: None,
                        maximum: None,
                        span: statement.span,
                    });
                }
            }
            FeaStatementKind::Block { header, body } => {
                if let Some((name, value_type)) = parse_input_header(
                    &source[header.span.start..header.span.end],
                    path,
                    header.span,
                    diagnostics,
                ) {
                    let members = input_members(body, path, diagnostics);
                    inputs.push(InputSyntax {
                        name,
                        value_type,
                        default: members.default,
                        minimum: members.minimum,
                        maximum: members.maximum,
                        span: statement.span,
                    });
                }
            }
            _ => {}
        }
    }
    inputs
}

pub(super) fn extend_declaration_signatures(
    scope: &mut ExpressionScope,
    bound: &BoundFeaProgram,
    path: &LogicalSourcePath,
) {
    let Some(module) = bound.graph.modules.get(path) else {
        return;
    };
    let Some(syntax) = &module.syntax else {
        return;
    };
    for item in &syntax.items {
        let runmat_design_syntax::FeaModuleItem::Declaration(declaration) = item else {
            continue;
        };
        let mut ignored_diagnostics = Vec::new();
        let inputs =
            declaration_inputs(syntax.source(), declaration, &mut ignored_diagnostics, path)
                .into_iter()
                .map(|input| {
                    (
                        input.name,
                        DeclarationInput {
                            value_type: input.value_type,
                            required: input.default.is_none(),
                        },
                    )
                })
                .collect();
        let kind = declaration_kind(declaration.kind);
        scope.insert_declaration(
            [kind.namespace().to_string(), declaration.name.text.clone()],
            DeclarationSignature {
                result: FeaType::Declaration(kind),
                inputs,
            },
        );
    }
}

fn model_header_inputs(
    source: &str,
    declaration: &FeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
    path: &LogicalSourcePath,
) -> Vec<InputSyntax> {
    if declaration.kind != runmat_design_syntax::FeaDeclarationKind::Model {
        return Vec::new();
    }
    let header = source[declaration.header.span.start..declaration.header.span.end].trim();
    let Some(parameters) = header
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
    else {
        return Vec::new();
    };
    split_parameters(parameters)
        .into_iter()
        .filter_map(|parameter| {
            let (left, right) = parameter.split_once(':')?;
            let name = left.split_whitespace().last()?.to_string();
            match parse_type_syntax(right) {
                Ok(value_type) => Some(InputSyntax {
                    name,
                    value_type,
                    default: None,
                    minimum: None,
                    maximum: None,
                    span: declaration.header.span,
                }),
                Err(error) => {
                    diagnostics.push(type_diagnostic(path, declaration.header.span, error));
                    None
                }
            }
        })
        .collect()
}

fn parse_input_header(
    header: &str,
    path: &LogicalSourcePath,
    span: Span,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(String, FeaType)> {
    let rest = header.trim().strip_prefix("input ")?;
    let Some((name, type_source)) = rest.split_once(':') else {
        diagnostics.push(FeaDiagnostic::error(
            "fea.semantic.input_type_missing",
            "input block requires a type",
            FeaDiagnosticPhase::Semantic,
            Some(FeaDiagnosticLocation {
                path: path.clone(),
                span,
            }),
        ));
        return None;
    };
    match parse_type_syntax(type_source) {
        Ok(value_type) => Some((name.trim().to_string(), value_type)),
        Err(error) => {
            diagnostics.push(type_diagnostic(path, span, error));
            None
        }
    }
}

fn parse_type_range(
    source: &str,
    range: FeaTokenRange,
    path: &LogicalSourcePath,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<FeaType> {
    match parse_type_syntax(&source[range.span.start..range.span.end]) {
        Ok(value_type) => Some(value_type),
        Err(error) => {
            diagnostics.push(type_diagnostic(path, range.span, error));
            None
        }
    }
}

fn type_diagnostic(
    path: &LogicalSourcePath,
    span: Span,
    error: super::TypeSyntaxError,
) -> FeaDiagnostic {
    FeaDiagnostic::error(
        "fea.semantic.invalid_type",
        error.to_string(),
        FeaDiagnosticPhase::Semantic,
        Some(FeaDiagnosticLocation {
            path: path.clone(),
            span,
        }),
    )
}

#[derive(Default)]
struct InputMembers {
    default: Option<FeaExpression>,
    minimum: Option<FeaExpression>,
    maximum: Option<FeaExpression>,
}

fn input_members(
    block: &FeaBlock,
    path: &LogicalSourcePath,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> InputMembers {
    let mut members = InputMembers::default();
    for statement in &block.statements {
        let FeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            diagnostics.push(FeaDiagnostic::error(
                "fea.semantic.input_member",
                "input blocks accept only `default`, `minimum`, and `maximum` assignments",
                FeaDiagnosticPhase::Semantic,
                Some(FeaDiagnosticLocation {
                    path: path.clone(),
                    span: statement.span,
                }),
            ));
            continue;
        };
        let Some(name) = (target.segments.len() == 1).then(|| target.segments[0].text.as_str())
        else {
            input_member_error(path, statement.span, "invalid input member", diagnostics);
            continue;
        };
        let slot = match (name, crate::FeaContextualWord::from_source(name)) {
            ("default", _) => &mut members.default,
            (_, Some(crate::FeaContextualWord::Minimum)) => &mut members.minimum,
            (_, Some(crate::FeaContextualWord::Maximum)) => &mut members.maximum,
            _ => {
                input_member_error(path, statement.span, "unknown input member", diagnostics);
                continue;
            }
        };
        if slot.replace(value.clone()).is_some() {
            input_member_error(path, statement.span, "duplicate input member", diagnostics);
        }
    }
    members
}

fn input_member_error(
    path: &LogicalSourcePath,
    span: Span,
    message: &'static str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.input_member",
        message,
        FeaDiagnosticPhase::Semantic,
        Some(FeaDiagnosticLocation {
            path: path.clone(),
            span,
        }),
    ));
}

fn extend_geometry_members(
    scope: &mut ExpressionScope,
    source: &str,
    declaration: &FeaDeclaration,
    exposed_name: &str,
) {
    for statement in &declaration.body.statements {
        let FeaStatementKind::Block { header, .. } = &statement.kind else {
            continue;
        };
        let header = source[header.span.start..header.span.end].trim();
        if let Some(rest) = header.strip_prefix("region ") {
            if let Some((name, role)) = rest.split_once(':') {
                if let Some(role) = topology_role(role.trim()) {
                    scope.insert_reference(
                        [
                            "geometry".to_string(),
                            exposed_name.to_string(),
                            "region".to_string(),
                            name.trim().to_string(),
                        ],
                        FeaType::Region(role),
                    );
                }
            }
        } else if let Some(rest) = header.strip_prefix("cad_parameter ") {
            if let Some((name, value_type)) = rest.split_once(':') {
                if let Ok(value_type) = parse_type_syntax(value_type) {
                    scope.insert_reference(
                        [
                            "geometry".to_string(),
                            exposed_name.to_string(),
                            "parameter".to_string(),
                            name.trim().to_string(),
                        ],
                        value_type,
                    );
                }
            }
        }
    }
}

fn topology_role(source: &str) -> Option<FeaTopologyRole> {
    Some(match source {
        "vertex" => FeaTopologyRole::Vertex,
        "edge" => FeaTopologyRole::Edge,
        "face" => FeaTopologyRole::Face,
        "solid" => FeaTopologyRole::Solid,
        "cell" => FeaTopologyRole::Cell,
        _ => return None,
    })
}

fn split_parameters(source: &str) -> Vec<&str> {
    let mut start = 0usize;
    let mut depth = 0usize;
    let mut parameters = Vec::new();
    for (index, value) in source.char_indices() {
        match value {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parameters.push(source[start..index].trim());
                start = index + value.len_utf8();
            }
            _ => {}
        }
    }
    let final_parameter = source[start..].trim();
    if !final_parameter.is_empty() {
        parameters.push(final_parameter);
    }
    parameters
}

fn declaration_kind(kind: runmat_design_syntax::FeaDeclarationKind) -> FeaSymbolKind {
    match kind {
        runmat_design_syntax::FeaDeclarationKind::Geometry => FeaSymbolKind::Geometry,
        runmat_design_syntax::FeaDeclarationKind::Mesh => FeaSymbolKind::Mesh,
        runmat_design_syntax::FeaDeclarationKind::Material => FeaSymbolKind::Material,
        runmat_design_syntax::FeaDeclarationKind::Model => FeaSymbolKind::Model,
        runmat_design_syntax::FeaDeclarationKind::Study => FeaSymbolKind::Study,
        runmat_design_syntax::FeaDeclarationKind::Sweep => FeaSymbolKind::Sweep,
        runmat_design_syntax::FeaDeclarationKind::Execution => FeaSymbolKind::Execution,
        runmat_design_syntax::FeaDeclarationKind::Table => FeaSymbolKind::Table,
    }
}

pub(super) fn local_declaration_key(
    bound: &BoundFeaProgram,
    path: &LogicalSourcePath,
    kind: runmat_design_syntax::FeaDeclarationKind,
    name: &str,
) -> Option<crate::FeaSymbolKey> {
    let kind = declaration_kind(kind);
    bound.symbols.get(path)?.values().find_map(|symbol| {
        (symbol.key.kind == kind
            && symbol.key.name == name
            && symbol.origin == FeaSymbolOrigin::Local)
            .then(|| symbol.key.clone())
    })
}
