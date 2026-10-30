use super::scope::{
    declaration_inputs, extend_declaration_signatures, extend_module_scope, local_declaration_key,
};
use super::{check_expression, CheckedExpression, ExpressionCheck, ExpressionScope, FeaType};
use crate::{
    AnalysisOperationCatalog, BoundFeaProgram, ExecutionProfileSource, FeaDiagnostic,
    FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey, GeometryInspectionSource,
    LogicalSourcePath,
};
use runmat_fea_syntax::{
    FeaBlock, FeaDeclaration, FeaModelDefinitionKind, FeaModuleItem, FeaStatementKind,
};
use runmat_types::Span;
use std::collections::BTreeMap;

mod call_sites;
mod material_source;
mod output;
mod types;

pub use types::{
    CheckedFeaCallSite, CheckedFeaCallable, CheckedFeaProgram, CheckedFeaStatement,
    CheckedFeaStatementKind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedFeaDeclaration {
    pub key: FeaSymbolKey,
    pub module: LogicalSourcePath,
    pub inputs: BTreeMap<String, CheckedFeaInput>,
    pub source: Option<Vec<String>>,
    pub material_source: Option<CheckedCadMaterialSource>,
    pub target: Option<Vec<String>>,
    pub statements: Vec<CheckedFeaStatement>,
    pub outputs: BTreeMap<String, FeaType>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedCadMaterialSource {
    pub geometry: String,
    pub material_id: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedFeaInput {
    pub name: String,
    pub value_type: FeaType,
    pub default: Option<CheckedExpression>,
    pub minimum: Option<CheckedExpression>,
    pub maximum: Option<CheckedExpression>,
    pub required: bool,
    pub span: Span,
}

pub fn check_program<
    C: AnalysisOperationCatalog + GeometryInspectionSource + ExecutionProfileSource + ?Sized,
>(
    catalog: &C,
    bound: BoundFeaProgram,
) -> CheckedFeaProgram {
    let mut declarations = BTreeMap::new();
    let mut diagnostics = Vec::new();
    for checking_sweeps in [false, true] {
        for (path, module) in &bound.graph.modules {
            let Some(syntax) = &module.syntax else {
                continue;
            };
            for item in &syntax.items {
                let FeaModuleItem::Declaration(declaration) = item else {
                    continue;
                };
                if (declaration.kind == runmat_fea_syntax::FeaDeclarationKind::Sweep)
                    != checking_sweeps
                {
                    continue;
                }
                let Some(key) =
                    local_declaration_key(&bound, path, declaration.kind, &declaration.name.text)
                else {
                    continue;
                };
                let checked = check_declaration(
                    DeclarationCheck {
                        catalog,
                        bound: &bound,
                        known: &declarations,
                        path,
                        source: syntax.source(),
                    },
                    declaration,
                    key.clone(),
                    &mut diagnostics,
                );
                declarations.insert((path.clone(), key), checked);
            }
        }
    }
    let call_sites = call_sites::collect(&declarations);
    let products =
        super::lowering::lower_products(catalog, &bound, &mut declarations, &mut diagnostics);
    CheckedFeaProgram {
        bound,
        declarations,
        call_sites,
        diagnostics,
        geometry_inspections: products.geometry_inspections,
        studies: products.studies,
        sweeps: products.sweeps,
        execution_policies: products.execution_policies,
        execution_bindings: products.execution_bindings,
        materials: products.materials,
        geometries: products.geometries,
        meshes: products.meshes,
        models: products.models,
    }
}

struct DeclarationCheck<'a, C: AnalysisOperationCatalog + ?Sized> {
    catalog: &'a C,
    bound: &'a BoundFeaProgram,
    known: &'a BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    path: &'a LogicalSourcePath,
    source: &'a str,
}

fn check_declaration<C: AnalysisOperationCatalog + ?Sized>(
    check: DeclarationCheck<'_, C>,
    declaration: &FeaDeclaration,
    key: FeaSymbolKey,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> CheckedFeaDeclaration {
    let DeclarationCheck {
        catalog,
        bound,
        known,
        path,
        source,
    } = check;
    let mut scope = ExpressionScope::new();
    extend_module_scope(&mut scope, bound, path);
    extend_declaration_signatures(&mut scope, bound, path);
    super::sweep::extend_output_scope(&mut scope, bound, path, declaration, known);
    let source_reference = declaration.source.as_ref().map(|reference| {
        reference
            .segments
            .iter()
            .map(|segment| segment.text.clone())
            .collect::<Vec<_>>()
    });
    let target_reference = declaration.target.as_ref().map(|reference| {
        reference
            .segments
            .iter()
            .map(|segment| segment.text.clone())
            .collect::<Vec<_>>()
    });
    let material_source = declaration
        .material_source
        .as_ref()
        .and_then(|expression| material_source::check(expression, path, diagnostics));
    if key.kind == crate::FeaSymbolKind::Geometry && source_reference.is_some() {
        scope.insert_reference(["source".to_string()], FeaType::GeometryFeature);
        scope.insert_reference(
            ["source".to_string(), "result".to_string()],
            FeaType::GeometryFeature,
        );
    }
    let input_syntax = declaration_inputs(source, declaration, diagnostics, path);
    for input in &input_syntax {
        scope.insert_reference(
            ["input".to_string(), input.name.clone()],
            input.value_type.clone(),
        );
    }
    let mut inputs = BTreeMap::new();
    for input in input_syntax {
        let input_check = InputCheck {
            catalog,
            path,
            source,
            scope: &scope,
        };
        let default = input_check.check(&input, input.default.as_ref(), "default", diagnostics);
        let minimum = input_check.check(&input, input.minimum.as_ref(), "minimum", diagnostics);
        let maximum = input_check.check(&input, input.maximum.as_ref(), "maximum", diagnostics);
        let checked = CheckedFeaInput {
            name: input.name.clone(),
            value_type: input.value_type,
            required: default.is_none(),
            default,
            minimum,
            maximum,
            span: input.span,
        };
        if let Some(existing) = inputs.insert(input.name.clone(), checked) {
            diagnostics.push(FeaDiagnostic::error(
                "fea.semantic.duplicate_input",
                format!("duplicate input `{}`", input.name),
                FeaDiagnosticPhase::Semantic,
                Some(FeaDiagnosticLocation {
                    path: path.clone(),
                    span: existing.span,
                }),
            ));
        }
    }
    let mut outputs = BTreeMap::new();
    let statements = check_block(
        catalog,
        path,
        source,
        &declaration.body,
        &mut scope,
        &mut outputs,
        diagnostics,
    );
    CheckedFeaDeclaration {
        key,
        module: path.clone(),
        inputs,
        source: source_reference,
        material_source,
        target: target_reference,
        statements,
        outputs,
        span: declaration.span,
    }
}

struct InputCheck<'a, C: AnalysisOperationCatalog + ?Sized> {
    catalog: &'a C,
    path: &'a LogicalSourcePath,
    source: &'a str,
    scope: &'a ExpressionScope,
}

impl<C: AnalysisOperationCatalog + ?Sized> InputCheck<'_, C> {
    fn check(
        &self,
        input: &super::scope::InputSyntax,
        expression: Option<&runmat_fea_syntax::FeaExpression>,
        member: &'static str,
        diagnostics: &mut Vec<FeaDiagnostic>,
    ) -> Option<CheckedExpression> {
        let expression = expression?;
        let (checked, mut expression_diagnostics) = check_expression(
            ExpressionCheck {
                path: self.path,
                source: self.source,
                scope: self.scope,
                catalog: self.catalog,
            },
            expression,
        );
        diagnostics.append(&mut expression_diagnostics);
        if !input.value_type.accepts(&checked.value_type) {
            diagnostics.push(FeaDiagnostic::error(
                "fea.semantic.input_value_type",
                format!(
                    "{member} for input `{}` has an incompatible type",
                    input.name
                ),
                FeaDiagnosticPhase::Semantic,
                Some(FeaDiagnosticLocation {
                    path: self.path.clone(),
                    span: expression.span,
                }),
            ));
        }
        Some(checked)
    }
}

fn check_block<C: AnalysisOperationCatalog + ?Sized>(
    catalog: &C,
    path: &LogicalSourcePath,
    source: &str,
    block: &FeaBlock,
    scope: &mut ExpressionScope,
    outputs: &mut BTreeMap<String, FeaType>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<CheckedFeaStatement> {
    for statement in &block.statements {
        if let FeaStatementKind::SignalDefinition { name, .. } = &statement.kind {
            scope.insert_reference(["signal".to_string(), name.text.clone()], FeaType::Signal);
            continue;
        }
        let FeaStatementKind::ModelDefinition { kind, name, .. } = &statement.kind else {
            continue;
        };
        let (namespace, value_type) = match kind {
            FeaModelDefinitionKind::Section => ("section", FeaType::Section),
            FeaModelDefinitionKind::Layup => ("layup", FeaType::Layup),
            FeaModelDefinitionKind::Orientation => ("orientation", FeaType::Orientation),
            FeaModelDefinitionKind::Constraint => ("constraint", FeaType::Constraint),
            FeaModelDefinitionKind::Load => ("load", FeaType::Load),
            FeaModelDefinitionKind::LoadCase => ("load_case", FeaType::LoadCase),
            FeaModelDefinitionKind::Interface => ("interface", FeaType::Interface),
            FeaModelDefinitionKind::Connector => ("connector", FeaType::Connector),
            FeaModelDefinitionKind::Excitation => ("excitation", FeaType::Excitation),
            FeaModelDefinitionKind::ResponseSpectrum => {
                ("response_spectrum", FeaType::ResponseSpectrum)
            }
            FeaModelDefinitionKind::RandomExcitation => {
                ("random_excitation", FeaType::RandomExcitation)
            }
            FeaModelDefinitionKind::Initial => ("initial", FeaType::InitialCondition),
        };
        scope.insert_reference([namespace.to_string(), name.text.clone()], value_type);
    }
    block
        .statements
        .iter()
        .map(|statement| {
            let kind = match &statement.kind {
                FeaStatementKind::GeneratedByFunction { function, body } => {
                    CheckedFeaStatementKind::GeneratedByFunction {
                        function: function
                            .segments
                            .iter()
                            .map(|segment| segment.text.clone())
                            .collect(),
                        statements: check_block(
                            catalog,
                            path,
                            source,
                            body,
                            scope,
                            outputs,
                            diagnostics,
                        ),
                    }
                }
                FeaStatementKind::UniformSample { start, end, rate } => {
                    CheckedFeaStatementKind::UniformSample {
                        start: check_value(catalog, path, source, scope, start, diagnostics),
                        end: check_value(catalog, path, source, scope, end, diagnostics),
                        rate: Box::new(check_value(
                            catalog,
                            path,
                            source,
                            scope,
                            rate,
                            diagnostics,
                        )),
                    }
                }
                FeaStatementKind::AppliedLoad {
                    kind,
                    target,
                    value,
                } => CheckedFeaStatementKind::AppliedLoad {
                    kind: kind.text.clone(),
                    target: check_value(catalog, path, source, scope, target, diagnostics),
                    value: check_value(catalog, path, source, scope, value, diagnostics),
                },
                FeaStatementKind::SignalDefinition {
                    name,
                    quantity,
                    domain,
                    body,
                } => CheckedFeaStatementKind::SignalDefinition {
                    name: name.text.clone(),
                    quantity: quantity.text.clone(),
                    domain: domain.text.clone(),
                    statements: check_block(
                        catalog,
                        path,
                        source,
                        body,
                        scope,
                        outputs,
                        diagnostics,
                    ),
                },
                FeaStatementKind::ModelDefinition { kind, name, body } => {
                    CheckedFeaStatementKind::ModelDefinition {
                        kind: *kind,
                        name: name.text.clone(),
                        statements: check_block(
                            catalog,
                            path,
                            source,
                            body,
                            scope,
                            outputs,
                            diagnostics,
                        ),
                    }
                }
                FeaStatementKind::Block { header, body } => {
                    let text = &source[header.span.start..header.span.end];
                    let parts = text.split_whitespace().collect::<Vec<_>>();
                    let statements =
                        check_block(catalog, path, source, body, scope, outputs, diagnostics);
                    if parts.len() == 2
                        && crate::FeaContextualWord::from_source(parts[0])
                            == Some(crate::FeaContextualWord::Assessment)
                    {
                        CheckedFeaStatementKind::AssessmentDefinition {
                            name: parts[1].to_string(),
                            statements,
                        }
                    } else {
                        CheckedFeaStatementKind::Block {
                            header: text.to_string(),
                            header_span: header.span,
                            statements,
                        }
                    }
                }
                FeaStatementKind::Input { name, .. } => {
                    CheckedFeaStatementKind::Input(name.text.clone())
                }
                FeaStatementKind::Let { name, value } => {
                    let value = check_value(catalog, path, source, scope, value, diagnostics);
                    scope.insert_reference([name.text.clone()], value.value_type.clone());
                    CheckedFeaStatementKind::Let {
                        name: name.text.clone(),
                        value,
                    }
                }
                FeaStatementKind::Feature { name, value } => {
                    let value = check_value(catalog, path, source, scope, value, diagnostics);
                    if value.value_type != FeaType::GeometryFeature
                        && value.value_type != FeaType::Error
                    {
                        diagnostics.push(FeaDiagnostic::error(
                            "fea.semantic.feature_type",
                            format!(
                                "feature `{}` requires a geometry feature operation",
                                name.text
                            ),
                            FeaDiagnosticPhase::Semantic,
                            Some(FeaDiagnosticLocation {
                                path: path.clone(),
                                span: value.span,
                            }),
                        ));
                    }
                    scope.insert_reference(
                        ["feature".to_string(), name.text.clone()],
                        FeaType::GeometryFeature,
                    );
                    CheckedFeaStatementKind::Feature {
                        name: name.text.clone(),
                        value,
                    }
                }
                FeaStatementKind::SourceParameterUpdate { target, value } => {
                    CheckedFeaStatementKind::SourceParameterUpdate {
                        target: target
                            .segments
                            .iter()
                            .map(|segment| segment.text.clone())
                            .collect(),
                        value: check_value(catalog, path, source, scope, value, diagnostics),
                    }
                }
                FeaStatementKind::Use(property) => CheckedFeaStatementKind::Use(
                    property
                        .segments
                        .iter()
                        .map(|segment| segment.text.clone())
                        .collect(),
                ),
                FeaStatementKind::Requirement { target, value } => {
                    CheckedFeaStatementKind::Requirement {
                        target: target
                            .segments
                            .iter()
                            .map(|segment| segment.text.clone())
                            .collect(),
                        value: check_value(catalog, path, source, scope, value, diagnostics),
                    }
                }
                FeaStatementKind::Assignment {
                    target,
                    relation,
                    value,
                } => CheckedFeaStatementKind::Assignment {
                    target: target
                        .segments
                        .iter()
                        .map(|segment| segment.text.clone())
                        .collect(),
                    relation: *relation,
                    value: check_value(catalog, path, source, scope, value, diagnostics),
                },
                FeaStatementKind::PropertyAssignment { property, target } => {
                    CheckedFeaStatementKind::PropertyAssignment {
                        property: check_value(catalog, path, source, scope, property, diagnostics),
                        target: check_value(catalog, path, source, scope, target, diagnostics),
                    }
                }
                FeaStatementKind::Output { name, value } => {
                    let value = check_value(catalog, path, source, scope, value, diagnostics);
                    output::register(
                        &name.text,
                        name.span,
                        value.value_type.clone(),
                        path,
                        scope,
                        outputs,
                        diagnostics,
                    );
                    CheckedFeaStatementKind::Output {
                        name: name.text.clone(),
                        value,
                    }
                }
                FeaStatementKind::OutputDefinition { name, body } => {
                    let mut definition_scope = scope.clone();
                    let mut definition_outputs = outputs.clone();
                    let statements = check_block(
                        catalog,
                        path,
                        source,
                        body,
                        &mut definition_scope,
                        &mut definition_outputs,
                        diagnostics,
                    );
                    let value_type =
                        output::definition_value(&statements, path, statement.span, diagnostics)
                            .map_or(FeaType::Error, |value| value.value_type.clone());
                    output::register(
                        &name.text,
                        name.span,
                        value_type,
                        path,
                        scope,
                        outputs,
                        diagnostics,
                    );
                    CheckedFeaStatementKind::OutputDefinition {
                        name: name.text.clone(),
                        statements,
                    }
                }
                FeaStatementKind::Include(value) => {
                    // `include model.m(name = value, ...)`: the model reference checks as usual;
                    // binding its inputs by name is a planned feature.
                    let value = match &value.kind {
                        runmat_fea_syntax::FeaExpressionKind::Call { callee, .. } => {
                            diagnostics.push(FeaDiagnostic::error(
                                "fea.semantic.include_arguments",
                                super::pending::with_pending(
                                    "an include binds no inputs yet; include the model by reference \
                                     and set its inputs inside it",
                                    Some(super::pending::INCLUDE_PARAMETERIZED),
                                ),
                                FeaDiagnosticPhase::Semantic,
                                Some(FeaDiagnosticLocation {
                                    path: path.clone(),
                                    span: value.span,
                                }),
                            ));
                            callee.as_ref()
                        }
                        _ => value,
                    };
                    CheckedFeaStatementKind::Include(check_value(
                        catalog,
                        path,
                        source,
                        scope,
                        value,
                        diagnostics,
                    ))
                }
                FeaStatementKind::Retain(value) => CheckedFeaStatementKind::Retain(check_value(
                    catalog,
                    path,
                    source,
                    scope,
                    value,
                    diagnostics,
                )),
                FeaStatementKind::Command { syntax } => CheckedFeaStatementKind::Command(
                    source[syntax.span.start..syntax.span.end].to_string(),
                ),
            };
            CheckedFeaStatement {
                kind,
                span: statement.span,
            }
        })
        .collect()
}

fn check_value<C: AnalysisOperationCatalog + ?Sized>(
    catalog: &C,
    path: &LogicalSourcePath,
    source: &str,
    scope: &ExpressionScope,
    value: &runmat_fea_syntax::FeaExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> CheckedExpression {
    let (checked, mut found) = check_expression(
        ExpressionCheck {
            path,
            source,
            scope,
            catalog,
        },
        value,
    );
    diagnostics.append(&mut found);
    checked
}
