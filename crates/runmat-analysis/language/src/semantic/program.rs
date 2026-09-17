use super::scope::{
    declaration_inputs, extend_declaration_signatures, extend_module_scope, local_declaration_key,
};
use super::{check_expression, CheckedExpression, ExpressionCheck, ExpressionScope, FeaType};
use crate::{
    AnalysisOperationCatalog, BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation,
    FeaDiagnosticPhase, FeaSymbolKey, GeometryInspectionSource, LogicalSourcePath,
};
use runmat_parser::fea::{
    FeaAssignmentRelation, FeaBlock, FeaDeclaration, FeaModelDefinitionKind, FeaModuleItem,
    FeaStatementKind,
};
use runmat_types::Span;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct CheckedFeaProgram {
    pub bound: BoundFeaProgram,
    pub declarations: BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub geometry_inspections:
        BTreeMap<(LogicalSourcePath, String), super::CheckedGeometryInspection>,
    pub analysis_sequences:
        BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AnalysisSequence>,
    pub materials: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::MaterialModel>,
    pub geometries:
        BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_geometry_core::GeometryDefinition>,
    pub meshes: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_meshing_core::MeshDefinition>,
    pub models: BTreeMap<(LogicalSourcePath, FeaSymbolKey), runmat_analysis_core::AuthoredModel>,
}

impl CheckedFeaProgram {
    pub fn is_valid(&self) -> bool {
        self.bound.is_valid()
            && !self.diagnostics.iter().any(|diagnostic| {
                diagnostic.severity == crate::FeaDiagnosticSeverity::Error
                    && diagnostic.availability == crate::SourceAvailability::Available
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedFeaDeclaration {
    pub key: FeaSymbolKey,
    pub module: LogicalSourcePath,
    pub inputs: BTreeMap<String, CheckedFeaInput>,
    pub source: Option<Vec<String>>,
    pub target: Option<Vec<String>>,
    pub statements: Vec<CheckedFeaStatement>,
    pub outputs: BTreeMap<String, FeaType>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedFeaInput {
    pub name: String,
    pub value_type: FeaType,
    pub default: Option<CheckedExpression>,
    pub required: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedFeaStatement {
    pub kind: CheckedFeaStatementKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckedFeaStatementKind {
    GeneratedByFunction {
        function: Vec<String>,
        statements: Vec<CheckedFeaStatement>,
    },
    UniformSample {
        start: CheckedExpression,
        end: CheckedExpression,
        rate: CheckedExpression,
    },
    SignalDefinition {
        name: String,
        quantity: String,
        domain: String,
        statements: Vec<CheckedFeaStatement>,
    },
    ModelDefinition {
        kind: FeaModelDefinitionKind,
        name: String,
        statements: Vec<CheckedFeaStatement>,
    },
    Block {
        header: String,
        header_span: Span,
        statements: Vec<CheckedFeaStatement>,
    },
    Input(String),
    Let {
        name: String,
        value: CheckedExpression,
    },
    Feature {
        name: String,
        value: CheckedExpression,
    },
    Assignment {
        target: Vec<String>,
        relation: FeaAssignmentRelation,
        value: CheckedExpression,
    },
    PropertyAssignment {
        property: CheckedExpression,
        target: CheckedExpression,
    },
    Output {
        name: String,
        value: CheckedExpression,
    },
    Include(CheckedExpression),
    Retain(CheckedExpression),
    Command(String),
}

pub fn check_program<C: AnalysisOperationCatalog + GeometryInspectionSource + ?Sized>(
    catalog: &C,
    bound: BoundFeaProgram,
) -> CheckedFeaProgram {
    let mut declarations = BTreeMap::new();
    let mut diagnostics = Vec::new();
    for (path, module) in &bound.graph.modules {
        let Some(syntax) = &module.syntax else {
            continue;
        };
        for item in &syntax.items {
            let FeaModuleItem::Declaration(declaration) = item else {
                continue;
            };
            let Some(key) = local_declaration_key(&bound, path, &declaration.name.text) else {
                continue;
            };
            let checked = check_declaration(
                catalog,
                &bound,
                path,
                syntax.source(),
                declaration,
                key.clone(),
                &mut diagnostics,
            );
            declarations.insert((path.clone(), key), checked);
        }
    }
    let geometry_inspections =
        super::inspect_geometry_declarations(catalog, &bound, &mut diagnostics);
    let analysis_sequences = super::lower_analysis_sequences(&declarations, &mut diagnostics);
    let materials = super::lower_materials(&declarations, &mut diagnostics);
    let geometries = super::lower_geometry_definitions(
        &bound,
        &declarations,
        &geometry_inspections,
        &mut diagnostics,
    );
    let meshes = super::lower_mesh_definitions(&bound, &declarations, &mut diagnostics);
    let models = super::lower_models(&bound, &declarations, &mut diagnostics);
    CheckedFeaProgram {
        bound,
        declarations,
        diagnostics,
        geometry_inspections,
        analysis_sequences,
        materials,
        geometries,
        meshes,
        models,
    }
}

fn check_declaration<C: AnalysisOperationCatalog + ?Sized>(
    catalog: &C,
    bound: &BoundFeaProgram,
    path: &LogicalSourcePath,
    source: &str,
    declaration: &FeaDeclaration,
    key: FeaSymbolKey,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> CheckedFeaDeclaration {
    let mut scope = ExpressionScope::new();
    extend_module_scope(&mut scope, bound, path);
    extend_declaration_signatures(&mut scope, bound, path);
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
    if source_reference.is_some() {
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
        let default = input.default.as_ref().map(|expression| {
            let (checked, mut expression_diagnostics) = check_expression(
                ExpressionCheck {
                    path,
                    source,
                    scope: &scope,
                    catalog,
                },
                expression,
            );
            diagnostics.append(&mut expression_diagnostics);
            if !input.value_type.accepts(&checked.value_type) {
                diagnostics.push(FeaDiagnostic::error(
                    "fea.semantic.input_default_type",
                    format!(
                        "default for input `{}` has an incompatible type",
                        input.name
                    ),
                    FeaDiagnosticPhase::Semantic,
                    Some(FeaDiagnosticLocation {
                        path: path.clone(),
                        span: expression.span,
                    }),
                ));
            }
            checked
        });
        let checked = CheckedFeaInput {
            name: input.name.clone(),
            value_type: input.value_type,
            required: default.is_none(),
            default,
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
        target: target_reference,
        statements,
        outputs,
        span: declaration.span,
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
            FeaModelDefinitionKind::Initial => ("initial", FeaType::InitialCondition),
        };
        scope.insert_reference([namespace.to_string(), name.text.clone()], value_type);
    }
    block
        .statements
        .iter()
        .map(|statement| {
            let kind =
                match &statement.kind {
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
                            rate: check_value(catalog, path, source, scope, rate, diagnostics),
                        }
                    }
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
                    FeaStatementKind::Block { header, body } => CheckedFeaStatementKind::Block {
                        header: source[header.span.start..header.span.end].to_string(),
                        header_span: header.span,
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
                            property: check_value(
                                catalog,
                                path,
                                source,
                                scope,
                                property,
                                diagnostics,
                            ),
                            target: check_value(catalog, path, source, scope, target, diagnostics),
                        }
                    }
                    FeaStatementKind::Output { name, value } => {
                        let value = check_value(catalog, path, source, scope, value, diagnostics);
                        if outputs
                            .insert(name.text.clone(), value.value_type.clone())
                            .is_some()
                        {
                            diagnostics.push(FeaDiagnostic::error(
                                "fea.semantic.duplicate_output",
                                format!("duplicate output `{}`", name.text),
                                FeaDiagnosticPhase::Semantic,
                                Some(FeaDiagnosticLocation {
                                    path: path.clone(),
                                    span: name.span,
                                }),
                            ));
                        }
                        scope.insert_reference(
                            ["output".to_string(), name.text.clone()],
                            value.value_type.clone(),
                        );
                        CheckedFeaStatementKind::Output {
                            name: name.text.clone(),
                            value,
                        }
                    }
                    FeaStatementKind::Include(value) => CheckedFeaStatementKind::Include(
                        check_value(catalog, path, source, scope, value, diagnostics),
                    ),
                    FeaStatementKind::Retain(value) => CheckedFeaStatementKind::Retain(
                        check_value(catalog, path, source, scope, value, diagnostics),
                    ),
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
    value: &runmat_parser::fea::FeaExpression,
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
