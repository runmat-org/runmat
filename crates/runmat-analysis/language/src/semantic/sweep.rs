use std::collections::BTreeMap;

use runmat_analysis_core::{
    AuthoredCaseOutputRef, AuthoredStudy, AuthoredStudyRef, AuthoredSweep, AuthoredSweepAxis,
    AuthoredSweepAxisValue, AuthoredSweepBinding, AuthoredSweepCase, AuthoredSweepDefinition,
    AuthoredSweepOutput, AuthoredSweepOutputExpression, CaseOutputProjection, SweepAxisCombination,
    SweepExtremum,
};

use crate::{
    BoundFeaProgram, CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration,
    CheckedFeaStatement, CheckedFeaStatementKind, ExpressionScope, FeaDiagnostic,
    FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey, FeaSymbolKind, FeaType,
    LogicalSourcePath,
};
use runmat_design_syntax::FeaDeclaration;

pub(super) fn extend_output_scope(
    scope: &mut ExpressionScope,
    bound: &BoundFeaProgram,
    path: &LogicalSourcePath,
    declaration: &FeaDeclaration,
    known: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
) {
    if declaration.kind != runmat_design_syntax::FeaDeclarationKind::Sweep {
        return;
    }
    let Some(reference) = declaration.source.as_ref() else {
        return;
    };
    let [namespace, name] = reference.segments.as_slice() else {
        return;
    };
    if namespace.text != "study" {
        return;
    }
    let key = FeaSymbolKey {
        kind: FeaSymbolKind::Study,
        name: name.text.clone(),
    };
    let Some(symbol) = bound
        .symbols
        .get(path)
        .and_then(|symbols| symbols.get(&key))
    else {
        return;
    };
    let resolved = match &symbol.origin {
        crate::FeaSymbolOrigin::Local => (path.clone(), key),
        crate::FeaSymbolOrigin::Import {
            target_module,
            target,
        } => (target_module.clone(), target.clone()),
        crate::FeaSymbolOrigin::Function => return,
    };
    let Some(study) = known.get(&resolved) else {
        return;
    };
    for (output, value_type) in &study.outputs {
        scope.insert_reference(
            ["case".to_string(), "output".to_string(), output.clone()],
            value_type.clone(),
        );
        if let FeaType::AttributedExtremum(value) = value_type {
            scope.insert_reference(
                [
                    "case".to_string(),
                    "output".to_string(),
                    output.clone(),
                    "value".to_string(),
                ],
                value.as_ref().clone(),
            );
        }
    }
}

pub fn lower_sweeps(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    studies: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredStudy>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredSweep> {
    declarations
        .iter()
        .filter_map(|(key, declaration)| {
            if key.1.kind != FeaSymbolKind::Sweep {
                return None;
            }
            lower_sweep(bound, declaration, studies, diagnostics).map(|sweep| (key.clone(), sweep))
        })
        .collect()
}

fn lower_sweep(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    studies: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredStudy>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredSweep> {
    let initial_errors = diagnostics.len();
    let source = declaration.source.as_ref().and_then(|path| {
        super::procedure::resolve_symbol_key(bound, declaration, path, FeaSymbolKind::Study)
    });
    let Some(source) = source else {
        error(
            declaration,
            "sweep header must reference an available `study.<name>` after `over`",
            diagnostics,
        );
        return None;
    };
    let Some(study) = studies.get(&source) else {
        error(
            declaration,
            "sweep source is not available as a valid typed study",
            diagnostics,
        );
        return None;
    };
    let axes = axis_blocks(bound, declaration, study, diagnostics);
    let cases = case_blocks(bound, declaration, study, diagnostics);
    let combination = combination(declaration, diagnostics);
    if !axes.is_empty() && !cases.is_empty() {
        error(
            declaration,
            "sweep must use either axes or explicit cases, not both",
            diagnostics,
        );
    }
    let definition = if !axes.is_empty() {
        AuthoredSweepDefinition::Axes {
            combination: combination.unwrap_or(SweepAxisCombination::Product),
            axes,
        }
    } else {
        if combination.is_some() {
            error(
                declaration,
                "combine is valid only for an axis-based sweep",
                diagnostics,
            );
        }
        AuthoredSweepDefinition::Cases { cases }
    };
    let outputs = sweep_outputs(declaration, diagnostics);
    if diagnostics.len() != initial_errors {
        return None;
    }
    let sweep = AuthoredSweep {
        module_path: declaration.module.as_str().to_string(),
        sweep_id: declaration.key.name.clone(),
        study: AuthoredStudyRef {
            module_path: study.module_path.clone(),
            study_id: study.study_id.clone(),
        },
        definition,
        outputs,
    };
    if let Err(reason) = sweep.validate(study) {
        error(declaration, reason.to_string(), diagnostics);
        return None;
    }
    Some(sweep)
}

fn sweep_outputs(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredSweepOutput> {
    declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::Output { name, value } = &statement.kind else {
                return None;
            };
            let CheckedExpressionKind::OperationCall {
                lowering,
                arguments,
                ..
            } = &value.kind
            else {
                error_at(
                    declaration,
                    statement,
                    "sweep outputs must use collect, retain_each, min, or max",
                    diagnostics,
                );
                return None;
            };
            let source = arguments.first().and_then(|argument| match argument {
                crate::CheckedCallArgument::Positional(value)
                | crate::CheckedCallArgument::Named { value, .. } => case_output_ref(value),
            });
            let Some(source) = source else {
                error_at(
                    declaration,
                    statement,
                    "sweep output must reference `case.output.<name>` or its `.value`",
                    diagnostics,
                );
                return None;
            };
            let expression = match lowering.as_str() {
                "fea.output.case.collect" => AuthoredSweepOutputExpression::Collect { source },
                "fea.output.case.retain_each" => {
                    AuthoredSweepOutputExpression::RetainEach { source }
                }
                "fea.output.case.min" => AuthoredSweepOutputExpression::Extremum {
                    reduction: SweepExtremum::Minimum,
                    source,
                },
                "fea.output.case.max" => AuthoredSweepOutputExpression::Extremum {
                    reduction: SweepExtremum::Maximum,
                    source,
                },
                _ => {
                    error_at(
                        declaration,
                        statement,
                        "operation is not valid for a sweep output",
                        diagnostics,
                    );
                    return None;
                }
            };
            Some(AuthoredSweepOutput {
                output_id: name.clone(),
                expression,
            })
        })
        .collect()
}

fn case_output_ref(expression: &CheckedExpression) -> Option<AuthoredCaseOutputRef> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        return None;
    };
    match path.as_slice() {
        [case, output, output_id] if case == "case" && output == "output" => {
            Some(AuthoredCaseOutputRef {
                output_id: output_id.clone(),
                projection: CaseOutputProjection::Whole,
            })
        }
        [case, output, output_id, value]
            if case == "case" && output == "output" && value == "value" =>
        {
            Some(AuthoredCaseOutputRef {
                output_id: output_id.clone(),
                projection: CaseOutputProjection::Value,
            })
        }
        _ => None,
    }
}

fn axis_blocks(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    study: &AuthoredStudy,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredSweepAxis> {
    declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::Block {
                header, statements, ..
            } = &statement.kind
            else {
                return None;
            };
            let input_id = named_block(header, "axis")?;
            let Some(input) = study.inputs.iter().find(|input| input.input_id == input_id) else {
                error_at(
                    declaration,
                    statement,
                    format!("sweep axis references unknown study input `{input_id}`"),
                    diagnostics,
                );
                return None;
            };
            let values = assignments(statements)
                .filter_map(|(value_id, expression)| {
                    super::input::lower_value(
                        bound,
                        declaration,
                        input.value_type,
                        expression,
                        diagnostics,
                    )
                    .map(|value| AuthoredSweepAxisValue { value_id, value })
                })
                .collect();
            Some(AuthoredSweepAxis { input_id, values })
        })
        .collect()
}

fn case_blocks(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    study: &AuthoredStudy,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredSweepCase> {
    declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::Block {
                header, statements, ..
            } = &statement.kind
            else {
                return None;
            };
            let case_id = named_block(header, "case")?;
            let bindings = assignments(statements)
                .filter_map(|(input_id, expression)| {
                    let Some(input) = study.inputs.iter().find(|input| input.input_id == input_id)
                    else {
                        error_at(
                            declaration,
                            statement,
                            format!("sweep case references unknown study input `{input_id}`"),
                            diagnostics,
                        );
                        return None;
                    };
                    super::input::lower_value(
                        bound,
                        declaration,
                        input.value_type,
                        expression,
                        diagnostics,
                    )
                    .map(|value| AuthoredSweepBinding { input_id, value })
                })
                .collect();
            Some(AuthoredSweepCase { case_id, bindings })
        })
        .collect()
}

fn combination(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SweepAxisCombination> {
    let expression = declaration.statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        matches!(target.as_slice(), [name] if name == "combine").then_some(value)
    })?;
    match &expression.kind {
        CheckedExpressionKind::Enum(value) if value == "product" => {
            Some(SweepAxisCombination::Product)
        }
        CheckedExpressionKind::Enum(value) if value == "zip" => Some(SweepAxisCombination::Zip),
        _ => {
            error(
                declaration,
                "sweep combine must be `product` or `zip`",
                diagnostics,
            );
            None
        }
    }
}

fn assignments(
    statements: &[CheckedFeaStatement],
) -> impl Iterator<Item = (String, &CheckedExpression)> {
    statements.iter().filter_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        let [name] = target.as_slice() else {
            return None;
        };
        Some((name.clone(), value))
    })
}

fn named_block(header: &str, expected: &str) -> Option<String> {
    let words = header.split_whitespace().collect::<Vec<_>>();
    match words.as_slice() {
        [kind, name] if *kind == expected => Some((*name).to_string()),
        _ => None,
    }
}

fn error(
    declaration: &CheckedFeaDeclaration,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.sweep",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span: declaration.span,
        }),
    ));
}

fn error_at(
    declaration: &CheckedFeaDeclaration,
    statement: &CheckedFeaStatement,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.sweep",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span: statement.span,
        }),
    ));
}
