use std::collections::BTreeMap;

use crate::{
    CheckedCallArgument, CheckedExpression, CheckedExpressionKind, CheckedFeaCallSite,
    CheckedFeaCallable, CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind,
    FeaSymbolKey, LogicalSourcePath,
};

pub(super) fn collect(
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
) -> Vec<CheckedFeaCallSite> {
    let mut sites = Vec::new();
    for ((module, _), declaration) in declarations {
        for input in declaration.inputs.values() {
            for value in [&input.default, &input.minimum, &input.maximum]
                .into_iter()
                .flatten()
            {
                expression(module, value, &mut sites);
            }
        }
        statements(module, &declaration.statements, &mut sites);
    }
    sites.sort_by(|left, right| {
        (&left.module, left.span.start, left.span.end).cmp(&(
            &right.module,
            right.span.start,
            right.span.end,
        ))
    });
    sites
}

fn statements(
    module: &LogicalSourcePath,
    values: &[CheckedFeaStatement],
    sites: &mut Vec<CheckedFeaCallSite>,
) {
    for statement in values {
        match &statement.kind {
            CheckedFeaStatementKind::UniformSample { start, end, rate } => {
                expression(module, start, sites);
                expression(module, end, sites);
                expression(module, rate, sites);
            }
            CheckedFeaStatementKind::AppliedLoad { target, value, .. } => {
                expression(module, target, sites);
                expression(module, value, sites);
            }
            CheckedFeaStatementKind::GeneratedByFunction {
                statements: body, ..
            }
            | CheckedFeaStatementKind::SignalDefinition {
                statements: body, ..
            }
            | CheckedFeaStatementKind::ModelDefinition {
                statements: body, ..
            }
            | CheckedFeaStatementKind::AssessmentDefinition {
                statements: body, ..
            }
            | CheckedFeaStatementKind::Block {
                statements: body, ..
            }
            | CheckedFeaStatementKind::OutputDefinition {
                statements: body, ..
            } => statements(module, body, sites),
            CheckedFeaStatementKind::Let { value, .. }
            | CheckedFeaStatementKind::Feature { value, .. }
            | CheckedFeaStatementKind::SourceParameterUpdate { value, .. }
            | CheckedFeaStatementKind::Requirement { value, .. }
            | CheckedFeaStatementKind::Assignment { value, .. }
            | CheckedFeaStatementKind::Output { value, .. }
            | CheckedFeaStatementKind::Include(value)
            | CheckedFeaStatementKind::Retain(value) => expression(module, value, sites),
            CheckedFeaStatementKind::PropertyAssignment { property, target } => {
                expression(module, property, sites);
                expression(module, target, sites);
            }
            CheckedFeaStatementKind::Input(_)
            | CheckedFeaStatementKind::Use(_)
            | CheckedFeaStatementKind::Command(_) => {}
        }
    }
}

fn expression(
    module: &LogicalSourcePath,
    value: &CheckedExpression,
    sites: &mut Vec<CheckedFeaCallSite>,
) {
    match &value.kind {
        CheckedExpressionKind::OperationCall {
            operation,
            overload,
            signature,
            arguments,
            ..
        } => {
            sites.push(CheckedFeaCallSite {
                module: module.clone(),
                span: value.span,
                callable: CheckedFeaCallable::Operation {
                    operation: operation.clone(),
                    overload: overload.clone(),
                    signature: signature.clone(),
                },
                named_arguments: named_arguments(arguments),
            });
            call_arguments(module, arguments, sites);
        }
        CheckedExpressionKind::FunctionCall {
            function,
            signature,
            arguments,
        } => {
            sites.push(CheckedFeaCallSite {
                module: module.clone(),
                span: value.span,
                callable: CheckedFeaCallable::Named {
                    path: function.clone(),
                    signature: signature.clone(),
                },
                named_arguments: named_arguments(arguments),
            });
            call_arguments(module, arguments, sites);
        }
        CheckedExpressionKind::List(values) => {
            for value in values {
                expression(module, value, sites);
            }
        }
        CheckedExpressionKind::Record(values) => {
            for value in values.values() {
                expression(module, value, sites);
            }
        }
        CheckedExpressionKind::Unary { operand, .. }
        | CheckedExpressionKind::Quantity { value: operand, .. } => {
            expression(module, operand, sites);
        }
        CheckedExpressionKind::Binary { left, right, .. } => {
            expression(module, left, sites);
            expression(module, right, sites);
        }
        CheckedExpressionKind::Number(_)
        | CheckedExpressionKind::String(_)
        | CheckedExpressionKind::Boolean(_)
        | CheckedExpressionKind::Enum(_)
        | CheckedExpressionKind::Reference(_)
        | CheckedExpressionKind::CadParameterValue(_)
        | CheckedExpressionKind::Missing => {}
    }
}

fn named_arguments(arguments: &[CheckedCallArgument]) -> Vec<(String, runmat_types::Span)> {
    arguments
        .iter()
        .filter_map(|argument| match argument {
            CheckedCallArgument::Named {
                name, name_span, ..
            } => Some((name.clone(), *name_span)),
            CheckedCallArgument::Positional(_) => None,
        })
        .collect()
}

fn call_arguments(
    module: &LogicalSourcePath,
    arguments: &[CheckedCallArgument],
    sites: &mut Vec<CheckedFeaCallSite>,
) {
    for argument in arguments {
        let value = match argument {
            CheckedCallArgument::Positional(value) | CheckedCallArgument::Named { value, .. } => {
                value
            }
        };
        expression(module, value, sites);
    }
}
