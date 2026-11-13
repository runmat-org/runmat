use super::{error, named_block, value};
use crate::{
    BoundFeaProgram, CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind,
    FeaDiagnostic,
};
use runmat_geometry_core::{
    GeneratedGeometryArgument, GeneratedGeometryDefinition, GeometryDefinitionSource,
    GeometryDigest,
};
use std::collections::BTreeMap;

pub(super) fn block(statements: &[CheckedFeaStatement]) -> Option<(&str, &[CheckedFeaStatement])> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::GeneratedByFunction {
            function,
            statements,
        } = &statement.kind
        else {
            return None;
        };
        function
            .as_slice()
            .strip_prefix(&["function".to_string()])
            .and_then(|tail| tail.first())
            .map(|function| (function.as_str(), statements.as_slice()))
    })
}

pub(super) fn source(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    (function, statements): (&str, &[CheckedFeaStatement]),
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryDefinitionSource> {
    let Some(facts) = bound
        .functions
        .get(&(declaration.module.clone(), function.to_string()))
    else {
        error(
            declaration,
            declaration.span,
            format!("generated geometry references unknown imported function `{function}`"),
            diagnostics,
        );
        return None;
    };
    if facts.outputs.as_slice()
        != [runmat_types::OperationTypePattern::Analysis(
            runmat_types::AnalysisValueType::Geometry,
        )]
    {
        error(
            declaration,
            declaration.span,
            "geometry generator must return exactly one typed geometry value",
            diagnostics,
        );
        return None;
    }
    if !facts.effects.is_deterministic_value_function() {
        error(
            declaration,
            declaration.span,
            "geometry generator must be pure; effectful functions are not admitted",
            diagnostics,
        );
        return None;
    }
    let Some(arguments) = named_block(statements, "arguments") else {
        error(
            declaration,
            declaration.span,
            "generated geometry requires an `arguments` block",
            diagnostics,
        );
        return None;
    };
    let mut lowered = BTreeMap::new();
    let mut checked_arguments = BTreeMap::new();
    for statement in arguments {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            error(
                declaration,
                statement.span,
                "generator arguments must be named assignments",
                diagnostics,
            );
            continue;
        };
        let [name] = target.as_slice() else {
            error(
                declaration,
                statement.span,
                "generator argument names cannot be qualified",
                diagnostics,
            );
            continue;
        };
        let Some(value) = value::authoring_value(value) else {
            error(
                declaration,
                statement.span,
                format!("generator argument `{name}` is not a canonical geometry value"),
                diagnostics,
            );
            continue;
        };
        checked_arguments.insert(name.clone(), value_type(statement));
        if lowered.insert(name.clone(), value).is_some() {
            error(
                declaration,
                statement.span,
                format!("duplicate generator argument `{name}`"),
                diagnostics,
            );
        }
    }
    validate_signature(declaration, facts, &checked_arguments, diagnostics);
    let lowered = facts
        .inputs
        .iter()
        .filter_map(|parameter| {
            lowered
                .remove(&parameter.name)
                .map(|value| GeneratedGeometryArgument {
                    parameter: parameter.name.clone(),
                    value,
                })
        })
        .collect();
    Some(GeometryDefinitionSource::Generated(
        GeneratedGeometryDefinition {
            module_path: facts.module_path.as_str().to_string(),
            function: facts.symbol.as_str().to_string(),
            source_digest: GeometryDigest::from_bytes(facts.source_digest.0),
            arguments: lowered,
        },
    ))
}

fn validate_signature(
    declaration: &CheckedFeaDeclaration,
    facts: &crate::FunctionFacts,
    arguments: &BTreeMap<String, crate::FeaType>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    for parameter in &facts.inputs {
        let supplied = arguments.get(&parameter.name);
        if parameter.arity == runmat_types::OperationParameterArity::Required && supplied.is_none()
        {
            error(
                declaration,
                declaration.span,
                format!("geometry generator requires argument `{}`", parameter.name),
                diagnostics,
            );
        }
        if supplied.is_some_and(|value_type| {
            !super::super::expression::operation_pattern_matches(&parameter.value, value_type)
        }) {
            error(
                declaration,
                declaration.span,
                format!(
                    "geometry generator argument `{}` has an incompatible type",
                    parameter.name
                ),
                diagnostics,
            );
        }
    }
    for name in arguments.keys() {
        if !facts.inputs.iter().any(|parameter| &parameter.name == name) {
            error(
                declaration,
                declaration.span,
                format!("geometry generator does not declare argument `{name}`"),
                diagnostics,
            );
        }
    }
}

fn value_type(statement: &CheckedFeaStatement) -> crate::FeaType {
    match &statement.kind {
        CheckedFeaStatementKind::Assignment { value, .. } => value.value_type.clone(),
        _ => crate::FeaType::Error,
    }
}
