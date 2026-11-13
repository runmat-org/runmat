use std::collections::BTreeMap;

use runmat_geometry_core::GeometryDefinition;

use super::{
    CheckedCallArgument, CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration,
    CheckedFeaStatement, CheckedFeaStatementKind, ResolvedCadParameterValue,
};
use crate::{BoundFeaProgram, FeaSymbolKey, FeaSymbolKind, FeaSymbolOrigin, LogicalSourcePath};

pub fn resolve(
    bound: &BoundFeaProgram,
    declarations: &mut BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    geometries: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition>,
) {
    for declaration in declarations.values_mut() {
        for input in declaration.inputs.values_mut() {
            for value in [
                input.default.as_mut(),
                input.minimum.as_mut(),
                input.maximum.as_mut(),
            ]
            .into_iter()
            .flatten()
            {
                resolve_expression(bound, &declaration.module, geometries, value);
            }
        }
        resolve_statements(
            bound,
            &declaration.module,
            geometries,
            &mut declaration.statements,
        );
    }
}

fn resolve_statements(
    bound: &BoundFeaProgram,
    module: &LogicalSourcePath,
    geometries: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition>,
    statements: &mut [CheckedFeaStatement],
) {
    for statement in statements {
        match &mut statement.kind {
            CheckedFeaStatementKind::GeneratedByFunction { statements, .. }
            | CheckedFeaStatementKind::SignalDefinition { statements, .. }
            | CheckedFeaStatementKind::ModelDefinition { statements, .. }
            | CheckedFeaStatementKind::AssessmentDefinition { statements, .. }
            | CheckedFeaStatementKind::Block { statements, .. }
            | CheckedFeaStatementKind::OutputDefinition { statements, .. } => {
                resolve_statements(bound, module, geometries, statements)
            }
            CheckedFeaStatementKind::UniformSample { start, end, rate } => {
                resolve_expression(bound, module, geometries, start);
                resolve_expression(bound, module, geometries, end);
                resolve_expression(bound, module, geometries, rate);
            }
            CheckedFeaStatementKind::AppliedLoad { target, value, .. }
            | CheckedFeaStatementKind::PropertyAssignment {
                property: target,
                target: value,
            } => {
                resolve_expression(bound, module, geometries, target);
                resolve_expression(bound, module, geometries, value);
            }
            CheckedFeaStatementKind::Let { value, .. }
            | CheckedFeaStatementKind::Feature { value, .. }
            | CheckedFeaStatementKind::SourceParameterUpdate { value, .. }
            | CheckedFeaStatementKind::Requirement { value, .. }
            | CheckedFeaStatementKind::Assignment { value, .. }
            | CheckedFeaStatementKind::Output { value, .. }
            | CheckedFeaStatementKind::Include(value)
            | CheckedFeaStatementKind::Retain(value) => {
                resolve_expression(bound, module, geometries, value)
            }
            CheckedFeaStatementKind::Input(_)
            | CheckedFeaStatementKind::Use(_)
            | CheckedFeaStatementKind::Command(_) => {}
        }
    }
}

fn resolve_expression(
    bound: &BoundFeaProgram,
    module: &LogicalSourcePath,
    geometries: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition>,
    expression: &mut CheckedExpression,
) {
    if let CheckedExpressionKind::Reference(path) = &expression.kind {
        if let Some(value) = resolve_reference(bound, module, geometries, path) {
            expression.kind = CheckedExpressionKind::CadParameterValue(value);
            return;
        }
    }
    match &mut expression.kind {
        CheckedExpressionKind::List(values) => {
            for value in values {
                resolve_expression(bound, module, geometries, value);
            }
        }
        CheckedExpressionKind::Record(values) => {
            for value in values.values_mut() {
                resolve_expression(bound, module, geometries, value);
            }
        }
        CheckedExpressionKind::Unary { operand, .. }
        | CheckedExpressionKind::Quantity { value: operand, .. } => {
            resolve_expression(bound, module, geometries, operand)
        }
        CheckedExpressionKind::Binary { left, right, .. } => {
            resolve_expression(bound, module, geometries, left);
            resolve_expression(bound, module, geometries, right);
        }
        CheckedExpressionKind::OperationCall { arguments, .. }
        | CheckedExpressionKind::FunctionCall { arguments, .. } => {
            for argument in arguments {
                let value = match argument {
                    CheckedCallArgument::Positional(value)
                    | CheckedCallArgument::Named { value, .. } => value,
                };
                resolve_expression(bound, module, geometries, value);
            }
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

fn resolve_reference(
    bound: &BoundFeaProgram,
    module: &LogicalSourcePath,
    geometries: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), GeometryDefinition>,
    path: &[String],
) -> Option<ResolvedCadParameterValue> {
    let [namespace, geometry_name, parameter_namespace, parameter_name] = path else {
        return None;
    };
    if namespace != "geometry" || parameter_namespace != "parameter" {
        return None;
    }
    let exposed_key = FeaSymbolKey {
        kind: FeaSymbolKind::Geometry,
        name: geometry_name.clone(),
    };
    let symbol = bound.symbols.get(module)?.get(&exposed_key)?;
    let (source_module, source_name) = match &symbol.origin {
        FeaSymbolOrigin::Local => (module.clone(), geometry_name.clone()),
        FeaSymbolOrigin::Import {
            target_module,
            target,
        } => (target_module.clone(), target.name.clone()),
        FeaSymbolOrigin::Function => return None,
    };
    let geometry = geometries.get(&(
        source_module.clone(),
        FeaSymbolKey {
            kind: FeaSymbolKind::Geometry,
            name: source_name.clone(),
        },
    ))?;
    let parameter = geometry
        .cad_parameters
        .iter()
        .find(|parameter| parameter.parameter_id == *parameter_name)?;
    Some(ResolvedCadParameterValue {
        geometry_module_path: source_module.as_str().to_string(),
        geometry_id: source_name,
        parameter_id: parameter.source.parameter_id.clone(),
        source_entity_id: parameter.source.source_entity_id,
        persistent_id: parameter.source.persistent_id.clone(),
        measure: parameter.source.value.clone(),
    })
}
