//! The unit a study writes a physical quantity in, which is the default unit of a field output's
//! legend. The first unit of the field's dimension found in this order wins:
//! 1. an assessment bound of the study (`maximum = 184 MPa`);
//! 2. the study's analysis blocks (initial values, reference temperatures);
//! 3. the loads, constraints, interfaces, excitations and initial conditions of the models the
//!    study runs, in the order the models include each other and then source order.
//!
//! Section, layup and orientation definitions are left out: their lengths describe the cross
//! section, not the result. Outputs are left out: a probe point in millimetres is a location.
use super::AuthoredUnitSources;
use crate::{
    CheckedCallArgument, CheckedExpression, CheckedExpressionKind, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaContextualWord,
};
use runmat_design_syntax::FeaModelDefinitionKind;
use runmat_types::physical::parse_unit_expression;
use runmat_types::PhysicalDimension;

pub(super) fn authored_unit(
    sources: &AuthoredUnitSources<'_>,
    dimension: PhysicalDimension,
) -> Option<String> {
    let study = &sources.study.statements;
    let assessments = study.iter().filter(|statement| {
        matches!(
            statement.kind,
            CheckedFeaStatementKind::AssessmentDefinition { .. }
        )
    });
    let analyses = study.iter().filter(|statement| {
        matches!(&statement.kind, CheckedFeaStatementKind::Block { header, .. }
            if header.split_whitespace().next() == Some("analysis"))
    });
    let models = sources
        .models
        .iter()
        .flat_map(|model| model.statements.iter());
    assessments
        .chain(analyses)
        .chain(models)
        .find_map(|statement| in_statement(statement, dimension))
}

/// Whether `symbol` is the SI unit of its dimension (scale 1, no offset): `Pa`, `K`, `m`.
pub(super) fn is_si(symbol: &str) -> bool {
    parse_unit_expression(symbol).is_ok_and(|unit| unit.scale() == 1.0 && unit.offset() == 0.0)
}

fn in_statement(statement: &CheckedFeaStatement, dimension: PhysicalDimension) -> Option<String> {
    let nested = |statements: &[CheckedFeaStatement]| {
        statements
            .iter()
            .find_map(|statement| in_statement(statement, dimension))
    };
    match &statement.kind {
        CheckedFeaStatementKind::AssessmentDefinition { statements, .. } => statements
            .iter()
            .find_map(|statement| match &statement.kind {
                CheckedFeaStatementKind::Assignment { target, value, .. }
                    if target.len() == 1
                        && matches!(
                            FeaContextualWord::from_source(&target[0]),
                            Some(FeaContextualWord::Minimum | FeaContextualWord::Maximum)
                        ) =>
                {
                    in_expression(value, dimension)
                }
                _ => None,
            }),
        CheckedFeaStatementKind::ModelDefinition {
            kind:
                FeaModelDefinitionKind::Section
                | FeaModelDefinitionKind::Layup
                | FeaModelDefinitionKind::Orientation,
            ..
        } => None,
        CheckedFeaStatementKind::ModelDefinition { statements, .. }
        | CheckedFeaStatementKind::Block { statements, .. }
        | CheckedFeaStatementKind::GeneratedByFunction { statements, .. } => nested(statements),
        CheckedFeaStatementKind::AppliedLoad { value, .. }
        | CheckedFeaStatementKind::Assignment { value, .. } => in_expression(value, dimension),
        _ => None,
    }
}

fn in_expression(expression: &CheckedExpression, dimension: PhysicalDimension) -> Option<String> {
    fn first<'a>(
        mut expressions: impl Iterator<Item = &'a CheckedExpression>,
        dimension: PhysicalDimension,
    ) -> Option<String> {
        expressions.find_map(|expression| in_expression(expression, dimension))
    }
    match &expression.kind {
        CheckedExpressionKind::Quantity {
            unit,
            source_unit_symbol,
            ..
        } => (unit.dimension() == dimension).then(|| source_unit_symbol.clone()),
        CheckedExpressionKind::Unary { operand, .. } => in_expression(operand, dimension),
        CheckedExpressionKind::Binary { left, right, .. } => {
            in_expression(left, dimension).or_else(|| in_expression(right, dimension))
        }
        CheckedExpressionKind::List(items) => first(items.iter(), dimension),
        CheckedExpressionKind::Record(fields) => first(fields.values(), dimension),
        CheckedExpressionKind::OperationCall { arguments, .. }
        | CheckedExpressionKind::FunctionCall { arguments, .. } => {
            let values = arguments.iter().map(|argument| match argument {
                CheckedCallArgument::Positional(value)
                | CheckedCallArgument::Named { value, .. } => value,
            });
            first(values, dimension)
        }
        _ => None,
    }
}
