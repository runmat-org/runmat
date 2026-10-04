use super::{lower, values};
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatementKind,
    FeaDiagnostic, FeaType,
};
use runmat_analysis_core::{
    AssessmentEvidenceRequirement, AssessmentFunctionReference, AssessmentQuantity, AssessmentRule,
    AuthoredAssessment,
};
use std::collections::BTreeMap;

pub(super) fn lower_assessments(
    bound_program: &crate::BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredAssessment> {
    let mut result = Vec::new();
    for statement in &declaration.statements {
        let (name, statements) = match &statement.kind {
            CheckedFeaStatementKind::AssessmentDefinition { name, statements } => {
                (name, statements)
            }
            CheckedFeaStatementKind::Block { header, .. }
                if header.split_whitespace().next() == Some("assessment") =>
            {
                lower::error(
                    declaration,
                    statement.span,
                    "assessment requires one name",
                    diagnostics,
                );
                continue;
            }
            _ => continue,
        };
        let initial_errors = diagnostics.len();
        let mut fields = BTreeMap::new();
        for field in statements {
            match &field.kind {
                CheckedFeaStatementKind::Assignment { target, value, .. }
                    if target.len() == 1
                        && matches!(
                            target[0].as_str(),
                            "value"
                                | "minimum"
                                | "maximum"
                                | "evidence"
                                | "function"
                                | "parameters"
                        ) =>
                {
                    if fields.insert(target[0].as_str(), value).is_some() {
                        lower::error(
                            declaration,
                            field.span,
                            "duplicate assessment setting",
                            diagnostics,
                        );
                    }
                }
                _ => lower::error(
                    declaration,
                    field.span,
                    "assessment accepts value, minimum, maximum, evidence, function and parameters",
                    diagnostics,
                ),
            }
        }
        let output = fields.get("value").and_then(|value| match &value.kind {
            CheckedExpressionKind::Reference(path) if path.len() == 2 && path[0] == "output" => {
                Some(path[1].clone())
            }
            _ => None,
        });
        let bound = |value: &&CheckedExpression| {
            let dimension = match value.value_type {
                FeaType::Quantity(dimension) => dimension,
                FeaType::Real | FeaType::Integer => runmat_types::PhysicalDimension::DIMENSIONLESS,
                _ => return None,
            };
            Some(AssessmentQuantity {
                value_si: values::canonical_scalar(value)?,
                dimension,
            })
        };
        let minimum = fields.get("minimum").and_then(bound);
        let maximum = fields.get("maximum").and_then(bound);
        let rule = if let Some(function) = fields.get("function") {
            if fields.contains_key("minimum") || fields.contains_key("maximum") {
                lower::error(
                    declaration,
                    statement.span,
                    "function and comparison bounds cannot be combined",
                    diagnostics,
                );
                None
            } else {
                lower_function_rule(
                    bound_program,
                    declaration,
                    function,
                    fields.get("parameters").copied(),
                    diagnostics,
                )
            }
        } else if fields.contains_key("parameters") {
            lower::error(
                declaration,
                statement.span,
                "assessment parameters require a function",
                diagnostics,
            );
            None
        } else {
            match (minimum, maximum) {
                (Some(minimum), Some(maximum)) => Some(AssessmentRule::Range { minimum, maximum }),
                (Some(bound), None) if !fields.contains_key("maximum") => {
                    Some(AssessmentRule::Minimum { bound })
                }
                (None, Some(bound)) if !fields.contains_key("minimum") => {
                    Some(AssessmentRule::Maximum { bound })
                }
                _ => None,
            }
        };
        let evidence = fields.get("evidence").and_then(|value| match &value.kind {
            CheckedExpressionKind::Enum(name) if name == "authored_samples" => {
                Some(AssessmentEvidenceRequirement::AuthoredSamples)
            }
            CheckedExpressionKind::Enum(name) if name == "certified_bound" => {
                Some(AssessmentEvidenceRequirement::CertifiedBound)
            }
            _ => None,
        });
        match (output,rule,evidence) {
            (Some(output_id),Some(rule),Some(evidence)) if diagnostics.len() == initial_errors => result.push(AuthoredAssessment { assessment_id: name.clone(), output_id, rule, evidence }),
            _ => lower::error(declaration, statement.span, "assessment requires a declared output, finite dimensioned bounds, and explicit evidence = authored_samples or certified_bound", diagnostics),
        }
    }
    result
}

fn lower_function_rule(
    bound: &crate::BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    parameters: Option<&CheckedExpression>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AssessmentRule> {
    use runmat_types::{EffectKind, OperationParameterArity, OperationTypePattern};
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        return None;
    };
    let [namespace, name] = path.as_slice() else {
        return None;
    };
    if namespace != "function" {
        return None;
    }
    let facts = bound
        .functions
        .get(&(declaration.module.clone(), name.clone()))?;
    let parameters = match parameters {
        None => Vec::new(),
        Some(value) => {
            let CheckedExpressionKind::List(values) = &value.kind else {
                return None;
            };
            values
                .iter()
                .map(|value| {
                    Some(AssessmentQuantity {
                        value_si: values::canonical_scalar(value)?,
                        dimension: match value.value_type {
                            FeaType::Quantity(dimension) => dimension,
                            FeaType::Real | FeaType::Integer => {
                                runmat_types::PhysicalDimension::DIMENSIONLESS
                            }
                            _ => return None,
                        },
                    })
                })
                .collect::<Option<Vec<_>>>()?
        }
    };
    let required_result = OperationTypePattern::Record(BTreeMap::from([
        ("evaluated".into(), OperationTypePattern::LogicalScalar),
        ("accepted".into(), OperationTypePattern::LogicalScalar),
    ]));
    if facts.inputs.len() != parameters.len() + 1
        || facts.inputs.iter().any(|input| {
            input.arity != OperationParameterArity::Required
                || input.value != OperationTypePattern::NumericScalar
        })
        || facts.outputs != [required_result]
        || facts
            .effects
            .0
            .iter()
            .any(|effect| *effect != EffectKind::MayThrow)
        || !facts.capabilities.0.is_empty()
    {
        lower::error(declaration, expression.span,
            "assessment function requires pure required real scalar inputs and one complete record with logical scalar evaluated and accepted fields", diagnostics);
        return None;
    }
    Some(AssessmentRule::Function {
        function: AssessmentFunctionReference {
            module_path: facts.module_path.as_str().into(),
            symbol: facts.symbol.as_str().into(),
            source_digest: facts.source_digest.0,
        },
        parameters,
    })
}
