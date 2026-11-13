use super::header::ProcedureHeader;
use super::lower::error;
use super::values;
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    FeaDiagnostic, FeaType,
};
use runmat_analysis_core::IndependentDomain;

pub fn count(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<usize> {
    values::assignment(statements, "modes")
        .and_then(values::positive_usize)
        .or_else(|| missing(declaration, header, "positive integer `modes`", diagnostics))
}

pub fn ratio(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    name: &str,
    allow_zero: bool,
    allow_one: bool,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let Some(expression) = values::assignment(statements, name) else {
        return missing(
            declaration,
            header,
            &format!("dimensionless ratio `{name}`"),
            diagnostics,
        );
    };
    let value = matches!(expression.value_type, FeaType::Integer | FeaType::Real)
        .then(|| values::canonical_scalar(expression))
        .flatten();
    let valid = value.is_some_and(|value| {
        value.is_finite()
            && (value < 1.0 || (allow_one && value == 1.0))
            && (value > 0.0 || (allow_zero && value == 0.0))
    });
    valid.then_some(value.unwrap_or_default()).or_else(|| {
        invalid(
            declaration,
            header,
            &format!(
                "`{name}` must be a finite dimensionless ratio in {}",
                match (allow_zero, allow_one) {
                    (true, true) => "[0, 1]",
                    (true, false) => "[0, 1)",
                    (false, true) => "(0, 1]",
                    (false, false) => "(0, 1)",
                }
            ),
            diagnostics,
        )
    })
}

pub fn time(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<IndependentDomain> {
    let start = time_scalar(declaration, header, statements, "start_time", diagnostics)?;
    let end = time_scalar(declaration, header, statements, "end_time", diagnostics)?;
    let time_step = time_scalar(declaration, header, statements, "time_step", diagnostics)?;
    let interval_count = exact_interval_count(start, end, time_step);
    (start >= 0.0 && end > start && interval_count.is_some())
        .then_some(IndependentDomain::TimeSeconds {
            origin: 0.0,
            start,
            end,
            interval_count: interval_count.unwrap_or_default(),
        })
        .or_else(|| {
            invalid(
                declaration,
                header,
                "time range must satisfy 0 <= start_time < end_time and positive time_step must exactly partition it",
                diagnostics,
            )
        })
}

fn exact_interval_count(start: f64, end: f64, time_step: f64) -> Option<usize> {
    if !start.is_finite() || !end.is_finite() || !time_step.is_finite() || time_step <= 0.0 {
        return None;
    }
    let count = (end - start) / time_step;
    let rounded = count.round();
    let tolerance = 64.0 * f64::EPSILON * count.abs().max(1.0);
    if rounded < 1.0
        || (count - rounded).abs() > tolerance
        || rounded > runmat_types::wire_integer::MAX_WIRE_INTEGER as f64
    {
        return None;
    }
    Some(rounded as usize)
}

fn time_scalar(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let value = values::assignment(statements, name);
    match value {
        Some(value) if (value.value_type).is_quantity((crate::PhysicalDimension::TIME).into()) => {
            values::canonical_scalar(value)
        }
        _ => missing(
            declaration,
            header,
            &format!("time quantity `{name}`"),
            diagnostics,
        ),
    }
}

pub fn frequencies(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<IndependentDomain> {
    let values = values::assignment(statements, "frequencies")
        .and_then(values::canonical_list)
        .or_else(|| {
            missing(
                declaration,
                header,
                "non-empty quantity list `frequencies`",
                diagnostics,
            )
        })?;
    let valid = !values.is_empty()
        && values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && values.windows(2).all(|pair| pair[0] < pair[1]);
    valid
        .then_some(IndependentDomain::FrequencyHertz(values))
        .or_else(|| {
            invalid(
                declaration,
                header,
                "frequencies must be finite, nonnegative, and strictly increasing",
                diagnostics,
            )
        })
}

pub fn load_factor(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<IndependentDomain> {
    let start = scalar(
        declaration,
        header,
        statements,
        "start_load_factor",
        diagnostics,
    )?;
    let end = scalar(
        declaration,
        header,
        statements,
        "end_load_factor",
        diagnostics,
    )?;
    (start.is_finite() && end > start)
        .then_some(IndependentDomain::LoadFactor { start, end })
        .or_else(|| {
            invalid(
                declaration,
                header,
                "load-factor range requires finite start_load_factor < end_load_factor",
                diagnostics,
            )
        })
}

pub fn state(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(String, String)> {
    let path = values::assignment(statements, name).and_then(values::reference);
    match path {
        Some([analysis, producer, state, state_id])
            if analysis == "analysis" && state == "state" =>
        {
            Some((producer.clone(), state_id.clone()))
        }
        _ => missing(
            declaration,
            header,
            &format!("`{name} = analysis.<name>.state.<name>`"),
            diagnostics,
        ),
    }
}

pub fn reference(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    name: &str,
    namespace: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<String> {
    let expression = values::assignment(statements, name)?;
    model_reference_identity(expression, namespace).or_else(|| {
        missing(
            declaration,
            header,
            &format!("`{name} = model.{namespace}.<name>`"),
            diagnostics,
        )
    })
}

pub fn references(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    name: &str,
    namespace: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Vec<String>> {
    let expression = values::assignment(statements, name)?;
    let CheckedExpressionKind::List(items) = &expression.kind else {
        return missing(
            declaration,
            header,
            &format!("reference list `{name}`"),
            diagnostics,
        );
    };
    let result = items
        .iter()
        .map(|item| model_reference_identity(item, namespace))
        .collect::<Option<Vec<_>>>()?;
    (!result.is_empty()).then_some(result).or_else(|| {
        missing(
            declaration,
            header,
            &format!("non-empty reference list `{name}`"),
            diagnostics,
        )
    })
}

fn scalar(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    values::assignment(statements, name)
        .and_then(values::canonical_scalar)
        .or_else(|| {
            missing(
                declaration,
                header,
                &format!("scalar `{name}`"),
                diagnostics,
            )
        })
}

fn model_reference_identity(value: &CheckedExpression, namespace: &str) -> Option<String> {
    let [model, actual_namespace, identity] = values::reference(value)? else {
        return None;
    };
    (model == "model" && actual_namespace == namespace).then(|| identity.clone())
}

fn missing<T>(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    requirement: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<T> {
    error(
        declaration,
        header.span,
        format!(
            "procedure `{} {}` requires {requirement}",
            header.family, header.procedure
        ),
        diagnostics,
    );
    None
}

fn invalid<T>(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    message: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<T> {
    error(declaration, header.span, message, diagnostics);
    None
}
