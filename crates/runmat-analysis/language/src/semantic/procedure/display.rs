//! `display <output> { unit = MPa; range = [0 MPa, 12 MPa]; precision = 3 }`: how a study shows
//! one of its field outputs. Every member is optional and at least one is required. Without a
//! `unit`, the legend takes the unit the study writes that quantity in elsewhere (see
//! [`units`]); a field output with no `display` statement still gets that default. The view's
//! solved bodies and deformation scale (`hide`, `context`, `deformation`) are in [`view`].
use super::values;
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaContextualWord, FeaDiagnostic, FeaDiagnosticLocation,
    FeaDiagnosticPhase, FeaSymbolKey, FeaSymbolKind, FeaType, LogicalSourcePath,
};
use runmat_analysis_core::{
    AuthoredOutput, AuthoredOutputDisplay, AuthoredOutputType, AuthoredOutputValue, DisplayRange,
    DisplayUnitSource, ProcedureStep, DISPLAY_PRECISION_DIGITS,
};
use runmat_types::physical::parse_unit_expression;
use runmat_types::PhysicalDimension;
use runmat_types::Span;
use std::collections::BTreeMap;

mod units;
mod view;

/// The checked statements whose authored units set a field output's default display unit:
/// the study's own and those of every model it runs.
pub(super) struct AuthoredUnitSources<'a> {
    pub study: &'a CheckedFeaDeclaration,
    pub models: Vec<&'a CheckedFeaDeclaration>,
}

impl<'a> AuthoredUnitSources<'a> {
    /// The study `declaration` and the checked declarations of `models` (its model closure).
    pub fn of(
        declarations: &'a BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
        declaration: &'a CheckedFeaDeclaration,
        models: &[&runmat_analysis_core::AuthoredModel],
    ) -> Self {
        Self {
            study: declaration,
            models: models
                .iter()
                .filter_map(|model| {
                    declarations.iter().find_map(|((path, key), checked)| {
                        (key.kind == FeaSymbolKind::Model
                            && key.name == model.model_id
                            && path.as_str() == model.module_path)
                            .then_some(checked)
                    })
                })
                .collect(),
        }
    }
}

/// The display of each field output of the study that has a `display` statement or a default
/// unit other than SI, in output order.
pub(super) fn lower_displays(
    sources: &AuthoredUnitSources<'_>,
    outputs: &[AuthoredOutput],
    steps: &[ProcedureStep],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredOutputDisplay> {
    let declaration = sources.study;
    let fields = field_dimensions(outputs, steps);
    let mut authored = BTreeMap::new();
    for statement in &declaration.statements {
        let CheckedFeaStatementKind::Block {
            header,
            header_span,
            statements,
        } = &statement.kind
        else {
            continue;
        };
        let mut words = header.split_whitespace();
        if words.next().and_then(FeaContextualWord::from_source) != Some(FeaContextualWord::Display)
        {
            continue;
        }
        let (Some(output), None) = (words.next(), words.next()) else {
            error(
                declaration,
                "fea.semantic.display_output",
                *header_span,
                "display requires one field output: `display <output> { … }`",
                diagnostics,
            );
            continue;
        };
        let Some(&dimension) = fields.get(output) else {
            error(
                declaration,
                "fea.semantic.display_output",
                *header_span,
                format!("`display {output}` names no field output of this study"),
                diagnostics,
            );
            continue;
        };
        if authored.contains_key(output) {
            error(
                declaration,
                "fea.semantic.display_output",
                *header_span,
                format!("output `{output}` has more than one display"),
                diagnostics,
            );
            continue;
        }
        if let Some(display) = lower_members(
            declaration,
            statement.span,
            output,
            dimension,
            statements,
            diagnostics,
        ) {
            authored.insert(output.to_string(), display);
        }
    }
    outputs
        .iter()
        .filter_map(|output| {
            let dimension = *fields.get(output.output_id.as_str())?;
            let mut display = authored.remove(&output.output_id).unwrap_or_else(|| {
                AuthoredOutputDisplay::unit_only(&output.output_id, DisplayUnitSource::Si)
            });
            if display.unit.is_none() {
                if let Some(unit) = units::authored_unit(sources, dimension) {
                    display.unit = Some(unit);
                    display.unit_source = DisplayUnitSource::Study;
                }
            }
            let shown = display.unit_source == DisplayUnitSource::Display
                || display.range.is_some()
                || display.precision.is_some()
                || !display.shows_every_region()
                || display.deformation.is_some()
                || display
                    .unit
                    .as_deref()
                    .is_some_and(|unit| !units::is_si(unit));
            shown.then_some(display)
        })
        .collect()
}

/// The physical dimension of each field output, by output id.
fn field_dimensions<'a>(
    outputs: &'a [AuthoredOutput],
    steps: &[ProcedureStep],
) -> BTreeMap<&'a str, PhysicalDimension> {
    let Ok(types) = runmat_analysis_core::authored_output_types(outputs, steps) else {
        return BTreeMap::new();
    };
    outputs
        .iter()
        .filter_map(|output| match types.get(&output.output_id)? {
            AuthoredOutputType::Field { value, .. } => Some((
                output.output_id.as_str(),
                match value {
                    AuthoredOutputValue::Physical(quantity) => quantity.physical_dimension(),
                    AuthoredOutputValue::Derived(dimension) => dimension.dimension(),
                    AuthoredOutputValue::Dimensionless => PhysicalDimension::DIMENSIONLESS,
                },
            )),
            _ => None,
        })
        .collect()
}

fn lower_members(
    declaration: &CheckedFeaDeclaration,
    span: Span,
    output: &str,
    dimension: PhysicalDimension,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredOutputDisplay> {
    let initial = diagnostics.len();
    let mut members = BTreeMap::new();
    for statement in statements {
        let word = match &statement.kind {
            CheckedFeaStatementKind::Assignment { target, value, .. } if target.len() == 1 => {
                FeaContextualWord::from_source(&target[0])
                    .filter(|word| {
                        matches!(
                            word,
                            FeaContextualWord::Unit
                                | FeaContextualWord::Range
                                | FeaContextualWord::Precision
                                | FeaContextualWord::Hide
                                | FeaContextualWord::Context
                                | FeaContextualWord::Deformation
                        )
                    })
                    .map(|word| (word, value))
            }
            _ => None,
        };
        let Some((word, value)) = word else {
            error(
                declaration,
                "fea.semantic.display_member",
                statement.span,
                "display accepts unit, range, precision, hide, context and deformation",
                diagnostics,
            );
            continue;
        };
        if members
            .insert(word.as_source(), (value, statement.span))
            .is_some()
        {
            error(
                declaration,
                "fea.semantic.display_member",
                statement.span,
                format!("duplicate display member `{}`", word.as_source()),
                diagnostics,
            );
        }
    }
    if members.is_empty() && diagnostics.len() == initial {
        error(
            declaration,
            "fea.semantic.display_member",
            span,
            format!(
                "`display {output}` sets nothing; give it a unit, a range, a precision, hide, \
                 context or deformation"
            ),
            diagnostics,
        );
    }
    let unit = members
        .get(FeaContextualWord::Unit.as_source())
        .and_then(|(value, _)| display_unit(declaration, value, dimension, diagnostics));
    let range = members
        .get(FeaContextualWord::Range.as_source())
        .and_then(|(value, _)| display_range(declaration, value, dimension, diagnostics));
    let precision = members
        .get(FeaContextualWord::Precision.as_source())
        .and_then(|(value, _)| display_precision(declaration, value, diagnostics));
    let member = |word: FeaContextualWord| members.get(word.as_source()).map(|(value, _)| *value);
    let regions = view::display_regions(
        declaration,
        member(FeaContextualWord::Hide),
        member(FeaContextualWord::Context),
        diagnostics,
    );
    let deformation = member(FeaContextualWord::Deformation)
        .and_then(|value| view::display_deformation(declaration, value, diagnostics));
    (diagnostics.len() == initial).then(|| AuthoredOutputDisplay {
        unit_source: if unit.is_some() {
            DisplayUnitSource::Display
        } else {
            DisplayUnitSource::Si
        },
        unit,
        range,
        precision,
        hide: regions.hide,
        context: regions.context,
        deformation,
        ..AuthoredOutputDisplay::unit_only(output, DisplayUnitSource::Si)
    })
}

/// `unit = MPa`, or a quoted symbol for a compound unit (`unit = "W/m^2"`).
fn display_unit(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<String> {
    let symbol = match &value.kind {
        CheckedExpressionKind::Enum(symbol) | CheckedExpressionKind::String(symbol) => symbol,
        _ => {
            error(
                declaration,
                "fea.semantic.display_unit",
                value.span,
                "display unit requires a unit symbol, such as `unit = MPa` or `unit = \"W/m^2\"`",
                diagnostics,
            );
            return None;
        }
    };
    match parse_unit_expression(symbol) {
        Ok(unit) if unit.dimension() == dimension => Some(symbol.trim().to_string()),
        Ok(_) => {
            error(
                declaration,
                "fea.semantic.display_unit",
                value.span,
                format!("display unit `{symbol}` does not measure this field's quantity"),
                diagnostics,
            );
            None
        }
        Err(problem) => {
            error(
                declaration,
                "fea.semantic.display_unit",
                value.span,
                format!("display unit `{symbol}`: {problem}"),
                diagnostics,
            );
            None
        }
    }
}

/// `range = [0 MPa, 12 MPa]` (or `[0, 12] MPa`), in the field's dimension, minimum below maximum.
fn display_range(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<DisplayRange> {
    let measures = |value: &CheckedExpression| match &value.value_type {
        FeaType::Quantity(found) => found.dimension() == dimension,
        FeaType::Real | FeaType::Integer => dimension == PhysicalDimension::DIMENSIONLESS,
        FeaType::Vector { element, .. } | FeaType::List(element) => match element.as_ref() {
            FeaType::Quantity(found) => found.dimension() == dimension,
            FeaType::Real | FeaType::Integer => dimension == PhysicalDimension::DIMENSIONLESS,
            _ => false,
        },
        _ => false,
    };
    let bounds = match &value.kind {
        CheckedExpressionKind::List(items) if items.len() == 2 => items
            .iter()
            .all(measures)
            .then(|| values::canonical_list(value))
            .flatten(),
        CheckedExpressionKind::Quantity { .. } if measures(value) => values::canonical_list(value),
        _ => None,
    };
    let Some([minimum, maximum]) = bounds.as_deref().map(|bounds| match bounds {
        [minimum, maximum] => [*minimum, *maximum],
        _ => [f64::NAN, f64::NAN],
    }) else {
        error(
            declaration,
            "fea.semantic.display_range",
            value.span,
            "display range requires two values in the field's unit dimension, such as \
             `range = [0 MPa, 12 MPa]`",
            diagnostics,
        );
        return None;
    };
    if !(minimum.is_finite() && maximum.is_finite() && minimum < maximum) {
        error(
            declaration,
            "fea.semantic.display_range",
            value.span,
            "display range requires two finite values with the minimum below the maximum",
            diagnostics,
        );
        return None;
    }
    Some(DisplayRange { minimum, maximum })
}

/// `precision = 3`: significant digits, an integer from 1 to 6.
fn display_precision(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<u8> {
    let digits = match &value.kind {
        CheckedExpressionKind::Number(source) => source.parse::<u8>().ok(),
        _ => None,
    }
    .filter(|digits| DISPLAY_PRECISION_DIGITS.contains(digits));
    if digits.is_none() {
        error(
            declaration,
            "fea.semantic.display_precision",
            value.span,
            "display precision is a number of significant digits, an integer from 1 to 6",
            diagnostics,
        );
    }
    digits
}

fn error(
    declaration: &CheckedFeaDeclaration,
    code: &str,
    span: Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        code,
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}
