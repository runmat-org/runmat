//! A field legend in the study's own terms: the unit its assessment of that quantity is
//! written in, and that assessment's limit with the run's own margin to it.
use anyhow::Result;
use runmat_analysis_core::{AssessmentRule, AssessmentStatus};
use runmat_analysis_language::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaProgram, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaSymbolKind,
};
use runmat_plot::geometry_scene::GeometrySceneScalarLegend;
use runmat_plot::legend_text::{format_stated, printed_unit, LegendPrecision};
use runmat_runtime::analysis::assessment::PublishedStudyAssessment;
use runmat_types::physical::{parse_unit_expression, UnitDefinition};
use std::collections::BTreeMap;

/// The unit each of the study's assessments writes its bound in (`maximum = 85 degC`), by
/// assessment id. The compiled study keeps only canonical values, so this reads the source.
pub(in crate::commands::fea) fn authored_bound_units(
    program: &CheckedFeaProgram,
    study_id: &str,
) -> BTreeMap<String, String> {
    let mut units = BTreeMap::new();
    for ((_, key), declaration) in &program.declarations {
        if key.kind == FeaSymbolKind::Study && key.name == study_id {
            collect(&declaration.statements, &mut units);
        }
    }
    units
}

fn collect(statements: &[CheckedFeaStatement], units: &mut BTreeMap<String, String>) {
    for statement in statements {
        match &statement.kind {
            CheckedFeaStatementKind::AssessmentDefinition { name, statements } => {
                let bound = |field: &str| {
                    statements
                        .iter()
                        .find_map(|statement| match &statement.kind {
                            CheckedFeaStatementKind::Assignment { target, value, .. }
                                if target.len() == 1 && target[0] == field =>
                            {
                                quantity_unit(value)
                            }
                            _ => None,
                        })
                };
                if let Some(unit) = bound("maximum").or_else(|| bound("minimum")) {
                    units.insert(name.clone(), unit);
                }
            }
            CheckedFeaStatementKind::Block { statements, .. } => collect(statements, units),
            _ => {}
        }
    }
}

fn quantity_unit(expression: &CheckedExpression) -> Option<String> {
    match &expression.kind {
        CheckedExpressionKind::Quantity {
            source_unit_symbol, ..
        } => Some(source_unit_symbol.trim().to_string()),
        CheckedExpressionKind::Unary { operand, .. } => quantity_unit(operand),
        _ => None,
    }
}

/// One assessment bound of the shown quantity and the run's result against it.
struct Bound {
    canonical: f64,
    maximum: bool,
    /// Signed distance inside the bound, canonical: negative when the result breaks it.
    margin: Option<f64>,
}

fn bounds_of(assessment: &PublishedStudyAssessment, field: &UnitDefinition) -> Vec<Bound> {
    let measured = assessment
        .result
        .measured
        .filter(|_| assessment.result.status != AssessmentStatus::Indeterminate)
        .map(|value| *value.canonical_values());
    let (minimum, maximum) = match &assessment.definition.rule {
        AssessmentRule::Function { .. } => return Vec::new(),
        rule => rule.bounds(),
    };
    [(minimum, false), (maximum, true)]
        .into_iter()
        .filter_map(|(bound, is_maximum)| {
            let bound = bound?;
            (bound.descriptor().dimension() == field.dimension()).then(|| {
                let canonical = *bound.canonical_values();
                Bound {
                    canonical,
                    maximum: is_maximum,
                    margin: measured.map(|measured| {
                        if is_maximum {
                            canonical - measured
                        } else {
                            measured - canonical
                        }
                    }),
                }
            })
        })
        .collect()
}

/// Write the legend in the display unit of the shown quantity: the unit the study's assessment
/// of that quantity states, else the field's own unit. When the study assesses the quantity,
/// the tightest bound is marked and the run's margin to it stated, such as "21.6 K under the
/// 85 °C limit". Returns the display unit when it differs from the field's.
pub(super) fn present(
    legend: &mut GeometrySceneScalarLegend,
    field_unit: &str,
    display_units: &BTreeMap<String, String>,
    assessments: &[PublishedStudyAssessment],
) -> Result<Option<String>> {
    let Ok(field) = parse_unit_expression(field_unit) else {
        return Ok(None);
    };
    let assessed: Vec<_> = assessments
        .iter()
        .map(|assessment| (assessment, bounds_of(assessment, &field)))
        .filter(|(_, bounds)| !bounds.is_empty())
        .collect();
    let display = assessed.iter().find_map(|(assessment, _)| {
        let symbol = display_units.get(&assessment.definition.assessment_id)?;
        let unit = parse_unit_expression(symbol).ok()?;
        (unit.dimension() == field.dimension()).then(|| (symbol.clone(), unit))
    });
    let (symbol, unit) = display
        .clone()
        .unwrap_or_else(|| (field_unit.to_string(), field));
    let shown =
        |value: f64| -> Result<f64> { Ok(unit.from_canonical(field.to_canonical(value)?)?) };
    legend.minimum = shown(legend.minimum)?;
    legend.maximum = shown(legend.maximum)?;
    legend.unit = symbol.clone();
    // The bound the run comes closest to (or breaks furthest); without a result, the first.
    let tightest = assessed
        .iter()
        .flat_map(|(_, bounds)| bounds)
        .min_by(|a, b| {
            let key = |bound: &Bound| bound.margin.unwrap_or(f64::INFINITY);
            key(a).total_cmp(&key(b))
        });
    if let Some(bound) = tightest {
        let limit = unit.from_canonical(bound.canonical)?;
        legend.limit = Some(limit);
        let stated = format!("{} {}", format_stated(limit), printed_unit(&symbol));
        let name = if bound.maximum { "limit" } else { "minimum" };
        legend.note = Some(match bound.margin {
            None => format!("{stated} {name}"),
            Some(margin) => {
                let precision = LegendPrecision::for_range(legend.minimum, legend.maximum);
                let amount = precision.format(margin.abs() / unit.scale());
                // A margin of an absolute temperature is a temperature difference.
                let difference = match symbol.as_str() {
                    "degC" => "K".to_string(),
                    other => printed_unit(other),
                };
                let side = match (margin >= 0.0, bound.maximum) {
                    _ if amount == "0" => "at",
                    (true, true) => "under",
                    (false, true) => "over",
                    (true, false) => "above",
                    (false, false) => "below",
                };
                if side == "at" {
                    format!("At the {stated} {name}")
                } else {
                    format!("{amount} {difference} {side} the {stated} {name}")
                }
            }
        });
    }
    Ok(display.map(|(symbol, _)| symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::{
        AssessmentEvidenceRequirement, AssessmentQuantity, AssessmentResult, AuthoredAssessment,
    };
    use runmat_types::PhysicalDimension;

    fn kelvin(value: f64) -> AssessmentQuantity {
        AssessmentQuantity::try_from_descriptor(PhysicalDimension::TEMPERATURE.into(), value)
            .unwrap()
    }

    fn assessment(rule: AssessmentRule, measured: f64) -> PublishedStudyAssessment {
        PublishedStudyAssessment {
            definition: AuthoredAssessment {
                assessment_id: "stays_below_85_celsius".into(),
                output_id: "peak_temperature".into(),
                rule,
                evidence: AssessmentEvidenceRequirement::AuthoredSamples,
            },
            result: AssessmentResult {
                assessment_id: "stays_below_85_celsius".into(),
                output_id: "peak_temperature".into(),
                status: AssessmentStatus::Passed,
                measured: Some(kelvin(measured)),
                issue: None,
            },
            study_definition:
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                    .parse()
                    .unwrap(),
            run_id: None,
            evidence: None,
            function_execution_identity: None,
            function_execution_diagnostic: None,
        }
    }

    fn busbar_legend() -> GeometrySceneScalarLegend {
        GeometrySceneScalarLegend {
            label: "Temperature".into(),
            unit: "K".into(),
            minimum: 329.9642170922981,
            maximum: 336.5473805835758,
            limit: None,
            note: None,
        }
    }

    #[test]
    fn the_busbar_legend_reads_in_celsius_with_its_margin_to_the_limit() {
        let units = BTreeMap::from([("stays_below_85_celsius".into(), "degC".into())]);
        let mut legend = busbar_legend();
        let assessed = [assessment(
            AssessmentRule::Maximum {
                bound: kelvin(358.15),
            },
            336.5473805835758,
        )];
        let display = present(&mut legend, "K", &units, &assessed).unwrap();
        assert_eq!(display.as_deref(), Some("degC"));
        let precision = LegendPrecision::for_range(legend.minimum, legend.maximum);
        assert_eq!(precision.format(legend.minimum), "56.8");
        assert_eq!(precision.format(legend.maximum), "63.4");
        assert_eq!(legend.unit, "degC");
        assert!((legend.limit.unwrap() - 85.0).abs() < 1e-9);
        assert_eq!(legend.note.as_deref(), Some("21.6 K under the 85 °C limit"));
    }

    #[test]
    fn without_an_assessment_the_legend_keeps_the_field_unit() {
        let mut legend = busbar_legend();
        assert_eq!(
            present(&mut legend, "K", &BTreeMap::new(), &[]).unwrap(),
            None
        );
        assert_eq!(legend.unit, "K");
        assert_eq!(legend.minimum, 329.9642170922981);
        assert!(legend.note.is_none() && legend.limit.is_none());
    }

    #[test]
    fn a_broken_limit_in_the_field_unit_is_stated_as_over() {
        let mut legend = busbar_legend();
        let assessed = [assessment(
            AssessmentRule::Maximum {
                bound: kelvin(333.15),
            },
            336.5473805835758,
        )];
        // The bound was written in kelvin: no display unit beyond the field's own.
        let units = BTreeMap::from([("stays_below_85_celsius".into(), "K".into())]);
        assert_eq!(
            present(&mut legend, "K", &units, &assessed).unwrap(),
            Some("K".into())
        );
        assert_eq!(
            legend.note.as_deref(),
            Some("3.4 K over the 333.15 K limit")
        );
    }
}
