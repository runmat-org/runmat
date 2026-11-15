//! A field legend in the study's own terms: the unit, fixed range and precision of the study's
//! display of the field output (`display <output> { … }`, or the unit the study writes that
//! quantity in), and the line that states the run's result against the study's limit on it.
//! The snapshot posters and the live result view both write their legend here.
use runmat_analysis_core::{AssessmentRule, AssessmentStatus, AuthoredOutputDisplay};
use runmat_plot::geometry_scene::{GeometrySceneScalarLegend, ScalarRangeSource};
use runmat_plot::legend_text::{assessment_line, AssessedResult};
use runmat_types::physical::{parse_unit_expression, UnitDefinition};
use runmat_types::PhysicalDimension;
use serde::{Deserialize, Serialize};

use crate::analysis::assessment::PublishedStudyAssessment;
use crate::analysis::AnalysisFieldLocation;

/// What a field legend is written in terms of: the study's display of the shown output (from
/// the compiled study's `displays`) and the run's published assessments.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldLegendTerms {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<AuthoredOutputDisplay>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assessments: Vec<PublishedStudyAssessment>,
}

impl FieldLegendTerms {
    /// The study's fixed colour range, canonical; the scene colours by it and clips.
    pub fn fixed_range(&self) -> Option<(f64, f64)> {
        self.display
            .as_ref()
            .and_then(|display| display.range)
            .map(|range| (range.minimum, range.maximum))
    }
}

/// One assessment bound of the shown quantity and the run's result against it.
struct Bound<'a> {
    canonical: f64,
    maximum: bool,
    measured: Option<f64>,
    /// The run field the assessed result reads (G77 evidence), when it reads one.
    source_field_id: Option<&'a str>,
}

/// The field a legend colours: its run field id and where its values sit.
#[derive(Debug, Clone, Copy)]
pub struct ShownField<'a> {
    pub field_id: &'a str,
    pub location: AnalysisFieldLocation,
}

/// How a location is named on a legend: after the assessed result ("Peak 76.3 MPa (element)")
/// and after the label of a view of another location ("Von Mises stress, nodal average").
fn location_names(location: AnalysisFieldLocation) -> Option<(&'static str, &'static str)> {
    match location {
        AnalysisFieldLocation::Element => Some(("element", "element")),
        AnalysisFieldLocation::Node => Some(("node", "nodal average")),
        _ => None,
    }
}

impl Bound<'_> {
    /// Signed distance inside the bound, canonical: negative when the result breaks it.
    fn margin(&self) -> Option<f64> {
        self.measured.map(|measured| {
            if self.maximum {
                self.canonical - measured
            } else {
                measured - self.canonical
            }
        })
    }
}

fn bounds_of<'a>(
    assessment: &'a PublishedStudyAssessment,
    field: &UnitDefinition,
) -> Vec<Bound<'a>> {
    let source_field_id = assessment
        .evidence
        .as_ref()
        .and_then(|evidence| evidence.source_field_id.as_deref());
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
        .filter_map(|(bound, maximum)| {
            let bound = bound?;
            (bound.descriptor().dimension() == field.dimension()).then(|| Bound {
                canonical: *bound.canonical_values(),
                maximum,
                measured,
                source_field_id,
            })
        })
        .collect()
}

/// Write `legend` (built in `field_unit`) in the display unit and precision of `terms`, convert
/// its range, overflow values and limit to that unit, and state the result against the tightest
/// assessment bound of the shown quantity ([`assessment_line`]). Returns the display unit when
/// it differs from `field_unit`.
///
/// `drawn` is the coloured field and `locate` gives a run field's location. The assessment line
/// names where its result is evaluated (the location of the field the assessment reads); when
/// the legend colours another location of the quantity, its label names that location too.
pub fn present(
    legend: &mut GeometrySceneScalarLegend,
    field_unit: &str,
    terms: &FieldLegendTerms,
    drawn: Option<ShownField<'_>>,
    locate: &dyn Fn(&str) -> Option<AnalysisFieldLocation>,
) -> Result<Option<String>, String> {
    let Ok(field) = parse_unit_expression(field_unit) else {
        return Ok(None);
    };
    let display = terms.display.as_ref();
    let (symbol, unit) = display
        .and_then(|display| display.unit.as_deref())
        .and_then(|symbol| {
            let unit = parse_unit_expression(symbol).ok()?;
            (unit.dimension() == field.dimension()).then(|| (symbol.to_string(), unit))
        })
        .unwrap_or_else(|| (field_unit.to_string(), field));
    let shown = |value: f64| -> Result<f64, String> {
        let canonical = field.to_canonical(value).map_err(|e| e.to_string())?;
        unit.from_canonical(canonical).map_err(|e| e.to_string())
    };
    legend.minimum = shown(legend.minimum)?;
    legend.maximum = shown(legend.maximum)?;
    legend.below = legend.below.map(shown).transpose()?;
    legend.above = legend.above.map(shown).transpose()?;
    legend.unit = symbol.clone();
    legend.precision = display.and_then(|display| display.precision);
    // The bound the run comes closest to (or breaks furthest); without a result, the first.
    let tightest = terms
        .assessments
        .iter()
        .flat_map(|assessment| bounds_of(assessment, &field))
        .min_by(|a, b| {
            let key = |bound: &Bound| bound.margin().unwrap_or(f64::INFINITY);
            key(a).total_cmp(&key(b))
        });
    if let Some(bound) = tightest {
        let in_unit = |canonical: f64| unit.from_canonical(canonical).map_err(|e| e.to_string());
        let limit = in_unit(bound.canonical)?;
        legend.limit = Some(limit);
        let assessed = bound.source_field_id.and_then(|id| Some((id, locate(id)?)));
        if let (Some(drawn), Some((assessed_id, assessed_location))) = (drawn, assessed) {
            if drawn.field_id != assessed_id && drawn.location != assessed_location {
                if let Some((_, label)) = location_names(drawn.location) {
                    legend.label = format!("{}, {label}", legend.label);
                }
            }
        }
        legend.note = Some(assessment_line(&AssessedResult {
            maximum: bound.maximum,
            measured: bound.measured.map(in_unit).transpose()?,
            limit,
            unit: &symbol,
            // A ratio of temperatures depends on the scale's zero; state the margin instead.
            ratio: field.dimension() != PhysicalDimension::TEMPERATURE,
            digits: legend.precision.unwrap_or(3),
            location: assessed
                .and_then(|(_, location)| location_names(location))
                .map(|(name, _)| name),
        }));
    }
    Ok((symbol != field_unit).then_some(symbol))
}

/// The colour scale of a field scene.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScalarScale {
    /// The field's true extremes when known, such as [`super::extrema::field_extrema`]; the drawn
    /// samples widen them. Without them the drawn samples give them.
    pub extremes: Option<(f64, f64)>,
    /// A fixed colour range the study sets (`display <output> { range = … }`), canonical. The
    /// legend shows it, values outside it take the end colours, and the legend states the true
    /// extremes beyond it ([`GeometrySceneScalarLegend::above`] and `below`).
    pub fixed: Option<(f64, f64)>,
}

/// Mark `legend`, drawn over a study's fixed range, as such and state the field's true
/// `extremes` that lie outside it (canonical, like the legend's range at this point).
pub fn mark_fixed_range(legend: &mut GeometrySceneScalarLegend, extremes: (f64, f64)) {
    legend.range_source = ScalarRangeSource::Study;
    legend.below = (extremes.0 < legend.minimum).then_some(extremes.0);
    legend.above = (extremes.1 > legend.maximum).then_some(extremes.1);
}

impl ScalarScale {
    /// The scale over the field's `extremes` (when known), with no fixed range.
    pub fn of_extremes(extremes: Option<(f64, f64)>) -> Self {
        Self {
            extremes,
            fixed: None,
        }
    }
}

impl super::SceneStats {
    /// These stats with the study's fixed range from `terms`.
    pub(super) fn with_terms(mut self, terms: &FieldLegendTerms) -> Self {
        self.scale.fixed = terms.fixed_range();
        self.fixed_deformation = terms
            .display
            .as_ref()
            .and_then(|display| display.deformation);
        self
    }
}

/// [`present`] the scalar legend of a scene's `overlay`, drawn in `field_unit`, when it has one.
pub fn present_overlay(
    overlay: &mut runmat_plot::geometry_scene::GeometrySceneOverlay,
    field_unit: Option<&str>,
    terms: &FieldLegendTerms,
    shown: Option<ShownField<'_>>,
    locate: &dyn Fn(&str) -> Option<AnalysisFieldLocation>,
) -> Result<(), String> {
    if let (Some(legend), Some(unit)) = (overlay.scalar_legend.as_mut(), field_unit) {
        present(legend, unit, terms, shown, locate)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
