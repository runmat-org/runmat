//! The line that states an assessed result against its limit in the study's terms, such as
//! "Peak 76.3 MPa (element) · limit 184 MPa · factor 2.41". Publication writes it on each
//! published assessment ([`PublishedStudyAssessment::summary_line`]); a field legend writes the
//! same line for the bound it draws. Both build it here, and the text itself comes from
//! [`runmat_types::physical::result_text::assessment_line`].
use super::PublishedStudyAssessment;
use crate::analysis::AnalysisFieldLocation;
use runmat_analysis_core::{
    AssessmentRule, AssessmentStatus, AuthoredOutputDisplay, DisplayUnitSource, FieldLocation,
    LoweredAuthoredObservation, ObservationSpec,
};
use runmat_types::physical::result_text::{assessment_line, AssessedResult};
use runmat_types::physical::{dimension_requirement, parse_unit_expression, UnitDefinition};
use runmat_types::PhysicalDimension;

/// One bound of an assessment and the run's result against it, canonical.
pub(crate) struct Bound<'a> {
    pub canonical: f64,
    pub maximum: bool,
    pub measured: Option<f64>,
    /// The run field the assessed result reads (G77 evidence), when it reads one. Only the
    /// field legend reads it, and the legend exists only with `plot-core`.
    #[cfg_attr(not(feature = "plot-core"), allow(dead_code))]
    pub source_field_id: Option<&'a str>,
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

/// The bounds of `assessment` in `dimension`. A function rule has none; an indeterminate result
/// states only its limit.
pub(crate) fn bounds_of(
    assessment: &PublishedStudyAssessment,
    dimension: PhysicalDimension,
) -> Vec<Bound<'_>> {
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
            (bound.descriptor().dimension() == dimension).then(|| Bound {
                canonical: *bound.canonical_values(),
                maximum,
                measured,
                source_field_id,
            })
        })
        .collect()
}

/// The bound the run comes closest to (or breaks furthest); without a result, the first.
pub(crate) fn tightest<'a>(bounds: impl IntoIterator<Item = Bound<'a>>) -> Option<Bound<'a>> {
    bounds.into_iter().min_by(|a, b| {
        let key = |bound: &Bound| bound.margin().unwrap_or(f64::INFINITY);
        key(a).total_cmp(&key(b))
    })
}

/// How a location is named: after the assessed result ("Peak 76.3 MPa (element)") and after
/// the label of a legend that colours another location ("Von Mises stress, nodal average").
pub(crate) fn location_names(
    location: AnalysisFieldLocation,
) -> Option<(&'static str, &'static str)> {
    match location {
        AnalysisFieldLocation::Element => Some(("element", "element")),
        AnalysisFieldLocation::Node => Some(("node", "nodal average")),
        _ => None,
    }
}

/// The limit of `bound` in `unit` (spelled `symbol`) and the line that states the result
/// against it with `digits` significant digits, naming `location` after the result.
pub(crate) fn bound_line(
    bound: &Bound<'_>,
    symbol: &str,
    unit: &UnitDefinition,
    digits: u8,
    location: Option<&str>,
) -> Result<(f64, String), String> {
    let in_unit = |canonical: f64| unit.from_canonical(canonical).map_err(|e| e.to_string());
    let limit = in_unit(bound.canonical)?;
    let line = assessment_line(&AssessedResult {
        maximum: bound.maximum,
        measured: bound.measured.map(in_unit).transpose()?,
        limit,
        unit: symbol,
        // A ratio of temperatures depends on the scale's zero; state the margin instead.
        ratio: unit.dimension() != PhysicalDimension::TEMPERATURE,
        digits,
        location,
    });
    Ok((limit, line))
}

/// The line publication writes on `assessment`, in the terms of the field it reads: the unit and
/// precision of that field output's display (`display <output> { unit; precision }`), else the
/// unit the study writes that quantity in, else SI; and the location of the assessed result.
/// `None` for a function rule and for a quantity with no printable unit.
///
/// The display chosen is the one of the field output with the assessed observation's step,
/// quantity and component, preferring the same location. A field legend of that output writes
/// the same line.
pub(crate) fn published_line(
    assessment: &PublishedStudyAssessment,
    producer: &LoweredAuthoredObservation,
    observations: &[LoweredAuthoredObservation],
    displays: &[AuthoredOutputDisplay],
) -> Option<String> {
    let (minimum, maximum) = assessment.definition.rule.bounds();
    let dimension = minimum.or(maximum)?.descriptor().dimension();
    let spec = producer.request.spec();
    let location = spec.map(assessed_location);
    let in_dimension = |display: &&AuthoredOutputDisplay| {
        let symbol = display.unit.as_deref()?;
        let unit = parse_unit_expression(symbol).ok()?;
        (unit.dimension() == dimension).then(|| (symbol.to_string(), unit))
    };
    let field_display = spec.and_then(|spec| {
        let reads = |display: &AuthoredOutputDisplay, same_location: bool| {
            observations.iter().any(|output| {
                output.request.observation_id().0 == display.output_id
                    && output.step_id == producer.step_id
                    && output.request.spec().is_some_and(|field| {
                        field.quantity == spec.quantity
                            && field.component == spec.component
                            && (!same_location
                                || Some(AnalysisFieldLocation::from(field.output_location))
                                    == location)
                    })
            })
        };
        displays
            .iter()
            .find(|display| reads(display, true))
            .or_else(|| displays.iter().find(|display| reads(display, false)))
    });
    let (symbol, unit, digits) = match field_display {
        Some(display) => {
            let (symbol, unit) = in_dimension(&display).or_else(|| si_unit(dimension))?;
            (symbol, unit, display.precision)
        }
        None => {
            let (symbol, unit) = displays
                .iter()
                .filter(|display| display.unit_source == DisplayUnitSource::Study)
                .find_map(|display| in_dimension(&display))
                .or_else(|| si_unit(dimension))?;
            (symbol, unit, None)
        }
    };
    let bound = tightest(bounds_of(assessment, dimension))?;
    let location = location.and_then(location_names).map(|(name, _)| name);
    bound_line(&bound, &symbol, &unit, digits.unwrap_or(3), location)
        .ok()
        .map(|(_, line)| line)
}

/// Where the assessed observation's result is evaluated: the location of the field it reduces
/// (after any recovery), which an extremum keeps and any other reduction replaces by `global`.
fn assessed_location(spec: &ObservationSpec) -> AnalysisFieldLocation {
    let location = match spec.output_location {
        FieldLocation::Global => spec.source_location,
        location => location,
    };
    location.into()
}

/// The coherent SI unit of `dimension` and its definition.
fn si_unit(dimension: PhysicalDimension) -> Option<(String, UnitDefinition)> {
    let (_, symbol) = dimension_requirement(dimension)?;
    let unit = parse_unit_expression(symbol).ok()?;
    Some((if symbol == "1" { "" } else { symbol }.to_string(), unit))
}
