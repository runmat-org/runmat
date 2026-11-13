use super::super::distance_geometry::matrix_size_for_condensed_length;
use super::super::pairwise_distance::DistanceMetricKind;
use super::{LinkageMethod, LinkageOption, SaveMemorySetting};
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINKAGE-ARITY",
            "linkage requires observations or condensed distances",
            0,
        ));
    }
    let observations = request
        .arguments
        .first()
        .and_then(|value| observation_count(value, &mut diagnostics));
    let controls = controls(request, &mut diagnostics);
    if observations.is_some_and(|count| count > 700) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINKAGE-RESOURCE",
            "current in-memory linkage implementation admits at most 700 observations",
            0,
        ));
    }
    let condensed_input = request
        .arguments
        .first()
        .is_some_and(|value| definitely_condensed(&value.shape));
    if controls.condensed_metric && condensed_input {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINKAGE-CONDENSED-METRIC",
            "condensed distances do not accept a distance metric",
            2,
        ));
    }
    if controls.save_memory == Some(SaveMemorySetting::On) {
        if condensed_input {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINKAGE-SAVE-MEMORY-INPUT",
                "SaveMemory on requires observation-matrix input",
                0,
            ));
        }
        if controls
            .method
            .is_some_and(|method| !method.requires_euclidean_distances())
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINKAGE-SAVE-MEMORY-METHOD",
                "SaveMemory on requires centroid, median, or ward linkage",
                1,
            ));
        }
        if controls
            .metric
            .is_some_and(|metric| metric != DistanceMetricKind::Euclidean)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINKAGE-SAVE-MEMORY-METRIC",
                "SaveMemory on requires Euclidean distance",
                2,
            ));
        }
    }
    let output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![
            observations.and_then(|count| count.checked_sub(1)),
            Some(3),
        ]),
        StorageFact::Dense,
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn definitely_condensed(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Shaped { dims } if dims.len() == 1 || matches!(dims.first(), Some(DimensionFact::Known(1))))
}

fn observation_count(
    value: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<usize> {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINKAGE-CLASS",
            "linkage input must be a real numeric or logical array",
            0,
        ));
    }
    let dims = match &value.shape {
        ShapeFact::Shaped { dims } if dims.len() <= 2 => dims,
        ShapeFact::Shaped { .. } => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINKAGE-RANK",
                "linkage input must be a vector or two-dimensional matrix",
                0,
            ));
            return None;
        }
        ShapeFact::Scalar => {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINKAGE-OBSERVATIONS",
                "linkage requires at least two observations",
                0,
            ));
            return Some(1);
        }
        ShapeFact::Ranked { .. } | ShapeFact::Unknown => return None,
    };
    let rows = dims.first().and_then(known_dimension);
    let columns = dims.get(1).and_then(known_dimension);
    let condensed_length = if dims.len() == 1 {
        rows
    } else if rows == Some(1) {
        columns
    } else {
        None
    };
    if let Some(length) = condensed_length {
        let count = matrix_size_for_condensed_length(length).filter(|count| *count >= 2);
        if count.is_none() {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINKAGE-CONDENSED-SHAPE",
                "condensed distance length must equal n*(n-1)/2 for n>=2",
                0,
            ));
        }
        return count;
    }
    if rows.is_some_and(|rows| rows < 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINKAGE-OBSERVATIONS",
            "linkage requires at least two observation rows",
            0,
        ));
    }
    rows
}

fn known_dimension(dimension: &DimensionFact) -> Option<usize> {
    match dimension {
        DimensionFact::Known(value) => Some(*value),
        _ => None,
    }
}

#[derive(Default)]
struct KnownControls {
    method: Option<LinkageMethod>,
    metric: Option<DistanceMetricKind>,
    condensed_metric: bool,
    save_memory: Option<SaveMemorySetting>,
}

fn controls(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) -> KnownControls {
    let mut known = KnownControls {
        method: Some(LinkageMethod::Single),
        ..Default::default()
    };
    let mut index = 1;
    if let Some(first) = request.arguments.get(index) {
        if let Some(text) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        {
            if LinkageOption::from_keyword(text).is_none() {
                known.method = LinkageMethod::from_keyword(text);
                if known.method.is_none() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-LINKAGE-METHOD",
                        "unsupported linkage method",
                        index,
                    ));
                }
                index += 1;
            }
        } else if !matches!(
            first.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINKAGE-METHOD-TYPE",
                "linkage method must be text",
                index,
            ));
            index += 1;
        } else {
            known.method = None;
            index += 1;
        }
    }
    while index < request.arguments.len() {
        let literal = request.literals.literal_args.get(index);
        let text = literal.and_then(literal_text);
        if text.and_then(LinkageOption::from_keyword) == Some(LinkageOption::SaveMemory) {
            let setting = request
                .literals
                .literal_args
                .get(index + 1)
                .and_then(literal_text)
                .and_then(SaveMemorySetting::from_keyword);
            if index + 1 >= request.arguments.len()
                || (request
                    .literals
                    .literal_args
                    .get(index + 1)
                    .and_then(literal_text)
                    .is_some()
                    && setting.is_none())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LINKAGE-SAVE-MEMORY",
                    "SaveMemory requires on or off",
                    index,
                ));
            }
            known.save_memory = setting;
            index += 2;
            continue;
        }
        known.condensed_metric = true;
        if let Some(text) = text {
            known.metric = DistanceMetricKind::from_keyword(text);
            if known.metric.is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LINKAGE-METRIC",
                    "unsupported named distance metric",
                    index,
                ));
            }
        }
        index += 1;
    }
    known
}

fn literal_text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::String(value)
        | LiteralValue::Character(value)
        | LiteralValue::Keyword(value) => Some(value),
        _ => None,
    }
}
