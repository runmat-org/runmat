//! Result-query and graphics boundary facts for FEA. Stored artifact contents
//! and field presence are deliberately not fabricated from an input class.

use super::constructor_inference::{
    host_text, known_class, object_fact, possible_exact_scalar, possible_object, resident,
};
use super::identities::{
    FEA_COMPARE_CLASS, FEA_FIELD_CLASS, FEA_FIELD_PAGE_CLASS, FEA_FIELD_SELECTOR_CLASSES,
    FEA_FREQUENCY_CLASS, FEA_RESULTS_CLASS, FEA_RUN_RESULT_CLASS, FEA_STEP_RUN_RESULT_CLASS,
    FEA_TIME_CLASS,
};
use super::options::{FeaMeshOption, FeaPlotOption};
use super::results_option::ResultsOption;
use crate::{
    catalog::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DimensionFact, InferenceDiagnostic, NumericClass, NumericDomain,
    NumericFact, ShapeFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaResultKind {
    /// `fea.results(runOrRunId, Name, Value, ...)`
    Results,
    /// `fea.field(resultsOrRun, selector, sample)`
    Field,
    /// `fea.fieldPage(field, start, count)`
    FieldPage,
    /// `fea.stepResult(run, stepId)`
    StepResult,
    /// `fea.compare(baselineRunId, candidateRunId)`
    Compare,
    /// `plot(field, Name, Value, ...)` overload
    Plot,
    /// `mesh(runOrResultsOrField, Name, Value, ...)` overload
    Mesh,
}

/// Run references accepted by `fea.results` and `fea.field`: a persisted run id,
/// a single-step `fea.RunResult`, a `fea.StepRunResult`, or (where allowed)
/// an existing `fea.Results`.
fn possible_run_reference(value: &ValueFact, accept_results: bool) -> bool {
    host_text(value)
        || possible_object(
            value,
            if accept_results {
                &[
                    FEA_RUN_RESULT_CLASS,
                    FEA_STEP_RUN_RESULT_CLASS,
                    FEA_RESULTS_CLASS,
                ]
            } else {
                &[FEA_RUN_RESULT_CLASS, FEA_STEP_RUN_RESULT_CLASS]
            },
        )
}

fn possible_index_vector(value: &ValueFact) -> bool {
    if resident(value) {
        return false;
    }
    if let ShapeFact::Shaped { dims } = &value.shape {
        if dims
            .iter()
            .filter(|dim| matches!(dim, DimensionFact::Known(size) if *size > 1))
            .count()
            > 1
        {
            return false;
        }
    }
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(number) => {
            number.domain == NumericDomain::Real && number.class != NumericClass::Single
        }
        _ => false,
    }
}

fn possible_text_list(value: &ValueFact) -> bool {
    if resident(value) {
        return false;
    }
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::String => true,
        ValueKindFact::Character => host_text(value),
        ValueKindFact::Cell(cell) => {
            host_text(&cell.element) && cell.elements.iter().all(host_text)
        }
        _ => false,
    }
}

fn arity(
    diagnostics: &mut Vec<InferenceDiagnostic>,
    request: &CallRequest,
    valid: bool,
    message: &str,
) {
    if !valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            message,
            request.arguments.len(),
        ));
    }
}

fn infer_results_options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let mut seen = BTreeSet::new();
    for index in (1..request.arguments.len()).step_by(2) {
        if !host_text(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-OPTION",
                "fea.results option names must be host text",
                index,
            ));
        }
        let Some(value) = request.arguments.get(index + 1) else {
            continue;
        };
        let Some(name) = request.literals.literal_string_at(index) else {
            if resident(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-RESIDENT",
                    "fea.results selectors are not implicitly gathered",
                    index + 1,
                ));
            }
            continue;
        };
        let normalized = ResultsOption::normalized_key(&name);
        let option = ResultsOption::from_normalized(&normalized);
        let canonical: &str = match option {
            Some(option) => option.canonical_name(),
            None => normalized.as_str(),
        };
        if !seen.insert(canonical.to_owned()) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-DUPLICATE",
                "duplicate fea.results option",
                index,
            ));
        }
        let (valid, detail) = match option {
            Some(ResultsOption::IncludeFields | ResultsOption::DiagnosticCodes) => {
                (possible_text_list(value), "text selector list")
            }
            Some(ResultsOption::ModeIndices | ResultsOption::TransientSnapshotIndices) => (
                possible_index_vector(value),
                "exact one-based numeric vector",
            ),
            Some(
                ResultsOption::IncludeFieldValues
                | ResultsOption::IncludeDiagnostics
                | ResultsOption::IncludeModalResults
                | ResultsOption::IncludeTransientResults
                | ResultsOption::IncludeNonlinearResults,
            ) => (
                possible_exact_scalar(value),
                "logical or exact zero/one scalar",
            ),
            None => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-OPTION",
                    "unsupported fea.results option",
                    index,
                ));
                continue;
            }
        };
        if !valid {
            diagnostics.push(argument_error("RM-CATALOG-FEA-SELECTOR", detail, index + 1));
        }
    }
}

/// Peel trailing literal display options; returns the positional prefix length.
fn peel_display_options(
    request: &CallRequest,
    kind: FeaResultKind,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> usize {
    let mut end = request.arguments.len();
    while end >= 2 {
        let Some(name) = request.literals.literal_string_at(end - 2) else {
            break;
        };
        let value = &request.arguments[end - 1];
        let valid = match kind {
            FeaResultKind::Plot => match FeaPlotOption::decode(&name) {
                Some(FeaPlotOption::Overlay) => host_text(value),
                Some(FeaPlotOption::Edges | FeaPlotOption::Deformed) => {
                    possible_exact_scalar(value)
                }
                None => break,
            },
            _ => match FeaMeshOption::decode(&name) {
                Some(FeaMeshOption::Source) => host_text(value),
                Some(FeaMeshOption::Edges) => possible_exact_scalar(value),
                None => break,
            },
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-DISPLAY-OPTION",
                "FEA display option value has an incompatible host type",
                end - 1,
            ));
        }
        end -= 2;
    }
    end
}

pub fn infer_result(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: FeaResultKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let arguments = &request.arguments;
    let result = match kind {
        FeaResultKind::Results => {
            arity(
                &mut diagnostics,
                request,
                !arguments.is_empty() && !arguments.len().is_multiple_of(2),
                "fea.results requires a run reference and complete Name, Value pairs",
            );
            if let Some(input) = arguments.first() {
                if !possible_run_reference(input, true) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-RUN",
                        "fea.results requires a run id, fea.RunResult, fea.StepRunResult, or fea.Results",
                        0,
                    ));
                }
                if arguments.len() > 1 && known_class(input, FEA_RESULTS_CLASS) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-RESULTS-OPTIONS",
                        "an existing fea.Results object is accepted only without options",
                        0,
                    ));
                }
            }
            infer_results_options(request, &mut diagnostics);
            object_fact(FEA_RESULTS_CLASS)
        }
        FeaResultKind::Field => {
            arity(
                &mut diagnostics,
                request,
                (2..=3).contains(&arguments.len()),
                "fea.field requires results or run input, a typed field selector, and an optional typed sample",
            );
            if arguments
                .first()
                .is_some_and(|value| !possible_run_reference(value, true))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-RUN",
                    "fea.field requires fea.Results, fea.RunResult, fea.StepRunResult, or a run id",
                    0,
                ));
            }
            if arguments
                .get(1)
                .is_some_and(|value| !possible_object(value, &FEA_FIELD_SELECTOR_CLASSES))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-FIELD-SELECTOR",
                    "fea.field requires a typed field selector such as fea.StructuralField.vonMises(); text field ids are not accepted",
                    1,
                ));
            }
            if arguments.get(2).is_some_and(|value| {
                !possible_object(value, &[FEA_FREQUENCY_CLASS, FEA_TIME_CLASS])
                    && !possible_exact_scalar(value)
            }) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-SAMPLE",
                    "fea.field sample must be a fea.Frequency, a fea.Time, or a one-based transient sample index",
                    2,
                ));
            }
            object_fact(FEA_FIELD_CLASS)
        }
        FeaResultKind::FieldPage => {
            arity(
                &mut diagnostics,
                request,
                (1..=3).contains(&arguments.len()),
                "fea.fieldPage requires a field and optional one-based start and count",
            );
            if arguments
                .first()
                .is_some_and(|value| !possible_object(value, &[FEA_FIELD_CLASS]))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-FIELD",
                    "fea.fieldPage requires an fea.Field",
                    0,
                ));
            }
            for (index, argument) in arguments.iter().enumerate().take(3).skip(1) {
                if !possible_exact_scalar(argument) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-PAGE-BOUND",
                        "field page start and count must be host numeric scalars",
                        index,
                    ));
                }
                if request
                    .literals
                    .numeric_at(index)
                    .is_some_and(|value| !value.is_finite() || value < 1.0 || value.fract() != 0.0)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-PAGE-BOUND",
                        "field page start and count must be positive integers",
                        index,
                    ));
                }
            }
            object_fact(FEA_FIELD_PAGE_CLASS)
        }
        FeaResultKind::StepResult => {
            arity(
                &mut diagnostics,
                request,
                arguments.len() == 2,
                "fea.stepResult requires an aggregate run and one step id",
            );
            if arguments
                .first()
                .is_some_and(|value| !possible_object(value, &[FEA_RUN_RESULT_CLASS]))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-RUN",
                    "fea.stepResult requires the aggregate fea.RunResult returned by fea.run",
                    0,
                ));
            }
            if arguments.get(1).is_some_and(|value| !host_text(value)) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-TEXT",
                    "fea.stepResult step id must be host text",
                    1,
                ));
            }
            object_fact(FEA_STEP_RUN_RESULT_CLASS)
        }
        FeaResultKind::Compare => {
            arity(
                &mut diagnostics,
                request,
                arguments.len() == 2,
                "fea.compare requires exactly two run ids",
            );
            for (index, value) in arguments.iter().enumerate().take(2) {
                if !host_text(value) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-RUN-ID",
                        "fea.compare run ids must be host text",
                        index,
                    ));
                }
            }
            object_fact(FEA_COMPARE_CLASS)
        }
        FeaResultKind::Plot | FeaResultKind::Mesh => {
            let core = peel_display_options(request, kind, &mut diagnostics);
            arity(
                &mut diagnostics,
                request,
                core == 1,
                "FEA plot and mesh overloads require one FEA context followed by display Name, Value options",
            );
            let classes: &[_] = if kind == FeaResultKind::Plot {
                &[FEA_FIELD_CLASS]
            } else {
                &[
                    FEA_RUN_RESULT_CLASS,
                    FEA_STEP_RUN_RESULT_CLASS,
                    FEA_RESULTS_CLASS,
                    FEA_FIELD_CLASS,
                ]
            };
            if arguments
                .first()
                .is_some_and(|value| !possible_object(value, classes))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-PLOT-CONTEXT",
                    if kind == FeaResultKind::Plot {
                        "the FEA plot overload requires an fea.Field"
                    } else {
                        "the FEA mesh overload requires a run, results, or field with study context"
                    },
                    0,
                ));
            }
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        }
    };
    finish_fixed(entry, request, result, diagnostics)
}

#[cfg(test)]
mod tests;
