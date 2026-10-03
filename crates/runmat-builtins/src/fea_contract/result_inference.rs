//! Result-query and graphics boundary facts for FEA. Stored artifact contents
//! and field presence are deliberately not fabricated from an input class.

use super::constructor_inference::host_text;
use super::identities::{
    FEA_COMPARE_CLASS, FEA_FIELD_CLASS, FEA_RESULTS_CLASS, FEA_RUN_RESULT_CLASS, FEA_STUDY_CLASS,
};
use super::options::FeaPlotOption;
use super::results_option::ResultsOption;
use crate::{
    catalog::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DimensionFact, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaResultKind {
    Results,
    Field,
    Compare,
    Plot,
}

fn possible_object(value: &ValueFact, classes: &[StaticClassIdentity]) -> bool {
    !matches!(value.residency, ResidencyFact::Device { .. })
        && !matches!(value.shape.element_count(), Some(count) if count != 1)
        && match &value.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_none_or(|class| classes.iter().any(|identity| class.is(*identity))),
            _ => false,
        }
}

fn possible_run_reference(value: &ValueFact, accept_results: bool) -> bool {
    host_text(value)
        || possible_object(
            value,
            if accept_results {
                &[FEA_RUN_RESULT_CLASS, FEA_RESULTS_CLASS]
            } else {
                &[FEA_RUN_RESULT_CLASS]
            },
        )
}

fn object_result(class: StaticClassIdentity) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn exact_numeric_class(value: &ValueFact) -> bool {
    matches!(&value.kind, ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real && number.class != NumericClass::Single)
}

fn possible_flag(value: &ValueFact) -> bool {
    !matches!(value.residency, ResidencyFact::Device { .. })
        && !matches!(value.shape.element_count(), Some(count) if count != 1)
        && (matches!(&value.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
            || exact_numeric_class(value))
}

fn possible_index_vector(value: &ValueFact) -> bool {
    if matches!(value.residency, ResidencyFact::Device { .. }) {
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
    matches!(&value.kind, ValueKindFact::Unknown) || exact_numeric_class(value)
}

fn possible_text_list(value: &ValueFact) -> bool {
    if matches!(value.residency, ResidencyFact::Device { .. }) {
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

fn possible_plot_bool(value: &ValueFact) -> bool {
    !matches!(value.residency, ResidencyFact::Device { .. })
        && !matches!(value.shape.element_count(), Some(count) if count != 1)
        && (matches!(&value.kind, ValueKindFact::Unknown | ValueKindFact::Logical)
            || matches!(&value.kind, ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                class,
            }) if *class != NumericClass::Single))
}

pub fn infer_result(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: FeaResultKind,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let result = match kind {
        FeaResultKind::Results => {
            if request.arguments.is_empty() || request.arguments.len().is_multiple_of(2) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-ARITY",
                    "fea.results requires a run reference and complete name/value pairs",
                    request.arguments.len(),
                ));
            }
            if let Some(input) = request.arguments.first() {
                if !possible_run_reference(input, true) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-RUN",
                        "fea.results requires a host run id, fea.RunResult, or fea.Results",
                        0,
                    ));
                }
                if request.arguments.len() > 1
                    && matches!(&input.kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(FEA_RESULTS_CLASS)))
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-RESULTS-OPTIONS",
                        "an existing fea.Results object is accepted only without options",
                        0,
                    ));
                }
            }
            let mut seen = BTreeSet::new();
            for index in (1..request.arguments.len()).step_by(2) {
                if !host_text(&request.arguments[index]) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-OPTION",
                        "fea.results option names must be host text",
                        index,
                    ));
                }
                if let Some(value) = request.arguments.get(index + 1) {
                    if let Some(name) = request.literals.literal_string_at(index) {
                        let normalized = ResultsOption::normalized_key(&name);
                        let option = ResultsOption::from_normalized(&normalized);
                        let canonical = match option {
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
                            Some(
                                ResultsOption::ModeIndices
                                | ResultsOption::TransientSnapshotIndices,
                            ) => (
                                possible_index_vector(value),
                                "exact one-based numeric vector",
                            ),
                            Some(
                                ResultsOption::IncludeFieldValues
                                | ResultsOption::IncludeDiagnostics
                                | ResultsOption::IncludeModalResults
                                | ResultsOption::IncludeTransientResults
                                | ResultsOption::IncludeNonlinearResults
                                | ResultsOption::IncludeElectromagneticResults,
                            ) => (possible_flag(value), "logical or exact zero/one scalar"),
                            None => (false, "supported option name"),
                        };
                        if !valid {
                            diagnostics.push(argument_error(
                                "RM-CATALOG-FEA-SELECTOR",
                                detail,
                                index + 1,
                            ));
                        }
                    } else if matches!(value.residency, ResidencyFact::Device { .. }) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-FEA-RESIDENT",
                            "fea.results selectors are not implicitly gathered",
                            index + 1,
                        ));
                    }
                }
            }
            object_result(FEA_RESULTS_CLASS)
        }
        FeaResultKind::Field => {
            if request.arguments.len() != 2 {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-ARITY",
                    "fea.field requires results or run and one field id",
                    request.arguments.len(),
                ));
            }
            if request
                .arguments
                .first()
                .is_some_and(|value| !possible_run_reference(value, true))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-RUN",
                    "fea.field requires a host run id, fea.RunResult, or fea.Results",
                    0,
                ));
            }
            if request
                .arguments
                .get(1)
                .is_some_and(|value| !host_text(value))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-FIELD-ID",
                    "fea.field id must be host text",
                    1,
                ));
            }
            object_result(FEA_FIELD_CLASS)
        }
        FeaResultKind::Compare => {
            if request.arguments.len() != 2 {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-ARITY",
                    "fea.compare requires exactly two run ids",
                    request.arguments.len(),
                ));
            }
            for (index, value) in request.arguments.iter().enumerate().take(2) {
                if !host_text(value) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-RUN-ID",
                        "fea.compare run ids must be host text",
                        index,
                    ));
                }
            }
            object_result(FEA_COMPARE_CLASS)
        }
        FeaResultKind::Plot => {
            if request.arguments.is_empty() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-ARITY",
                    "fea.plot requires a run, results, field, or study/run context",
                    0,
                ));
            }
            if request.arguments.first().is_some_and(|value| {
                !possible_object(
                    value,
                    &[
                        FEA_RUN_RESULT_CLASS,
                        FEA_RESULTS_CLASS,
                        FEA_FIELD_CLASS,
                        FEA_STUDY_CLASS,
                    ],
                )
            }) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-PLOT-CONTEXT",
                    "fea.plot requires host FEA context, not a numeric figure handle or resident field",
                    0,
                ));
            }
            let mut core_len = request.arguments.len();
            while core_len >= 2 {
                let Some(name) = request.literals.literal_string_at(core_len - 2) else {
                    break;
                };
                let Some(option) = FeaPlotOption::decode(&name) else {
                    break;
                };
                if !host_text(&request.arguments[core_len - 2]) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-PLOT-OPTION",
                        "fea.plot option names must be host text",
                        core_len - 2,
                    ));
                }
                let value = &request.arguments[core_len - 1];
                let valid = match option {
                    FeaPlotOption::Field | FeaPlotOption::Mesh | FeaPlotOption::Overlay => {
                        host_text(value)
                    }
                    FeaPlotOption::Deformed => possible_plot_bool(value),
                };
                if !valid {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-PLOT-OPTION",
                        "fea.plot option value has an incompatible host type",
                        core_len - 1,
                    ));
                }
                core_len -= 2;
            }
            // Peel every possible nonliteral trailing option pair only to
            // establish the smallest positional prefix. Its value must not
            // be diagnosed as a field id, but earlier operands remain
            // positional under every possible interpretation of the tail.
            let mut minimum_core_len = core_len;
            while minimum_core_len >= 3
                && request
                    .literals
                    .literal_string_at(minimum_core_len - 2)
                    .is_none()
                && host_text(&request.arguments[minimum_core_len - 2])
            {
                minimum_core_len -= 2;
            }
            if minimum_core_len > 3 {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-ARITY",
                    "fea.plot requires one context, optional field, or study/run/field operands",
                    core_len,
                ));
            }
            let definitely_nonstudy = request.arguments.first().is_some_and(|value| {
                matches!(&value.kind, ValueKindFact::Object(object)
                    if object.runtime_class.as_ref().is_some_and(|class| !class.is(FEA_STUDY_CLASS)))
            });
            if minimum_core_len == 3 && definitely_nonstudy {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-PLOT-CONTEXT",
                    "three positional fea.plot operands require a fea.Study first",
                    0,
                ));
            }
            for index in 1..minimum_core_len.min(3) {
                if !host_text(&request.arguments[index]) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-PLOT-FIELD",
                        "fea.plot run id and field id operands must be host text",
                        index,
                    ));
                }
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
