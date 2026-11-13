use std::collections::BTreeMap;

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, LiteralValue, NumericClass, NumericDomain, NumericFact, ObjectFact,
    ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};

fn plugin(class: StaticClassIdentity, properties: BTreeMap<String, ValueFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: true,
            // The current runtime object is value-backed, but MATLAB's
            // plugin classes are handles. Do not promise handle identity.
            handle_semantics: None,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn test_runner_plugin(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-TESTING-PLUGIN-ARITY",
            "TestRunnerPlugin accepts no inputs",
            0,
        )]
    };
    finish_fixed(
        entry,
        request,
        plugin(
            runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN,
            BTreeMap::new(),
        ),
        diagnostics,
    )
}

fn is_host_folder(value: &ValueFact) -> bool {
    if matches!(value.residency, ResidencyFact::Device { .. }) {
        return false;
    }
    match &value.kind {
        ValueKindFact::String | ValueKindFact::Unknown => true,
        ValueKindFact::Character => value
            .shape
            .known_dims()
            .is_none_or(|dims| dims.first().copied().flatten().is_none_or(|rows| rows <= 1)),
        _ => false,
    }
}

fn option_boolean(value: &ValueFact, literal: Option<&LiteralValue>) -> bool {
    if matches!(value.residency, ResidencyFact::Device { .. })
        || value.shape.element_count().is_some_and(|count| count != 1)
    {
        return false;
    }
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::Logical => true,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }) => match literal {
            Some(LiteralValue::Number(number)) => *number == 0.0 || *number == 1.0,
            Some(LiteralValue::Real { text, .. }) => text
                .parse::<f64>()
                .is_ok_and(|number| number == 0.0 || number == 1.0),
            None | Some(LiteralValue::Unknown) => true,
            _ => false,
        },
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum CoverageOption {
    IncludingSubfolders,
    Unsupported,
}

fn coverage_option(literal: Option<&LiteralValue>) -> Option<CoverageOption> {
    let name = match literal? {
        LiteralValue::String(name) | LiteralValue::Character(name) => name,
        _ => return None,
    };
    if name.eq_ignore_ascii_case("IncludingSubfolders") {
        Some(CoverageOption::IncludingSubfolders)
    } else if ["Producing", "MetricLevel", "Filter"]
        .iter()
        .any(|reserved| name.eq_ignore_ascii_case(reserved))
    {
        Some(CoverageOption::Unsupported)
    } else {
        None
    }
}

pub(super) fn code_coverage_for_folder(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-TESTING-COVERAGE-FOLDER-ARITY",
            "CodeCoveragePlugin.forFolder requires at least one folder",
            0,
        ));
    }
    let mut possible_folder = false;
    let mut index = 0;
    while index < request.arguments.len() {
        if index > 0 {
            if let Some(option) = coverage_option(request.literals.literal_args.get(index)) {
                match option {
                    CoverageOption::IncludingSubfolders => {
                        if let Some(value) = request.arguments.get(index + 1) {
                            if !option_boolean(value, request.literals.literal_args.get(index + 1))
                            {
                                diagnostics.push(argument_error(
                                    "RM-TESTING-COVERAGE-SUBFOLDERS",
                                    "IncludingSubfolders requires scalar logical or double 0/1",
                                    index + 1,
                                ));
                            }
                        } else {
                            diagnostics.push(argument_error(
                                "RM-TESTING-COVERAGE-SUBFOLDERS",
                                "IncludingSubfolders requires a value",
                                index,
                            ));
                        }
                    }
                    CoverageOption::Unsupported => diagnostics.push(argument_error(
                        "RM-TESTING-COVERAGE-UNSUPPORTED-OPTION",
                        "CodeCoveragePlugin.forFolder does not support this option",
                        index,
                    )),
                }
                index += if index + 1 < request.arguments.len() {
                    2
                } else {
                    1
                };
                continue;
            }
        }
        let folder = &request.arguments[index];
        if !is_host_folder(folder) {
            diagnostics.push(argument_error(
                "RM-TESTING-COVERAGE-FOLDER",
                "coverage folders must be host string or character-row values",
                index,
            ));
        } else if matches!(
            request.literals.literal_args.get(index),
            Some(LiteralValue::String(text) | LiteralValue::Character(text)) if text.trim().is_empty()
        ) || (matches!(&folder.kind, ValueKindFact::String)
            && folder.shape.element_count() == Some(0))
        {
            diagnostics.push(argument_error(
                "RM-TESTING-COVERAGE-FOLDER-EMPTY",
                "coverage folders must be nonempty",
                index,
            ));
        } else {
            possible_folder = true;
        }
        index += 1;
    }
    if !request.arguments.is_empty() && !possible_folder {
        diagnostics.push(argument_error(
            "RM-TESTING-COVERAGE-NO-FOLDERS",
            "CodeCoveragePlugin.forFolder requires at least one folder",
            0,
        ));
    }

    let folders = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    let subfolders = ValueFact::scalar(ValueKindFact::Logical);
    finish_fixed(
        entry,
        request,
        plugin(
            runmat_types::standard::UNIT_TEST_CODE_COVERAGE_PLUGIN,
            BTreeMap::from([
                ("Folders".into(), folders),
                ("IncludingSubfolders".into(), subfolders),
            ]),
        ),
        diagnostics,
    )
}
