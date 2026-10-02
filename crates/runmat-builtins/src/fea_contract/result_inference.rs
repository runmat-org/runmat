//! Result-query and graphics boundary facts for FEA. Stored artifact contents
//! and field presence are deliberately not fabricated from an input class.

use super::constructor_inference::host_text;
use super::identities::{
    FEA_COMPARE_CLASS, FEA_FIELD_CLASS, FEA_RESULTS_CLASS, FEA_RUN_RESULT_CLASS, FEA_STUDY_CLASS,
};
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ObjectFact,
    ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeaResultKind {
    Results,
    Field,
    Compare,
    Plot,
}

fn possible_object(value: &ValueFact, classes: &[StaticClassIdentity]) -> bool {
    !matches!(value.residency, ResidencyFact::Device { .. })
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
            }
            for index in (1..request.arguments.len()).step_by(2) {
                if !host_text(&request.arguments[index]) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-OPTION",
                        "fea.results option names must be host text",
                        index,
                    ));
                }
                if request
                    .arguments
                    .get(index + 1)
                    .is_some_and(|value| matches!(value.residency, ResidencyFact::Device { .. }))
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-FEA-RESIDENT",
                        "fea.results selectors are not implicitly gathered",
                        index + 1,
                    ));
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
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        }
    };
    finish_fixed(entry, request, result, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fea_contract::{
        integer::FEA_FIELD_DESCRIPTOR,
        operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    };
    use crate::{
        BuiltinAcceleratorPolicy, BuiltinCatalogProvenance, BuiltinDistributedPolicy,
        BuiltinDocumentation,
    };
    use runmat_types::{
        DynamicReason, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
    };

    fn entry() -> BuiltinCatalogEntry {
        operation_entry::entry(OperationEntryDefinition {
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            name: "fea.field",
            documentation: BuiltinDocumentation::EMPTY,
            descriptor: &FEA_FIELD_DESCRIPTOR,
            inference: |request, entry| infer_result(request, entry, FeaResultKind::Field),
            integer_capabilities: &[],
            integer_audit: None,
            effects: OperationEffectProfile::ArtifactRead,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            distributed: BuiltinDistributedPolicy::Unsupported,
        })
    }

    #[test]
    fn result_references_do_not_accept_numeric_or_resident_values() {
        let number = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        assert!(!possible_run_reference(&number, true));
        let mut unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        assert!(possible_run_reference(&unknown, true));
        unknown.residency = ResidencyFact::Device { provider: None };
        assert!(!possible_run_reference(&unknown, true));

        let inferred = infer_result(
            &CallRequest {
                arguments: vec![number, ValueFact::scalar(ValueKindFact::String)],
                literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            },
            &entry(),
            FeaResultKind::Field,
        );
        assert!(inferred
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-FEA-RUN"));
        assert!(
            matches!(&inferred.outputs[0].kind, ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(FEA_FIELD_CLASS)))
        );
    }
}
