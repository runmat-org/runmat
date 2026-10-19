use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.RecoveryPolicy.shapeFunctionInterpolation",
    summary: "Evaluate values at physical points with the element shape functions.",
    description: concat!(
        "`fea.RecoveryPolicy.shapeFunctionInterpolation()` returns the `fea.RecoveryPolicy` ",
        "whose `value` is `shape_function_interpolation`. Shape-function interpolation ",
        "evaluates node, element, or integration-point data at physical positions inside the ",
        "owning element; use it with point and path targets and a `point` output location.",
    ),
    keywords: &["fea", "observation", "recovery policy", "shape function interpolation"],
    related: &["fea.RecoveryPolicy.native", "fea.RecoveryPolicy.extrapolate", "fea.RecoveryPolicy.volumeWeightedAverage", "fea.RecoveryPolicy.areaWeightedAverage", "fea.FieldLocation.node", "fea.FieldLocation.integrationPoint"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Shape-function interpolation evaluates node, element, or integration-point data at ",
                    "physical positions inside the owning element; use it with point and path targets and",
                    " a `point` output location.",
                ),
                super::ROLE,
            ],
        ),
        section(
            "Use in a study",
            &[
                super::STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "shape-function-interpolation",
            "Evaluate values at physical points with the element shape functions",
            "recovery = fea.RecoveryPolicy.shapeFunctionInterpolation();",
            concat!(
                "assert(strcmp(class(recovery), 'fea.RecoveryPolicy'));\n",
                "assert(recovery.value == \"shape_function_interpolation\");",
            ),
        ),
    ],
    faqs: &[
        super::FAQ,
    ],
    links: &[
        page("Procedure sequences, observations, and accuracy", "/docs/simulation/study-contracts"),
        page("Outputs, fields, and observations", "/docs/simulation/language/outputs-fields-and-observations"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/study_policies/observation.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/recovery_policy/tests.rs::recovery_policy_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_RECOVERY_POLICY_SHAPE_FUNCTION_INTERPOLATION_CATALOG_ENTRY,
    name: "fea.RecoveryPolicy.shapeFunctionInterpolation",
    result: classes::RECOVERY_POLICY,
    signature: "recovery = fea.RecoveryPolicy.shapeFunctionInterpolation()",
    inputs: [
    ],
    output: ("recovery", "Typed fea.RecoveryPolicy value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
