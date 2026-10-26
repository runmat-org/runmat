use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.ExtremaTiePolicy.firstCanonical",
    summary: "Keep only the first exact tie in canonical order.",
    description: concat!(
        "`fea.ExtremaTiePolicy.firstCanonical()` returns the `fea.ExtremaTiePolicy` whose ",
        "`value` is `first_canonical`. First canonical retains the first exactly tied ",
        "location in canonical target order, independent of worker count and partition ",
        "completion order.",
    ),
    keywords: &["fea", "observation", "extrema tie policy", "first canonical"],
    related: &["fea.ExtremaTiePolicy.allLocations", "fea.ReductionPolicy.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "First canonical retains the first exactly tied location in canonical target order, ",
                    "independent of worker count and partition completion order.",
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
            "first-canonical",
            "Keep only the first exact tie in canonical order",
            concat!(
                "ties = fea.ExtremaTiePolicy.firstCanonical();\n",
                "reduction = fea.ReductionPolicy.new(fea.SpatialReduction.maximum(), ...\n",
                "    fea.SeriesReduction.none(), ties);",
            ),
            concat!(
                "assert(strcmp(class(ties), 'fea.ExtremaTiePolicy'));\n",
                "assert(ties.value == \"first_canonical\");\n",
                "assert(strcmp(class(reduction), 'fea.ReductionPolicy'));",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/extrema_tie_policy/tests.rs::extrema_tie_policy_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_EXTREMA_TIE_POLICY_FIRST_CANONICAL_CATALOG_ENTRY,
    name: "fea.ExtremaTiePolicy.firstCanonical",
    result: classes::EXTREMA_TIE_POLICY,
    signature: "ties = fea.ExtremaTiePolicy.firstCanonical()",
    inputs: [
    ],
    output: ("ties", "Typed fea.ExtremaTiePolicy value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
