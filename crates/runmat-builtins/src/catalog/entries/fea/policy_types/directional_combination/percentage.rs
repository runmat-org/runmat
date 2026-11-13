use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{
    example, faq, page, section, source, unit_test,
};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.DirectionalCombination.percentage",
    summary: "Combine directional responses with a percentage rule such as 100-40-40.",
    description: concat!(
        "`fea.DirectionalCombination.percentage(primary, secondary)` returns an ",
        "`fea.DirectionalCombination` that combines responses to simultaneous excitation ",
        "directions by taking `primary` times the governing direction plus `secondary` times ",
        "each other direction, as in the 100-40-40 or 100-30-30 rules.",
    ),
    keywords: &["fea", "response spectrum", "directional combination", "percentage"],
    related: &["fea.DirectionalCombination.srss", "fea.DirectionalCombination.absoluteSum", "fea.ResponseSpectrum.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Both values must be finite fractions in [0, 1] with `primary >= secondary`; for the ",
                    "100-40-40 rule write `percentage(1.0, 0.4)`. Invalid fractions fail with ",
                    "`RunMat:fea:InvalidInput`.",
                ),
                "The stored value is `value.percentage_rule` with fields `primary` and `secondary`.",
            ],
        ),
        section(
            "Use in a study",
            &[
                super::super::response_spectrum::SPECTRUM_STUDY,
            ],
        ),
    ],
    examples: &[
        example(
            "hundred-forty",
            "Use the 100-40-40 rule",
            "combination = fea.DirectionalCombination.percentage(1.0, 0.4);",
            concat!(
                "assert(strcmp(class(combination), 'fea.DirectionalCombination'));\n",
                "assert(combination.value.percentage_rule.secondary == 0.4);",
            ),
        ),
    ],
    faqs: &[
        faq(
            "Should I write 40 or 0.4?",
            "0.4. Both arguments are fractions between 0 and 1.",
        ),
    ],
    links: &[
        page("Build a modal response-spectrum study", "/docs/simulation/language/guides/modal-and-response-spectrum-study"),
        page("Procedure sequences, observations, and accuracy", "/docs/simulation/study-contracts"),
    ],
    implementation: &[source(
        "Runtime implementation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/response_spectrum_policies/parameters.rs",
    )],
    verification: &[
        unit_test(
            "Catalog contract conformance",
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/directional_combination/tests.rs::directional_percentage_identity_conforms",
        ),
        unit_test(
            "Typed spectrum round-trips portably",
            "crates/runmat-runtime/src/builtins/fea/response_spectra.rs::typed_spectrum_round_trips_portably",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_DIRECTIONAL_COMBINATION_PERCENTAGE_CATALOG_ENTRY,
    name: "fea.DirectionalCombination.percentage",
    result: classes::DIRECTIONAL_COMBINATION,
    signature: "combination = fea.DirectionalCombination.percentage(primary, secondary)",
    inputs: [
        (
            "primary",
            NumericScalar,
            "Fraction applied to the governing direction's response, in [0, 1].",
            PolicyArg::RealScalar,
        ),
        (
            "secondary",
            NumericScalar,
            concat!(
                "Fraction applied to each other direction's response, in [0, 1] and not greater than ",
                "primary.",
            ),
            PolicyArg::RealScalar,
        ),
    ],
    output: ("combination", "Typed fea.DirectionalCombination value."),
    integer_capabilities: &integer::PERCENTAGE_CAPABILITIES,
    integer_audit: None,
    documentation: DOCUMENTATION,
);
