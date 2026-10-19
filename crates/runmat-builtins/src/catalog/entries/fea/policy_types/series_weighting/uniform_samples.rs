use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SeriesWeighting.uniformSamples",
    summary: "Give every retained sample equal weight.",
    description: concat!(
        "`fea.SeriesWeighting.uniformSamples()` returns the `fea.SeriesWeighting` whose ",
        "`value` is `uniform_samples`. Uniform samples weights each retained sample equally, ",
        "which suits modes and genuinely uniform sample grids.",
    ),
    keywords: &["fea", "observation", "series weighting", "uniform samples"],
    related: &["fea.SeriesWeighting.trapezoidalIndependentDomain", "fea.SeriesReduction.mean", "fea.SeriesReduction.rootMeanSquare"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Uniform samples weights each retained sample equally, which suits modes and ",
                    "genuinely uniform sample grids.",
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
            "uniform-samples",
            "Give every retained sample equal weight",
            "weighting = fea.SeriesWeighting.uniformSamples();",
            concat!(
                "assert(strcmp(class(weighting), 'fea.SeriesWeighting'));\n",
                "assert(weighting.value == \"uniform_samples\");",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/series_weighting/tests.rs::series_weighting_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SERIES_WEIGHTING_UNIFORM_SAMPLES_CATALOG_ENTRY,
    name: "fea.SeriesWeighting.uniformSamples",
    result: classes::SERIES_WEIGHTING,
    signature: "weighting = fea.SeriesWeighting.uniformSamples()",
    inputs: [
    ],
    output: ("weighting", "Typed fea.SeriesWeighting value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
