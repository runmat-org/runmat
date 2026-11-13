use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SeriesWeighting.trapezoidalIndependentDomain",
    summary: "Integrate piecewise linearly over the independent domain.",
    description: concat!(
        "`fea.SeriesWeighting.trapezoidalIndependentDomain()` returns the ",
        "`fea.SeriesWeighting` whose `value` is `trapezoidal_independent_domain`. Trapezoidal",
        " weighting integrates over physical time, frequency, load factor, or a named study ",
        "parameter and preserves nonuniform spacing.",
    ),
    keywords: &["fea", "observation", "series weighting", "trapezoidal independent domain"],
    related: &["fea.SeriesWeighting.uniformSamples", "fea.SeriesReduction.mean", "fea.SeriesReduction.rootMeanSquare"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Trapezoidal weighting integrates over physical time, frequency, load factor, or a ",
                    "named study parameter and preserves nonuniform spacing. It is rejected for static ",
                    "and discrete mode domains.",
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
            "trapezoidal-independent-domain",
            "Integrate piecewise linearly over the independent domain",
            "weighting = fea.SeriesWeighting.trapezoidalIndependentDomain();",
            concat!(
                "assert(strcmp(class(weighting), 'fea.SeriesWeighting'));\n",
                "assert(weighting.value == \"trapezoidal_independent_domain\");",
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
    FEA_SERIES_WEIGHTING_TRAPEZOIDAL_INDEPENDENT_DOMAIN_CATALOG_ENTRY,
    name: "fea.SeriesWeighting.trapezoidalIndependentDomain",
    result: classes::SERIES_WEIGHTING,
    signature: "weighting = fea.SeriesWeighting.trapezoidalIndependentDomain()",
    inputs: [
    ],
    output: ("weighting", "Typed fea.SeriesWeighting value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
