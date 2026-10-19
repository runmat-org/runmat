use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SeriesReduction.absoluteMaximum",
    summary: "Find the largest magnitude over the series and when it occurs.",
    description: concat!(
        "`fea.SeriesReduction.absoluteMaximum()` returns the `fea.SeriesReduction` whose ",
        "`value` is `absolute_maximum`. Absolute maximum selects the sample with the largest ",
        "absolute value and retains its sample, coordinate, and location.",
    ),
    keywords: &["fea", "observation", "series reduction", "absolute maximum"],
    related: &["fea.SeriesReduction.none", "fea.SeriesReduction.minimum", "fea.SeriesReduction.maximum", "fea.SeriesReduction.envelope", "fea.ReductionPolicy.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Absolute maximum selects the sample with the largest absolute value and retains its ",
                    "sample, coordinate, and location.",
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
            "absolute-maximum",
            "Find the largest magnitude over the series and when it occurs",
            concat!(
                "series = fea.SeriesReduction.absoluteMaximum();\n",
                "reduction = fea.ReductionPolicy.new(fea.SpatialReduction.none(), series, ...\n",
                "    fea.ExtremaTiePolicy.firstCanonical());",
            ),
            concat!(
                "assert(strcmp(class(series), 'fea.SeriesReduction'));\n",
                "assert(series.value == \"absolute_maximum\");\n",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/series_reduction/tests.rs::series_reduction_identities_conform",
        ),
        unit_test(
            "Series statistics preserve explicit weighting",
            "crates/runmat-runtime/src/builtins/fea/study_policies/observation.rs::series_statistics_preserve_explicit_typed_weighting",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SERIES_REDUCTION_ABSOLUTE_MAXIMUM_CATALOG_ENTRY,
    name: "fea.SeriesReduction.absoluteMaximum",
    result: classes::SERIES_REDUCTION,
    signature: "series = fea.SeriesReduction.absoluteMaximum()",
    inputs: [
    ],
    output: ("series", "Typed fea.SeriesReduction value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
