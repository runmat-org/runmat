use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SeriesReduction.rootMeanSquare",
    summary: "Reduce a history to its root mean square with explicit sample weighting.",
    description: concat!(
        "`fea.SeriesReduction.rootMeanSquare(weighting)` returns an `fea.SeriesReduction` ",
        "whose `value.root_mean_square.weighting` names how samples are weighted. ",
        "Root-mean-square is the square root of the weighted mean of the squared samples.",
    ),
    keywords: &["fea", "observation", "series reduction", "root mean square", "weighting"],
    related: &["fea.SeriesWeighting.uniformSamples", "fea.SeriesWeighting.trapezoidalIndependentDomain", "fea.ReductionPolicy.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Root-mean-square is the square root of the weighted mean of the squared samples. ",
                    "`weighting` must be a typed `fea.SeriesWeighting`; other values fail with ",
                    "`RunMat:fea:InvalidInput`. Statistics other than the complex mean require an ",
                    "explicit real, imaginary, magnitude, or phase representation.",
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
            "root-mean-square",
            "Take a time-weighted root mean square",
            concat!(
                "series = fea.SeriesReduction.rootMeanSquare(fea.SeriesWeighting.trapezoidalIndependentDomain());\n",
                "reduction = fea.ReductionPolicy.new(fea.SpatialReduction.none(), series, ...\n",
                "    fea.ExtremaTiePolicy.firstCanonical());",
            ),
            concat!(
                "assert(strcmp(class(series), 'fea.SeriesReduction'));\n",
                "assert(series.value.root_mean_square.weighting == \"trapezoidal_independent_domain\");\n",
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
    FEA_SERIES_REDUCTION_ROOT_MEAN_SQUARE_CATALOG_ENTRY,
    name: "fea.SeriesReduction.rootMeanSquare",
    result: classes::SERIES_REDUCTION,
    signature: "series = fea.SeriesReduction.rootMeanSquare(weighting)",
    inputs: [
        (
            "weighting",
            Any,
            "Typed fea.SeriesWeighting over the independent-domain samples.",
            PolicyArg::Typed(&[classes::SERIES_WEIGHTING]),
        ),
    ],
    output: ("series", "Typed fea.SeriesReduction value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
