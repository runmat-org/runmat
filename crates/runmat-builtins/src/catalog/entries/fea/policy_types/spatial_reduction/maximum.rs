use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpatialReduction.maximum",
    summary: "Find the maximum value over the target and where it occurs.",
    description: concat!(
        "`fea.SpatialReduction.maximum()` returns the `fea.SpatialReduction` whose `value` is",
        " `maximum`. Maximum retains the canonical node, element, or interpolated position ",
        "where the maximum occurs; exact ties follow the extrema-tie policy.",
    ),
    keywords: &["fea", "observation", "spatial reduction", "maximum"],
    related: &["fea.SpatialReduction.none", "fea.SpatialReduction.integral", "fea.SpatialReduction.average", "fea.SpatialReduction.rootMeanSquare", "fea.ReductionPolicy.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Maximum retains the canonical node, element, or interpolated position where the ",
                    "maximum occurs; exact ties follow the extrema-tie policy.",
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
            "maximum",
            "Find the maximum value over the target and where it occurs",
            concat!(
                "spatial = fea.SpatialReduction.maximum();\n",
                "reduction = fea.ReductionPolicy.new(spatial, fea.SeriesReduction.none(), ...\n",
                "    fea.ExtremaTiePolicy.firstCanonical());",
            ),
            concat!(
                "assert(strcmp(class(spatial), 'fea.SpatialReduction'));\n",
                "assert(spatial.value == \"maximum\");\n",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/spatial_reduction/tests.rs::spatial_reduction_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SPATIAL_REDUCTION_MAXIMUM_CATALOG_ENTRY,
    name: "fea.SpatialReduction.maximum",
    result: classes::SPATIAL_REDUCTION,
    signature: "spatial = fea.SpatialReduction.maximum()",
    inputs: [
    ],
    output: ("spatial", "Typed fea.SpatialReduction value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
