use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpatialReduction.rootMeanSquare",
    summary: "Take the root-mean-square of the quantity over the target.",
    description: concat!(
        "`fea.SpatialReduction.rootMeanSquare()` returns the `fea.SpatialReduction` whose ",
        "`value` is `root_mean_square`. Root-mean-square integrates the squared value with ",
        "element or face quadrature and publishes one global value.",
    ),
    keywords: &["fea", "observation", "spatial reduction", "root mean square"],
    related: &["fea.SpatialReduction.none", "fea.SpatialReduction.integral", "fea.SpatialReduction.average", "fea.SpatialReduction.euclideanNorm", "fea.ReductionPolicy.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Root-mean-square integrates the squared value with element or face quadrature and ",
                    "publishes one global value.",
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
            "Take the root-mean-square of the quantity over the target",
            concat!(
                "spatial = fea.SpatialReduction.rootMeanSquare();\n",
                "reduction = fea.ReductionPolicy.new(spatial, fea.SeriesReduction.none(), ...\n",
                "    fea.ExtremaTiePolicy.firstCanonical());",
            ),
            concat!(
                "assert(strcmp(class(spatial), 'fea.SpatialReduction'));\n",
                "assert(spatial.value == \"root_mean_square\");\n",
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
    FEA_SPATIAL_REDUCTION_ROOT_MEAN_SQUARE_CATALOG_ENTRY,
    name: "fea.SpatialReduction.rootMeanSquare",
    result: classes::SPATIAL_REDUCTION,
    signature: "spatial = fea.SpatialReduction.rootMeanSquare()",
    inputs: [
    ],
    output: ("spatial", "Typed fea.SpatialReduction value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
