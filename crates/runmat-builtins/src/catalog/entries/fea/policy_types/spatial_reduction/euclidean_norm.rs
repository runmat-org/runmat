use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpatialReduction.euclideanNorm",
    summary: "Take the L2 norm of the quantity over the target.",
    description: concat!(
        "`fea.SpatialReduction.euclideanNorm()` returns the `fea.SpatialReduction` whose ",
        "`value` is `euclidean_norm`. Euclidean norm is the L2 norm of the field over the ",
        "target, computed with quadrature; the result is a scalar.",
    ),
    keywords: &["fea", "observation", "spatial reduction", "euclidean norm"],
    related: &["fea.SpatialReduction.none", "fea.SpatialReduction.integral", "fea.SpatialReduction.average", "fea.SpatialReduction.rootMeanSquare", "fea.ReductionPolicy.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Euclidean norm is the L2 norm of the field over the target, computed with ",
                    "quadrature; the result is a scalar.",
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
            "euclidean-norm",
            "Take the L2 norm of the quantity over the target",
            concat!(
                "spatial = fea.SpatialReduction.euclideanNorm();\n",
                "reduction = fea.ReductionPolicy.new(spatial, fea.SeriesReduction.none(), ...\n",
                "    fea.ExtremaTiePolicy.firstCanonical());",
            ),
            concat!(
                "assert(strcmp(class(spatial), 'fea.SpatialReduction'));\n",
                "assert(spatial.value == \"euclidean_norm\");\n",
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
    FEA_SPATIAL_REDUCTION_EUCLIDEAN_NORM_CATALOG_ENTRY,
    name: "fea.SpatialReduction.euclideanNorm",
    result: classes::SPATIAL_REDUCTION,
    signature: "spatial = fea.SpatialReduction.euclideanNorm()",
    inputs: [
    ],
    output: ("spatial", "Typed fea.SpatialReduction value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
