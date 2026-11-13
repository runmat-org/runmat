use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::inference::PolicyArg;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.SpatialReduction.moment",
    summary: "Integrate the moment of a force-like field about an explicit reference point.",
    description: concat!(
        "`fea.SpatialReduction.moment(about)` returns an `fea.SpatialReduction` whose ",
        "`value.moment.about` is the reference point. The reduction integrates r x f over the",
        " target, where r runs from `about` to each point.",
    ),
    keywords: &["fea", "observation", "spatial reduction", "moment", "reference point"],
    related: &["fea.SpatialReduction.resultant", "fea.Position.meters", "fea.ReductionPolicy.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "`about` must be a typed `fea.Position`, for example `fea.Position.meters([0 0 0])`; ",
                    "coordinates are stored in meters. A plain numeric vector fails with ",
                    "`RunMat:fea:InvalidInput`. The result is a three-component moment vector in N m. For",
                    " reaction forces the moment is the sum of each nodal force crossed with its lever ",
                    "arm.",
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
            "root-moment",
            "Take the moment of a traction about a bolt center",
            concat!(
                "spatial = fea.SpatialReduction.moment(fea.Position.meters([0.1 0 0.25]));\n",
                "reduction = fea.ReductionPolicy.new(spatial, fea.SeriesReduction.none(), ...\n",
                "    fea.ExtremaTiePolicy.firstCanonical());",
            ),
            concat!(
                "assert(strcmp(class(spatial), 'fea.SpatialReduction'));\n",
                "assert(spatial.value.moment.about.meters(3) == 0.25);\n",
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
        unit_test(
            "Moment reduction preserves its reference point",
            "crates/runmat-runtime/src/builtins/fea/study_policies/observation.rs::moment_reduction_preserves_its_typed_reference_point",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_SPATIAL_REDUCTION_MOMENT_CATALOG_ENTRY,
    name: "fea.SpatialReduction.moment",
    result: classes::SPATIAL_REDUCTION,
    signature: "spatial = fea.SpatialReduction.moment(about)",
    inputs: [
        (
            "about",
            Any,
            "Typed fea.Position of the moment reference point, in meters.",
            PolicyArg::Typed(&[classes::POSITION]),
        ),
    ],
    output: ("spatial", "Typed fea.SpatialReduction value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_NUMERIC_INPUT_AUDIT),
    documentation: DOCUMENTATION,
);
