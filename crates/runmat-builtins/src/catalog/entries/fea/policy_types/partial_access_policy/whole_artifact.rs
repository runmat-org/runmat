use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.PartialAccessPolicy.wholeArtifact",
    summary: "Publish the observation as one artifact.",
    description: concat!(
        "`fea.PartialAccessPolicy.wholeArtifact()` returns the `fea.PartialAccessPolicy` ",
        "whose `value` is `whole_artifact`. Whole artifact stores the observation as a single",
        " object that consumers read in full.",
    ),
    keywords: &["fea", "observation", "partial access policy", "whole artifact"],
    related: &["fea.PartialAccessPolicy.manifestAndChunks", "fea.ObservationStorage.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Whole artifact stores the observation as a single object that consumers read in ",
                    "full.",
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
            "whole-artifact",
            "Publish the observation as one artifact",
            concat!(
                "access = fea.PartialAccessPolicy.wholeArtifact();\n",
                "storage = fea.ObservationStorage.new(fea.SignalPrecision.real64(), ...\n",
                "    fea.CompressionPolicy.lossless(), fea.RetentionPolicy.canonical(), access);",
            ),
            concat!(
                "assert(strcmp(class(access), 'fea.PartialAccessPolicy'));\n",
                "assert(access.value == \"whole_artifact\");\n",
                "assert(strcmp(class(storage), 'fea.ObservationStorage'));",
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
            "crates/runmat-builtins/src/catalog/entries/fea/policy_types/partial_access_policy/tests.rs::partial_access_policy_identities_conform",
        ),
    ],
    notes: &[],
});

policy_entry!(
    FEA_PARTIAL_ACCESS_POLICY_WHOLE_ARTIFACT_CATALOG_ENTRY,
    name: "fea.PartialAccessPolicy.wholeArtifact",
    result: classes::PARTIAL_ACCESS_POLICY,
    signature: "access = fea.PartialAccessPolicy.wholeArtifact()",
    inputs: [
    ],
    output: ("access", "Typed fea.PartialAccessPolicy value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
