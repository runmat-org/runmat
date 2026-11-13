use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::fea_contract::policy_types::documentation::{example, page, section, source, unit_test};
use crate::fea_contract::policy_types::entry::policy_entry;
use crate::fea_contract::policy_types::{classes, integer};
use crate::BuiltinDocumentation;

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.PartialAccessPolicy.manifestAndChunks",
    summary: "Publish a manifest and independently addressable result chunks.",
    description: concat!(
        "`fea.PartialAccessPolicy.manifestAndChunks()` returns the `fea.PartialAccessPolicy` ",
        "whose `value` is `manifest_and_chunks`. Manifest and chunks lets a local or remote ",
        "consumer fetch metadata and only the samples or chunks it needs; the observation's ",
        "declared cost must then estimate nonzero stored bytes.",
    ),
    keywords: &["fea", "observation", "partial access policy", "manifest and chunks"],
    related: &["fea.PartialAccessPolicy.wholeArtifact", "fea.ObservationStorage.new"],
    sections: &[
        section(
            "Behavior",
            &[
                concat!(
                    "Manifest and chunks lets a local or remote consumer fetch metadata and only the ",
                    "samples or chunks it needs; the observation's declared cost must then estimate ",
                    "nonzero stored bytes.",
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
            "manifest-and-chunks",
            "Publish a manifest and independently addressable result chunks",
            concat!(
                "access = fea.PartialAccessPolicy.manifestAndChunks();\n",
                "storage = fea.ObservationStorage.new(fea.SignalPrecision.real64(), ...\n",
                "    fea.CompressionPolicy.lossless(), fea.RetentionPolicy.canonical(), access);",
            ),
            concat!(
                "assert(strcmp(class(access), 'fea.PartialAccessPolicy'));\n",
                "assert(access.value == \"manifest_and_chunks\");\n",
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
    FEA_PARTIAL_ACCESS_POLICY_MANIFEST_AND_CHUNKS_CATALOG_ENTRY,
    name: "fea.PartialAccessPolicy.manifestAndChunks",
    result: classes::PARTIAL_ACCESS_POLICY,
    signature: "access = fea.PartialAccessPolicy.manifestAndChunks()",
    inputs: [
    ],
    output: ("access", "Typed fea.PartialAccessPolicy value."),
    integer_capabilities: &[],
    integer_audit: Some(&integer::NO_ARGUMENT_AUDIT),
    documentation: DOCUMENTATION,
);
