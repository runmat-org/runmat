use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Each argument must be the typed value of its class; there are no defaults. The result exposes `precision`, `compression`, `retention`, and `partial_access` properties.",
        "`fea.PartialAccessPolicy.manifestAndChunks()` lets a local or remote consumer fetch the manifest and only the sample chunks it needs; `wholeArtifact()` stores one artifact. A retained history uses one manifest payload and one addressable payload per retained sample.",
        "Model validation checks the combination with the rest of the observation: complex precision must match a complex representation, error-bounded compression cannot feed solver control or downstream state, retention counts must be nonzero, and chunk-addressable or retained output must declare nonzero stored bytes in `fea.ObservationCost`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the storage object as the 13th argument (`storage`) of `fea.Observation.field`, `history`, `probe`, or `reduction`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an argument is not the expected typed policy.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "chunked",
        title: "Lossless, canonical, chunk-addressable storage",
        program: "storage = fea.ObservationStorage.new( ...\n    fea.SignalPrecision.real64(), fea.CompressionPolicy.lossless(), ...\n    fea.RetentionPolicy.canonical(), fea.PartialAccessPolicy.manifestAndChunks());\n",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: concat!(
            "assert(strcmp(class(storage), 'fea.ObservationStorage'));\n",
            "assert(strcmp(storage.precision, 'real64'));\n",
            "assert(strcmp(storage.compression, 'lossless'));\n",
            "assert(strcmp(storage.retention, 'canonical'));\n",
            "assert(strcmp(storage.partial_access, 'manifest_and_chunks'));",
        ) },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationStorage.new",
        summary: "Declare how an observation's samples are stored, compressed, retained, and accessed.",
        description: "`fea.ObservationStorage.new(precision, compression, retention, partialAccess)` combines the four storage decisions for one observation into a `fea.ObservationStorage` object.",
        keywords: &["fea", "observation", "storage", "retention", "compression", "precision"],
        related: &["fea.SignalPrecision.real64", "fea.CompressionPolicy.lossless", "fea.RetentionPolicy.canonical", "fea.ObservationCost.new", "fea.Observation.field"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Does storage policy change the computed values?", answer: "Only precision and error-bounded compression affect stored values; the solve itself is unchanged." }, BuiltinDocumentationFaq { question: "Can I fetch one time sample of a long history?", answer: "Yes, with `manifestAndChunks`: the manifest maps physical coordinates to sample identities and only that sample's chunks are imported." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation_storage/tests.rs::observation_storage_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        ],
        notes: &[],
    });
