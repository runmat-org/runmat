use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`spectra` is one `fea.ResponseSpectrum` or a cell array of them. At least one spectrum is required and spectrum ids must be unique.",
        "The domain stores only the spectrum ids; the spectra themselves are declared on the model. Model validation requires every id to name an existing typed spectrum.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the domain as the second argument (`independentDomain`) of `fea.ProcedureStep.configure`. It replaces the step's default domain, and model validation checks it against the step's procedure and against every observation's sampling policy.",
        "Spectrum directions are independent tasks followed by a deterministic fan-in.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `spectra` is empty, contains a non-spectrum value, or repeats a spectrum id.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "launch",
        title: "Bind a launch acceleration spectrum",
        program: concat!(
            "spectrum = fea.ResponseSpectrum.new(\"launch_x\", fea.SpectrumAxis.frequencyHertz([5 10 20]), [12 18 9], ...\n",
            "    fea.SpectrumDamping.ratio(0.05), fea.SpectrumResponse.absoluteAcceleration(), fea.SpectrumEnvelope.maximax(), ...\n",
            "    fea.SpectrumInterpolation.linearFrequency(), fea.SpectrumExtrapolation.reject(), ...\n",
            "    fea.SpectrumDirection.new(fea.CoordinateFrame.globalFrame(), [1 0 0]), ...\n",
            "    fea.ModalCombination.completeQuadratic(), fea.DirectionalCombination.srss(), ...\n",
            "    fea.ModalTruncation.new(0.9, fea.ModalTruncationCorrection.missingMass(9.81)), fea.RigidResponse.disabled(), ...\n",
            "    fea.SpectrumConstruction.authoredDesign(), fea.SpectrumEvidence.bounded(0.95, 0.01), \"sha256:launch-spectrum\");\n",
            "domain = fea.IndependentDomain.responseSpectra({spectrum});",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.IndependentDomain'));\nassert(strcmp(domain.value.response_spectra.spectrum_ids, 'launch_x'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.IndependentDomain.responseSpectra",
        summary: "Bind a modal-response step to one or more typed response spectra.",
        description: "`fea.IndependentDomain.responseSpectra(spectra)` builds a response-spectrum `fea.IndependentDomain` that names the `fea.ResponseSpectrum` inputs a modal-response procedure consumes.",
        keywords: &["fea", "domain", "response", "spectrum", "seismic", "step"],
        related: &["fea.ResponseSpectrum.new", "fea.ProcedureStep.configure", "fea.SamplingPolicy.finalSample"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Does the domain copy the spectrum data?", answer: "No. It keeps the spectrum ids; the model owns the spectra." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/independent_domain/tests.rs::independent_domain_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Response-spectrum domain", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::response_spectrum_domain_retains_typed_spectrum_id" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Spectrum ids", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::modal_response_requires_typed_existing_spectrum_ids" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Static constructor table", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::independent_domain_class_exposes_every_static_constructor" },
        ],
        notes: &[],
    });
