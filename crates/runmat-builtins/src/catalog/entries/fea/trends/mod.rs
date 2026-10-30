use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_TRENDS_DESCRIPTOR,
    doc_links,
    integer::FEA_TRENDS_INTEGER_CAPABILITIES,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    operation_inference::{infer_evidence, FeaEvidenceKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_evidence(request, entry, FeaEvidenceKind::Trends)
}

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "recent-trends",
    title: "Summarize the eight most recent runs of each kind",
    program: "trends = fea.trends(\"WindowSize\", 8);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Native,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(trends, 'fea.Trends'));\nassert(trends.window_size == 8);",
    },
}];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.trends",
    summary: "Summarize solve time and quality across recent persisted FEA runs.",
    description: "`trends = fea.trends()` summarizes the 16 most recent stored runs of each analysis kind. `fea.trends(\"WindowSize\", n)` uses the `n` most recent runs instead.",
    keywords: &["fea", "trends", "history", "quality", "solve time", "regression", "monitoring"],
    related: &["fea.compare", "fea.run", "fea.results"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "RunMat lists the runs in the analysis artifact store and groups them by run identity, meaning physics family and procedure (for example structural static or nonlinear). Within each group it keeps the first `WindowSize` runs in descending run-id order and summarizes them.",
            "The `fea.Trends` result holds `window_size` and one entry per group in `summaries`. Each summary reports the group identity, `sample_count`, median and 95th-percentile solve time in milliseconds when recorded, and the fraction of runs that were numerically acceptable. Nonlinear groups add the failed-increment rate and mean iteration-spike and convergence-stall counts; thermo-mechanical diagnostics add coupling, warning, and threshold-breach rates when present.",
            "`WindowSize` must be a positive integer, given as any integer class or an integral double, and may appear once. The option name is case-insensitive with punctuation ignored.",
            "Trends only read stored run metadata; nothing is solved or modified.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Use `fea.trends` after a series of `fea.run` calls, for example in a nightly qualification script, to watch solve time and quality drift. Drill into a specific pair of runs with `fea.compare`.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports an incomplete Name/Value pair, an option other than `WindowSize`, a repeated `WindowSize`, or a window that is zero, negative, or fractional. `RunMat:fea:OperationFailed` reports a run store that cannot be listed.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is the default window?", answer: "16 runs per analysis kind." },
        BuiltinDocumentationFaq { question: "Which runs are included?", answer: "Every run in the configured analysis artifact store, grouped by physics family and procedure. Point `RUNMAT_FEA_STUDY_ARTIFACT_ROOT` at another store to summarize different runs." },
    ],
    links: &[doc_links::EVIDENCE, doc_links::OPERATION_REFERENCE],
    implementation: &[doc_links::FEA_ENTRYPOINTS_SOURCE],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Trends summarize recent nonlinear runs", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_trends_summarizes_recent_nonlinear_runs_rejects_unsupported_envelope" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Trends classify acoustic runs separately", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_trends_classifies_acoustic_runs_separately" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Execution controls enforce exact structural boundaries", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_execution_controls_enforce_exact_structural_boundaries" },
    ],
    notes: &[],
});

pub const FEA_TRENDS_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.trends",
        documentation: DOCUMENTATION,
        descriptor: &FEA_TRENDS_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_TRENDS_INTEGER_CAPABILITIES,
        integer_audit: None,
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_TRENDS_CATALOG_ENTRY];
