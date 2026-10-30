use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_RESULTS_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_run_setup, BOX_STUDY, NATIVE_CAD_MESHING},
    integer::FEA_RESULTS_INTEGER_CAPABILITIES,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::Results)
}

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "results-from-run",
        title: "Open the results of a solved study",
        program: concat!(fea_box_run_setup!(), "results = fea.results(run);"),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_STUDY,
        requirements: NATIVE_CAD_MESHING,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(results, 'fea.Results'));\nassert(strcmp(results.run_id, run.steps(1).run_id));\nassert(results.summary.field_count > 0);",
        },
    },
    BuiltinExample {
        id: "results-by-run-id",
        title: "Reopen results by run id, metadata only",
        program: concat!(
            fea_box_run_setup!(),
            "runId = run.steps(1).run_id;\n",
            "results = fea.results(runId, \"IncludeFieldValues\", false, \"IncludeDiagnostics\", true);",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_STUDY,
        requirements: NATIVE_CAD_MESHING,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(results, 'fea.Results'));\nassert(strcmp(results.run_id, runId));",
        },
    },
];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.results",
    summary: "Load the persisted results of an FEA run for post-processing.",
    description: "`results = fea.results(run)` loads the results of a single-step `fea.RunResult` or a `fea.StepRunResult`; `fea.results(runId)` loads them by persisted run id. Name/value options select which fields, diagnostics, modes, snapshots, and nonlinear details to include.",
    keywords: &["fea", "results", "post-processing", "run id", "fields", "diagnostics", "modes", "snapshots"],
    related: &["fea.run", "fea.stepResult", "fea.field", "fea.fieldPage", "fea.compare"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The first argument identifies one solved procedure step: a run id string, a `fea.StepRunResult`, or a `fea.RunResult` with exactly one step. A multi-step `fea.RunResult` is rejected; select a step with `run.step(stepId)`. A `fea.SweepRunResult` is rejected; pick one of its `run_entries`. Passing an existing `fea.Results` with no options returns it unchanged.",
            "The returned `fea.Results` carries the run identity, field descriptors, fields, modal, thermal, transient, nonlinear, and structural-frequency results when present, diagnostics, run status, quality and publication reasons, provenance, a `summary`, and the `run_id`. When the input carried its study, the study is kept so later `fea.field`, `plot`, and `mesh` calls can reach the geometry.",
            "Options (names are case-insensitive, punctuation ignored, each at most once): `IncludeFields` (alias `Fields`) takes a string, char row, or cell of field ids or unique dotted suffixes; `IncludeFieldValues` (alias `FieldValues`) chooses whether field values are materialized or only described; `IncludeDiagnostics`, `IncludeModalResults`, `IncludeTransientResults`, and `IncludeNonlinearResults` toggle those sections; `DiagnosticCodes` filters diagnostics by code; `ModeIndices` and `TransientSnapshotIndices` select modes and snapshots.",
            "Toggles accept a logical scalar or an exact numeric 0 or 1. Mode and snapshot selectors are one-based vectors of positive integers in any integer or double class; order and duplicates are preserved. The `summary.available_mode_indices` list is also one-based.",
            "Counts, indexes, shapes, and fingerprints in the result stay exact integers; physical values stay double precision.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Call `fea.results` on what `fea.run` returned, or on a saved run id in a later session. Pass the results to `fea.field` with a typed selector to extract one field, then page through it with `fea.fieldPage` or plot it with `plot(field)`.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports a missing run reference, a multi-step run or a sweep run, an unsupported or repeated option, a non-logical toggle, or a selector index that is zero, negative, fractional, or a matrix. `RunMat:fea:OperationFailed` reports a run id that is not in the artifact store or results that cannot be read.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "Are mode indices zero- or one-based?", answer: "One-based, like all RunMat indexing. `\"ModeIndices\", [1 2 3]` selects the first three modes, and `summary.available_mode_indices` is one-based too." },
        BuiltinDocumentationFaq { question: "How do I get results for one step of a multi-step study?", answer: "Select the step first with `run.step(\"stepId\")` or `fea.stepResult(run, \"stepId\")`, then call `fea.results` on it." },
        BuiltinDocumentationFaq { question: "Can I load results without their field values?", answer: "Yes. Pass `\"IncludeFieldValues\", false` to get descriptors and metadata only, then read values on demand with `fea.field` and `fea.fieldPage`." },
    ],
    links: &[doc_links::EVIDENCE, doc_links::OUTPUTS_AND_FIELDS, doc_links::OPERATION_REFERENCE],
    implementation: &[doc_links::FEA_ENTRYPOINTS_SOURCE],
    verification: &[
        doc_links::RESULTS_FIELD_PAGE_TEST,
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Snapshot selectors translate once from one-based indexes", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_results_translates_successful_snapshot_selectors_once" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Result selectors are exact one-based vectors and flags are zero or one", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_result_selectors_are_exact_one_based_vectors_and_flags_are_zero_one" },
    ],
    notes: &[],
});

pub const FEA_RESULTS_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.results",
        documentation: DOCUMENTATION,
        descriptor: &FEA_RESULTS_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_RESULTS_INTEGER_CAPABILITIES,
        integer_audit: None,
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_RESULTS_CATALOG_ENTRY];
