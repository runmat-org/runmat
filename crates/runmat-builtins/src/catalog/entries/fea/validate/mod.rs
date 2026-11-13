use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_VALIDATE_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_model_setup, BOX_STUDY, NATIVE_CAD},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    operation_inference::{infer_evidence, FeaEvidenceKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_evidence(request, entry, FeaEvidenceKind::Validate)
}

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "validate-box-study",
    title: "Check a study before meshing or solving",
    program: concat!(
        fea_box_model_setup!(),
        "study = fea.study(\"box_static\", part, model);\n",
        "check = fea.validate(study);",
    ),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BOX_STUDY,
    requirements: NATIVE_CAD,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(check, 'fea.Validation'));\nassert(check.valid);\nassert(isempty(check.issue_codes));\nassert(~isempty(check.evidence_artifact_path));",
    },
}];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.validate",
    summary: "Check a study or sweep for problems without planning or solving it.",
    description: "`result = fea.validate(study)` returns a `fea.Validation` with a `valid` flag, sorted `issue_codes`, and an `issues` list of codes with messages. `fea.validate(sweep)` returns a `fea.SweepValidation` with one entry per study.",
    keywords: &["fea", "validate", "check", "study", "sweep", "preflight"],
    related: &["fea.study", "fea.sweep", "fea.plan", "fea.run"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Validation is the cheapest way to find authoring mistakes. It does not mesh, plan solver operations, or solve. A study is checked for an empty study or model id, geometry with no meshes or unspecified units, an empty or mismatched execution plan, a model bound to a different geometry revision, and every whole-model rule (materials, constraints or boundaries, references, procedure semantics, units, and frame). Electromagnetic steps are also checked for a residual target strictly between 0 and 1.",
            "Problems are reported, not thrown: the result's `valid` is `false` and `issue_codes` lists stable codes such as `RM.FEA.STUDY.MODEL_INVALID`. The matching entry in `issues` carries a readable message; for a model failure, the message includes the specific validation detail.",
            "Each call records a content-addressed validation evidence artifact for the study and returns its path in `evidence_artifact_path`. The artifact root is taken from the analysis artifact store configuration (for example `RUNMAT_FEA_STUDY_ARTIFACT_ROOT`).",
            "For a sweep, the result has the sweep id, an overall `valid`, sweep-level `issue_codes`, and `study_entries` in the order the studies were given.",
            "The input must be a `fea.Study` or `fea.Sweep` object. Paths to `.fea` files are not accepted; check `.fea` sources with `runmat check`.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Call `fea.validate` after `fea.study` or `fea.sweep` and before `fea.plan` or `fea.run`. The same object can then be planned and run unchanged.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports an argument that is not a `fea.Study` or `fea.Sweep`. `RunMat:fea:OperationFailed` reports a failure to persist validation evidence. Study defects themselves are returned in the result, not raised.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "Why is my study valid but fea.run still fails?", answer: "Validation does not mesh or solve. Solving also needs a solver mesh attached with `fea.study(..., \"Mesh\", mesh)` and can fail for numerical reasons; `fea.plan` additionally checks that every step maps to a supported solver operation." },
        BuiltinDocumentationFaq { question: "Does fea.validate write files?", answer: "Yes. It records a small validation evidence artifact in the analysis artifact store and returns its path." },
    ],
    links: &[doc_links::STUDY_CONTRACTS, doc_links::EVIDENCE, doc_links::OPERATION_REFERENCE],
    implementation: &[doc_links::FEA_ENTRYPOINTS_SOURCE, doc_links::ANALYSIS_OPERATIONS_SOURCE],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validation reports an invalid study id", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_validate_study_reports_invalid_study_id" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validation preserves typed model failure detail", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_validate_study_preserves_typed_model_failure_detail" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Sweep validation reports entries and persists evidence", location: "crates/runmat-runtime/src/analysis/tests.rs::analysis_validate_study_sweep_reports_valid_entries_and_persists_artifact" },
        doc_links::SWEEP_CLASSES_TEST,
    ],
    notes: &[],
});

pub const FEA_VALIDATE_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.validate",
        documentation: DOCUMENTATION,
        descriptor: &FEA_VALIDATE_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::SynchronousEvidenceWrite,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_VALIDATE_CATALOG_ENTRY];
