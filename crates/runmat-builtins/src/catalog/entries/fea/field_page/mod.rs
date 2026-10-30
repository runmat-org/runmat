use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_FIELD_PAGE_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_run_setup, BOX_STUDY, NATIVE_CAD_MESHING},
    integer::FEA_FIELD_PAGE_INTEGER_CAPABILITIES,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::FieldPage)
}

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "page-displacements",
    title: "Read nodal displacements one page at a time",
    program: concat!(
        fea_box_run_setup!(),
        "displacement = fea.field(run, fea.StructuralField.displacement());\n",
        "page = fea.fieldPage(displacement, 1, 4);",
    ),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BOX_STUDY,
    requirements: NATIVE_CAD_MESHING,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(page, 'fea.FieldPage'));\nassert(page.start == 1);\nassert(page.count <= 4);\nassert(page.total_count >= page.count);\nassert(strcmp(page.numeric_representation, \"real\"));",
    },
}];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.fieldPage",
    summary: "Read a bounded, one-based page of values from a stored FEA field.",
    description: "`page = fea.fieldPage(field)` reads the first 256 logical values of a `fea.Field`. `fea.fieldPage(field, start, count)` reads up to `count` values starting at one-based logical index `start`; a single page holds at most 4096 values.",
    keywords: &["fea", "field", "page", "values", "complex", "artifact", "streaming"],
    related: &["fea.field", "fea.results", "fea.run"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "Large result fields stay in the run's artifact store; `fea.fieldPage` reads one bounded slice so scripts can walk a field of any size without loading it all. The field must come from `fea.field` and carry its `run_id`; its field id and frequency sample (if any) select the stored data.",
            "`start` is a one-based logical value index and defaults to 1. `count` is the maximum number of values and defaults to 256; requests above 4096 are capped at 4096. Both must be positive integers, given as any integer class or an integral double.",
            "The returned `fea.FieldPage` holds `run_id`, `field_id`, `sample`, `values`, `start`, `count` (the number actually returned), `total_count`, `next_start` (empty when this is the last page), `has_more`, and `numeric_representation`. Real fields return real values; complex frequency-domain fields return complex values with magnitude and phase intact. Indexes and counts are exact integers.",
            "A logical value is one stored entry of the field; for vector and tensor fields each node or element contributes `component_count` entries.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "After `fea.run`, select a field with `fea.field` and loop `fea.fieldPage(field, page.next_start, count)` while `page.has_more` is true. For small fields loaded inline (`field.materialized` is true), the values are already in `field.values`.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports more than three arguments, a first argument that is not an `fea.Field`, a field without a run id, a malformed sample, or a zero, negative, or fractional `start` or `count`. `RunMat:fea:OperationFailed` reports a run or field that cannot be found or read.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "How do I read a whole field?", answer: "Start at 1 and keep calling `fea.fieldPage(field, page.next_start, 4096)` while `page.has_more` is true, appending `page.values` each time." },
        BuiltinDocumentationFaq { question: "Can start and count be integer types?", answer: "Yes. Any integer class or an integral double is accepted and kept exact; values must be at least 1." },
    ],
    links: &[doc_links::OUTPUTS_AND_FIELDS, doc_links::OPERATION_REFERENCE],
    implementation: &[doc_links::FEA_FIELD_ACCESS_SOURCE],
    verification: &[
        doc_links::RESULTS_FIELD_PAGE_TEST,
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Page values preserve real and complex numeric semantics", location: "crates/runmat-runtime/src/builtins/fea/field_access.rs::page_values_preserve_real_and_complex_numeric_semantics" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Complex pages expose one-based progress", location: "crates/runmat-runtime/src/builtins/fea/field_access.rs::complex_page_exposes_one_based_progress_without_flattening_phase" },
    ],
    notes: &[],
});

pub const FEA_FIELD_PAGE_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.fieldPage",
        documentation: DOCUMENTATION,
        descriptor: &FEA_FIELD_PAGE_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_FIELD_PAGE_INTEGER_CAPABILITIES,
        integer_audit: None,
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_FIELD_PAGE_CATALOG_ENTRY];
