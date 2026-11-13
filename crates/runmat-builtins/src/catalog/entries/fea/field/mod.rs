use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition},
    descriptors::FEA_FIELD_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_run_setup, BOX_STUDY, NATIVE_CAD_MESHING},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
    operation_entry::{self, OperationEffectProfile, OperationEntryDefinition},
    result_inference::{infer_result, FeaResultKind},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_result(request, entry, FeaResultKind::Field)
}

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "von-mises-field",
        title: "Extract element von Mises stress from a run",
        program: concat!(
            fea_box_run_setup!(),
            "results = fea.results(run);\n",
            "stress = fea.field(results, fea.StructuralField.vonMises());",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_STUDY,
        requirements: NATIVE_CAD_MESHING,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(stress, 'fea.Field'));\nassert(strcmp(stress.field_id, \"structural.von_mises\"));\nassert(strcmp(stress.location, \"element\"));\nassert(stress.value_count > 0);",
        },
    },
    BuiltinExample {
        id: "displacement-field",
        title: "Read nodal displacement directly from the run",
        program: concat!(
            fea_box_run_setup!(),
            "displacement = fea.field(run, fea.StructuralField.displacement());",
        ),
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BOX_STUDY,
        requirements: NATIVE_CAD_MESHING,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(displacement, 'fea.Field'));\nassert(strcmp(displacement.location, \"node\"));\nassert(~isempty(displacement.run_id));",
        },
    },
];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.field",
    summary: "Select one result field, such as stress or displacement, from an FEA run.",
    description: "`field = fea.field(resultsOrRun, selector)` returns the `fea.Field` named by a typed selector such as `fea.StructuralField.vonMises()` or `fea.OutputField.named(\"peak_stress\")`. `fea.field(resultsOrRun, selector, fea.Frequency.hertz(f))` selects the field at one frequency sample; `fea.field(resultsOrRun, selector, fea.Time.seconds(t))` or `fea.field(resultsOrRun, selector, k)` selects it at one retained transient time.",
    keywords: &["fea", "field", "stress", "displacement", "von Mises", "post-processing", "results"],
    related: &["fea.results", "fea.fieldPage", "fea.StructuralField.vonMises", "fea.OutputField.named", "fea.Frequency.hertz", "fea.Time.seconds", "fea.run"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The first argument is a `fea.Results`, a single-step `fea.RunResult`, a `fea.StepRunResult`, or a persisted run id. The second is a typed selector: a `fea.StructuralField`, `fea.ResponseSpectrumField`, `fea.RandomResponseField`, or `fea.OutputField` value. Field ids given as text are not accepted, so a misspelled quantity is caught when the selector is built.",
            "The optional third argument selects a sample. A `fea.Frequency` (finite and positive) selects a frequency-domain sample. A `fea.Time` selects the retained transient state at that time; it matches a retained time to within one part in 10^9 of the run's time span, so times written in source match the times the integrator retained. A positive integer `k` selects the k-th retained transient state in time order. Structural displacement, rotation, strain, stress, von Mises stress, strain energy density, reaction force, and reaction moment have transient states.",
            "When the results already hold the selected field and no sample is given, the field is returned with its values in `values` and `materialized` set to `true`. Otherwise RunMat looks up the field in the run's artifacts and returns its descriptor with `materialized` set to `false`; read its values with `fea.fieldPage`. That lookup needs a run identity, so the input must be a run, a step run, a run id, or results that carry a `run_id`.",
            "A field reports `field_id`, `shape`, `unit`, `location` (lowercase, such as `node`, `element`, or `integrationpoint`), `kind`, `family`, `quantity`, `statistic`, topology and element metadata, `component_count`, `element_count`, `entity_count`, `value_count`, `storage`, `sample`, and `numeric_representation` (`real` or `complex`). Counts and shapes are exact integers. The field keeps the run id and study so it can be paged, plotted, or shown on its mesh.",
            "An exact field id wins; otherwise a unique dotted-suffix match is used.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Call `fea.field` on the output of `fea.run` or `fea.results`. Selectors come from the `fea.StructuralField.*`, `fea.ResponseSpectrumField.*`, and `fea.RandomResponseField.*` constructors, or from `fea.OutputField.named` for outputs a `.fea` study declares by name. Pass the field to `fea.fieldPage`, `plot(field)`, or `mesh(field)`.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports the wrong number of arguments, a text or unknown selector, a sample that is not a positive `fea.Frequency`, a `fea.Time`, or a positive integer, a multi-step run, or an artifact lookup without a run identity. `RunMat:fea:OperationFailed` reports a field that the run did not produce or artifacts that cannot be read.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "Why does fea.field reject \"von_mises\" as text?", answer: "Fields are chosen with typed selectors so the quantity is checked when you write it. Use `fea.StructuralField.vonMises()` instead of a string." },
        BuiltinDocumentationFaq { question: "Why is materialized false and values missing?", answer: "The field was described from the run's artifacts without loading every value. Read values in bounded pages with `fea.fieldPage(field, start, count)`." },
    ],
    links: &[doc_links::OUTPUTS_AND_FIELDS, doc_links::EVIDENCE, doc_links::OPERATION_REFERENCE],
    implementation: &[doc_links::FEA_ENTRYPOINTS_SOURCE],
    verification: &[
        doc_links::RESULTS_FIELD_PAGE_TEST,
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Field metadata stays exact in public objects", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_field_metadata_remains_exact_in_public_objects" },
    ],
    notes: &[],
});

pub const FEA_FIELD_CATALOG_ENTRY: BuiltinCatalogEntry =
    operation_entry::entry(OperationEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.field",
        documentation: DOCUMENTATION,
        descriptor: &FEA_FIELD_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        effects: OperationEffectProfile::ArtifactRead,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_FIELD_CATALOG_ENTRY];
