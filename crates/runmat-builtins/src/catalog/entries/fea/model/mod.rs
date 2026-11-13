use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_MODEL_DESCRIPTOR,
    doc_links,
    example_fixtures::{fea_box_model_setup, BOX_STUDY, NATIVE_CAD},
    integer::FEA_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Model)
}

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "loaded-box-model",
    title: "Model a steel box fixed at one face and loaded at the other",
    program: fea_box_model_setup!(),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BOX_STUDY,
    requirements: NATIVE_CAD,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(model, 'fea.Model'));\nassert(strcmp(model.model_id, \"box_model\"));\nassert(numel(model.materials) == 1);\nassert(numel(model.constraints) == 1);",
    },
}];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.model",
    summary: "Assign physical meaning to a geometry: materials, constraints, loads, and procedure steps.",
    description: "`model = fea.model(id, geometry, Name, Value, ...)` creates a `fea.Model` bound to one `geometry.Asset`. Name/value options supply typed components: `Materials`, `MaterialAssignments`, `Constraints`, `Boundaries`, `Signals`, `Excitations`, `ExcitationCases`, `InitialConditions`, `ResponseSpectra`, `Domains`, `Interfaces`, `Frame`, and exactly one of `Steps` or `Sequence`.",
    keywords: &["fea", "model", "materials", "constraints", "loads", "excitations", "steps", "boundary conditions"],
    related: &["fea.study", "fea.material", "fea.materialAssignment", "fea.StructuralConstraint.fixed", "fea.StructuralExcitation.force", "fea.ExcitationCase.always", "fea.StructuralStep.static", "fea.AnalysisSequence.new"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "A model separates physics from topology: the geometry defines regions, and the model states which material owns each solid, where the part is held, which loads act, and which analysis steps run. The model records the geometry id, revision, and units of the asset it was built on.",
            "Every component option takes a typed object or a cell array of typed objects from the matching constructor family: `fea.material` for `Materials`; `fea.materialAssignment` for `MaterialAssignments`; constraint constructors such as `fea.StructuralConstraint.fixed` for `Constraints`; boundary constructors such as `fea.ThermalBoundary.convection` for `Boundaries`; signal constructors such as `fea.ForceSignal.constantNewtons` for `Signals`; excitation constructors such as `fea.StructuralExcitation.force` for `Excitations`; `fea.ExcitationCase.*` for `ExcitationCases`; initial-condition, response-spectrum, domain, and interface constructors for the remaining options. Text and numeric values are not accepted in place of these objects.",
            "Assignments, constraints, boundaries, excitations, initial conditions, and interfaces carry the geometry they were selected on. Each must belong to the model's geometry; a region selected on another asset is rejected.",
            "The procedure comes from exactly one of `Steps` (a step object or cell of steps, composed into a sequence named `<id>.sequence`) or `Sequence` (a `fea.AnalysisSequence` with explicit activations, state transfer, and branching). Supplying both, or neither, is an error.",
            "`Frame` names the model's coordinate frame and defaults to the global frame. `Domains` may hold at most one domain object per physics family; a repeated family is rejected.",
            "Construction checks component classes and geometry ownership. Whole-model rules (at least one material, at least one constraint or boundary, valid references and procedure semantics, units and frame matching the geometry) are enforced when the model is placed in a study by `fea.study`.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "Pass the model as the third argument of `fea.study(id, geometry, model)`, using the same `geometry.Asset`. One model can back several studies, for example with different meshes or execution plans, and several models on one geometry can be combined in a `fea.sweep`.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports a non-text id, a geometry that is not a `geometry.Asset`, an unsupported option name, a component of the wrong class, a component bound to a different geometry, or a missing or doubled procedure source.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "When should I use Sequence instead of Steps?", answer: "Use `Steps` for independent single-step analyses. Use a `fea.AnalysisSequence` whenever activation, state transfer between steps, branching, or restart policy matters." },
        BuiltinDocumentationFaq { question: "Why does fea.model accept a model without materials?", answer: "Construction only checks component types and geometry ownership. The complete model rules, including at least one material and one constraint or boundary, are checked when you build a study." },
    ],
    links: &[doc_links::MODELS, doc_links::STUDY_CONTRACTS, doc_links::GET_STARTED],
    implementation: &[doc_links::FEA_MODELING_SOURCE],
    verification: &[
        doc_links::STUDY_AND_SWEEP_CONSTRUCTION_TEST,
        doc_links::ENUM_ATTRIBUTION_TEST,
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "A model requires an explicit procedure sequence", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_model_requires_explicit_procedure_sequence" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Ambiguous component option aliases are rejected", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_model_rejects_ambiguous_component_option_aliases" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Concrete steps compose and mislabeled steps are rejected", location: "crates/runmat-runtime/src/builtins/fea/steps.rs::model_composition_accepts_concrete_steps_and_rejects_mislabeled_steps" },
    ],
    notes: &[],
});

pub const FEA_MODEL_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.model",
        documentation: DOCUMENTATION,
        descriptor: &FEA_MODEL_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &[],
        integer_audit: Some(&FEA_STRUCTURAL_INTEGER_AUDIT),
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_MODEL_CATALOG_ENTRY];
