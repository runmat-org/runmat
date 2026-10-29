//! Shared documentation links and evidence for the FEA operation surface.

use crate::*;

const fn doc(label: &'static str, path: &'static str) -> BuiltinDocumentationLink {
    BuiltinDocumentationLink {
        label,
        target: BuiltinDocumentationLinkTarget::Documentation(path),
    }
}

const fn source(label: &'static str, url: &'static str) -> BuiltinDocumentationLink {
    BuiltinDocumentationLink {
        label,
        target: BuiltinDocumentationLinkTarget::Source(url),
    }
}

pub const GET_STARTED: BuiltinDocumentationLink = doc("Get started with FEA", "/docs/simulation/get-started");
pub const MODELS: BuiltinDocumentationLink = doc("FEA models", "/docs/simulation/models");
pub const STUDY_CONTRACTS: BuiltinDocumentationLink =
    doc("Study contracts", "/docs/simulation/study-contracts");
pub const SOLVES: BuiltinDocumentationLink = doc("FEA solves", "/docs/simulation/solves");
pub const EVIDENCE: BuiltinDocumentationLink = doc("Run evidence", "/docs/simulation/evidence");
pub const MESHING: BuiltinDocumentationLink = doc("Meshing", "/docs/simulation/meshing");
pub const OPERATION_REFERENCE: BuiltinDocumentationLink = doc(
    "FEA operation reference",
    "/docs/simulation/operation-reference",
);
pub const OUTPUTS_AND_FIELDS: BuiltinDocumentationLink = doc(
    "Outputs, fields, and observations",
    "/docs/simulation/language/outputs-fields-and-observations",
);
pub const TYPES_AND_UNITS: BuiltinDocumentationLink = doc(
    "Types, units, and expressions",
    "/docs/simulation/language/types-units-and-expressions",
);
pub const SWEEPS: BuiltinDocumentationLink = doc(
    "Inputs, sweeps, and cases",
    "/docs/simulation/language/inputs-sweeps-and-cases",
);
pub const STRUCTURAL_PHYSICS: BuiltinDocumentationLink =
    doc("Structural physics", "/docs/simulation/physics/structural");

pub const FEA_ENTRYPOINTS_SOURCE: BuiltinDocumentationLink = source(
    "FEA runtime entrypoints",
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/mod.rs",
);
pub const FEA_MODELING_SOURCE: BuiltinDocumentationLink = source(
    "FEA study, model, and material construction",
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/modeling.rs",
);
pub const FEA_FIELD_ACCESS_SOURCE: BuiltinDocumentationLink = source(
    "FEA field paging",
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/field_access.rs",
);
pub const FEA_VISUALIZATION_SOURCE: BuiltinDocumentationLink = source(
    "FEA plot and mesh overloads",
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/visualization.rs",
);
pub const FEA_RUN_RESULTS_SOURCE: BuiltinDocumentationLink = source(
    "FEA step-result selection",
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/run_results.rs",
);
pub const FEA_FIELD_SELECTORS_SOURCE: BuiltinDocumentationLink = source(
    "FEA field selectors",
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/field_selectors.rs",
);
pub const FEA_STATISTICAL_FIELD_SELECTORS_SOURCE: BuiltinDocumentationLink = source(
    "FEA statistical field selectors",
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/statistical_field_selectors.rs",
);
pub const FEA_VALUES_SOURCE: BuiltinDocumentationLink = source(
    "FEA dimensioned values",
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/values.rs",
);
pub const ANALYSIS_OPERATIONS_SOURCE: BuiltinDocumentationLink = source(
    "FEA analysis operations",
    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/analysis/mod.rs",
);

const fn unit(label: &'static str, location: &'static str) -> BuiltinEvidenceReference {
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label,
        location,
    }
}

pub const STUDY_AND_SWEEP_CONSTRUCTION_TEST: BuiltinEvidenceReference = unit(
    "Typed constructors build full study and sweep objects",
    "crates/runmat-runtime/src/builtins/fea/mod.rs::typed_constructors_build_full_study_and_sweep_objects",
);
pub const SWEEP_CLASSES_TEST: BuiltinEvidenceReference = unit(
    "Sweep and single-run operation classes stay distinct",
    "crates/runmat-runtime/src/builtins/fea/mod.rs::sweep_operation_classes_are_exact_and_do_not_inherit_single_run_methods",
);
pub const RESULTS_FIELD_PAGE_TEST: BuiltinEvidenceReference = unit(
    "Results, field, and page objects expose values and context",
    "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_results_field_exposes_values_metadata_and_plot_context",
);
pub const PLOT_MESH_TEST: BuiltinEvidenceReference = unit(
    "Plot and mesh overloads return figure handles",
    "crates/runmat-runtime/src/builtins/fea/mod.rs::plot_and_mesh_overloads_return_figure_handles_for_fea_objects",
);
pub const STRUCTURAL_SELECTORS_TEST: BuiltinEvidenceReference = unit(
    "Structural field selectors are typed and portable",
    "crates/runmat-runtime/src/builtins/fea/mod.rs::structural_field_selectors_are_typed_and_portable",
);
pub const STATISTICAL_SELECTORS_TEST: BuiltinEvidenceReference = unit(
    "Statistical field selectors are typed, composable, and portable",
    "crates/runmat-runtime/src/builtins/fea/mod.rs::statistical_field_selectors_are_typed_composable_and_portable",
);
pub const OUTPUT_SELECTOR_TEST: BuiltinEvidenceReference = unit(
    "Authored output selectors are typed named data",
    "crates/runmat-runtime/src/builtins/fea/field_selectors.rs::authored_output_field_selectors_are_typed_named_data",
);
pub const QUANTITY_SI_TEST: BuiltinEvidenceReference = unit(
    "Scalar quantities convert to canonical SI and remain portable",
    "crates/runmat-runtime/src/builtins/fea/values.rs::scalar_quantities_convert_to_canonical_si_and_remain_portable",
);
pub const QUANTITY_RANGE_TEST: BuiltinEvidenceReference = unit(
    "Quantity classes reject wrong dimensions and invalid values",
    "crates/runmat-runtime/src/builtins/fea/values.rs::quantity_classes_reject_wrong_dimensions_and_invalid_values",
);
pub const VECTOR_QUANTITY_TEST: BuiltinEvidenceReference = unit(
    "Positions preserve vector shape and unit conversion",
    "crates/runmat-runtime/src/builtins/fea/values.rs::positions_preserve_vector_shape_and_unit_conversion",
);
pub const PORTABLE_SCHEMA_TEST: BuiltinEvidenceReference = unit(
    "Material and plan objects match their portable schemas",
    "crates/runmat-runtime/src/builtins/fea/mod.rs::material_and_plan_objects_match_their_portable_schemas",
);
pub const ENUM_ATTRIBUTION_TEST: BuiltinEvidenceReference = unit(
    "Frame and Confidence errors are attributed to the calling builtin",
    "crates/runmat-runtime/src/builtins/fea/mod.rs::scalar_enum_errors_are_attributed_to_the_calling_builtin",
);
pub const STEP_RESULT_TEST: BuiltinEvidenceReference = unit(
    "A step result is selected by its authored step id",
    "crates/runmat-runtime/src/analysis/tests.rs::fea_step_result_selects_one_procedure_step_by_authored_id",
);
pub const ASSIGNMENT_REVISION_TEST: BuiltinEvidenceReference = unit(
    "Material assignments expose a uint32 geometry revision",
    "crates/runmat-runtime/src/builtins/fea/mod.rs::material_assignment_revision_is_uint32_and_round_trips_into_a_model",
);
