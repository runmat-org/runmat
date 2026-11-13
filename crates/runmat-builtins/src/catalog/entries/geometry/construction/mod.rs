//! Programmatic exact-geometry builder: feature constructors, Boolean and local
//! operations, typed topology selections, and generated-part finalization.
//!
//! Shared layout lives here; every identity module owns its prose, examples,
//! and argument roles.

use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod descriptor;
mod inference;
#[cfg(test)]
mod tests;

mod r#box;
mod chamfer;
mod cylinder;
mod edges;
mod edges_introduced_by;
mod extrude;
mod faces;
mod faces_introduced_by;
mod fillet;
mod finish;
mod hole;
mod instance;
mod intersect;
mod pocket;
mod revolve;
mod solid;
mod sphere;
mod subtract;
mod transform;
mod union;
mod with_region;

pub use descriptor::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &r#box::ENTRY,
    &cylinder::ENTRY,
    &sphere::ENTRY,
    &transform::ENTRY,
    &instance::ENTRY,
    &extrude::ENTRY,
    &revolve::ENTRY,
    &hole::ENTRY,
    &pocket::ENTRY,
    &fillet::ENTRY,
    &chamfer::ENTRY,
    &union::ENTRY,
    &intersect::ENTRY,
    &subtract::ENTRY,
    &finish::ENTRY,
    &solid::ENTRY,
    &faces::ENTRY,
    &edges::ENTRY,
    &faces_introduced_by::ENTRY,
    &edges_introduced_by::ENTRY,
    &with_region::ENTRY,
];

const GEOMETRY_GUIDE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Geometry",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/geometry"),
};
const GEOMETRY_LANGUAGE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Geometry, models, and assignments",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/language/geometry-models-and-assignments",
    ),
};
const FUNCTION_INTEGRATION: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "RunMat function integration",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/language/runmat-function-integration",
    ),
};
pub(super) const LINKS: &[BuiltinDocumentationLink] =
    &[GEOMETRY_GUIDE, GEOMETRY_LANGUAGE, FUNCTION_INTEGRATION];

pub(super) const PRIMITIVES_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Primitive and placement builders",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/generated/primitives.rs",
    ),
};
pub(super) const FEATURES_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Sweep, cut, and edge-treatment builders",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/generated/features.rs",
    ),
};
pub(super) const COMBINE_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Boolean builders and finalization",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/generated/combine.rs",
    ),
};
pub(super) const REGIONS_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Typed region selections",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/generated/regions.rs",
    ),
};
pub(super) const GRAPH_VALIDATION_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Feature-graph contract validation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-geometry/core/src/model/authoring_feature_validation.rs",
    ),
};

pub(super) const COMPOSITION_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Builder composes regions and crosses the portable value boundary",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::typed_builder_composes_regions_and_crosses_the_portable_value_boundary",
};
pub(super) const FULL_SURFACE_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Full builder surface preserves typed dependencies and selections",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::full_feature_builder_surface_preserves_typed_dependencies_and_selections",
};
pub(super) const FAIL_CLOSED_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Duplicate region names and untyped values fail closed",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::duplicate_region_names_and_untyped_values_fail_closed",
};
pub(super) const INSTANCE_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Instances require explicit affine occurrences",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::instances_require_explicit_affine_occurrences",
};
pub(super) const TRANSFORM_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "MATLAB transforms convert to the row-major kernel contract",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::transform_converts_matlab_matrices_to_the_row_major_contract",
};
pub(super) const AFFINE_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Non-affine and wrongly shaped transforms are rejected",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::transform_rejects_non_affine_and_wrongly_shaped_matrices",
};
pub(super) const ERROR_OWNER_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Errors report the calling builtin identifier",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::errors_report_the_calling_builtin_identifier",
};
pub(super) const BOOLEAN_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Sphere and Boolean builders record typed operations",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::sphere_and_booleans_record_typed_operations",
};
pub(super) const SELECTION_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Face and edge selections carry role, cardinality, and lineage",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::face_and_edge_selections_carry_role_cardinality_and_lineage",
};
pub(super) const INTEGER_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Integer scalars are admitted and integer vectors rejected",
    location: "crates/runmat-runtime/src/builtins/geometry/generated/tests.rs::integer_scalars_are_admitted_and_integer_vectors_rejected",
};
pub(super) const OCCT_PROFILE_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::ProviderTest,
    label: "Exact profile sweeps are qualified with complete lineage",
    location: "crates/runmat-geometry/io/src/occt/feature_profile_tests.rs::exact_profile_sweeps_are_qualified_with_complete_lineage",
};

/// Paragraph shared by every feature-producing builder.
pub(super) const DEFERRED_REALIZATION: &str = "The call records construction intent only. It appends one operation to an immutable, content-addressed feature graph and returns a new `geometry.Feature`; it does not run a CAD kernel, tessellate, or touch the filesystem. The graph is realized as exact B-rep topology when a `.fea` geometry declaration that imports the generating function is materialized for meshing or analysis.";

/// Paragraph shared by every builder that takes physical values.
pub(super) const SI_VALUES: &str = "All lengths are meters and all angles are radians; there is no unit option. Scalars accept `double`, `single`, or an exactly representable integer-class scalar. Vector arguments must be real, finite, non-integer-class numeric arrays with the stated element count; row and column orientation are both accepted.";

/// Paragraph shared by every builder that returns a `geometry.Feature`.
pub(super) const VALIDATION_BEFORE_RESULT: &str = "Every failure, including the wrong number of arguments, a value of the wrong kind, and a value that violates the feature-graph contract (a nonpositive or non-finite length, radius, depth, distance, or angle, or an all-zero direction), raises the builtin's own `RunMat:geometry:<name>:InvalidInput` identifier, and no value is returned. Feature identities are derived from a digest of each operation, so repeating the same construction yields the same feature identity.";

pub(super) const USE_IN_A_STUDY: &str = "Return the finished part from a pure `.m` function: `geometry.finish(feature)` turns the final feature into a `geometry.Generated` product, and `geometry.withRegion` publishes typed selections under names. A `.fea` geometry declaration written as `generated by function.<name>` calls that function, realizes the exact geometry, and exposes each named selection as `output.region.<name>` for meshing, material assignment, constraints, and loads.";

pub(super) const REALIZATION_FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Can I display or mesh a feature directly in a script?",
    answer: "Not yet. `geometry.Feature` and `geometry.Generated` are construction values; `mesh` and `meshing.generate` accept a loaded `geometry.Asset`. Realization happens when a `.fea` geometry declaration imports the generating function.",
};
pub(super) const UNITS_FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "How do I enter millimeter dimensions?",
    answer: "Convert to meters before the call, for example `geometry.box(80e-3, 50e-3, 8e-3)`. When the function is called from a `.fea` declaration, `length` inputs already arrive in meters.",
};

pub(super) struct ConstructionDocumentation {
    pub title: &'static str,
    pub summary: &'static str,
    pub description: &'static str,
    pub keywords: &'static [&'static str],
    pub related: &'static [&'static str],
    pub sections: &'static [BuiltinDocumentationSection],
    pub examples: &'static [BuiltinExample],
    pub faqs: &'static [BuiltinDocumentationFaq],
    pub implementation: &'static [BuiltinDocumentationLink],
    pub verification: &'static [BuiltinEvidenceReference],
    pub notes: &'static [&'static str],
    pub status: BuiltinDocumentationStatus,
}

pub(super) const fn documentation(definition: ConstructionDocumentation) -> BuiltinDocumentation {
    BuiltinDocumentation {
        authority: BuiltinDocumentationAuthority::Catalog,
        title: Some(definition.title),
        slug: Some(definition.title),
        summary: definition.summary,
        description: definition.description,
        keywords: definition.keywords,
        related: definition.related,
        sections: definition.sections,
        examples: definition.examples,
        example_exemption: None,
        faqs: definition.faqs,
        links: LINKS,
        media: &[],
        evidence: BuiltinDocumentationEvidence {
            implementation: definition.implementation,
            verification: definition.verification,
            notes: definition.notes,
        },
        introduced: None,
        status: Some(definition.status),
    }
}

/// Portable, fixture-free example verified by assertions.
pub(super) const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    assertions: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}

/// Portable example whose documented outcome is a typed error.
pub(super) const fn error_example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    identifier: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError { identifier },
    }
}

/// Pure host constructor layout shared by every builder identity.
pub(super) const fn entry(
    provenance: BuiltinCatalogProvenance,
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    inference: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance,
        identity: BuiltinCatalogIdentity { name },
        category: "geometry/construction",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::RunMat,
            async_behavior: BuiltinAsyncBehavior::NeverSuspends,
            purity: BuiltinPurity::Pure,
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: &[EffectKind::MayThrow],
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::Host,
            fusion: BuiltinFusionPolicy::Boundary,
            distributed: BuiltinDistributedPolicy::Unsupported,
        },
        link: BuiltinLinkContract {
            reachability: BuiltinReachability::Always,
            policy: BuiltinLinkPolicy::PortableRuntime,
            execution_stack: ExecutionStackRequirement::Any,
            artifact_dependencies: &[],
        },
        bindings: &REQUIRED_DEFAULT_BINDING,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: Some(&GEOMETRY_CONSTRUCTION_INTEGER_AUDIT),
        suppress_auto_output: false,
    }
}
