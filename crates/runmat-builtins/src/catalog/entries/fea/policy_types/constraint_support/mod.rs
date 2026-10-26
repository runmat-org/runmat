use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

mod contract;
mod examples;
mod inference;
pub use contract::*;

macro_rules! entry {
    ($constant:ident, $name:literal, $documentation:ident, $descriptor:ident, $infer:path, $integers:expr, $audit:expr) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: BuiltinCatalogAudience::Public,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "fea",
            documentation: $documentation,
            descriptor: &$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Incomplete,
                inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new($infer)),
                compatibility: BuiltinCompatibility::RunMat,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Pure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[
                    runmat_types::EffectKind::MaySuspend,
                    runmat_types::EffectKind::MayThrow,
                ],
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
                execution_stack: runmat_types::ExecutionStackRequirement::Any,
                artifact_dependencies: &[],
            },
            bindings: &REQUIRED_DEFAULT_BINDING,
            extensions: &[],
            integer_capabilities: $integers,
            integer_audit: $audit,
            suppress_auto_output: false,
        };
    };
}

const DIRECTION_DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(
    ConstructorDocumentationDefinition {
        title: "fea.Direction.vector",
        summary: "Create a normalized typed direction from three components.",
        description:
            "Creates an immutable typed FEA direction from a finite nonzero three-component vector.",
        keywords: &["fea", "direction", "axis", "frame", "typed"],
        related: &[],
        sections: &[],
        examples: examples::DIRECTION,
        faqs: &[],
        links: &[],
        implementation: &[],
        verification: &[],
        notes: &[],
    },
);
const MATCHING_DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.PeriodicPairing.matching",
        summary: "Require an already matching one-to-one periodic topology.",
        description: "Creates the typed matching periodic-pairing policy.",
        keywords: &["fea", "periodic", "pairing", "matching", "typed"],
        related: &[],
        sections: &[],
        examples: examples::MATCHING,
        faqs: &[],
        links: &[],
        implementation: &[],
        verification: &[],
        notes: &[],
    });
const GEOMETRIC_DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(
    ConstructorDocumentationDefinition {
        title: "fea.PeriodicPairing.geometric",
        summary: "Pair periodic entities geometrically within a typed length tolerance.",
        description: "Creates the typed geometric periodic-pairing policy with a nonnegative fea.Length tolerance.",
        keywords: &["fea", "periodic", "pairing", "geometric", "tolerance", "typed"],
        related: &[], sections: &[], examples: examples::GEOMETRIC, faqs: &[], links: &[], implementation: &[], verification: &[], notes: &[],
    },
);
const IDENTITY_DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.PeriodicTransform.identity",
        summary: "Create an identity periodic transform.",
        description: "Creates the typed identity periodic transform.",
        keywords: &["fea", "periodic", "transform", "identity", "typed"],
        related: &[],
        sections: &[],
        examples: examples::IDENTITY,
        faqs: &[],
        links: &[],
        implementation: &[],
        verification: &[],
        notes: &[],
    });
const TRANSLATION_DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.PeriodicTransform.translation",
        summary: "Create a translational periodic transform.",
        description: "Creates a typed periodic transform from a fea.Translation value.",
        keywords: &["fea", "periodic", "transform", "translation", "typed"],
        related: &[],
        sections: &[],
        examples: examples::TRANSLATION,
        faqs: &[],
        links: &[],
        implementation: &[],
        verification: &[],
        notes: &[],
    });
const ROTATION_DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(
    ConstructorDocumentationDefinition {
        title: "fea.PeriodicTransform.rotation",
        summary: "Create a rotational periodic transform.",
        description:
            "Creates a typed periodic transform from an origin, direction axis, and rotation angle.",
        keywords: &["fea", "periodic", "transform", "rotation", "axis", "typed"],
        related: &[],
        sections: &[],
        examples: examples::ROTATION,
        faqs: &[],
        links: &[],
        implementation: &[],
        verification: &[],
        notes: &[],
    },
);

entry!(
    FEA_DIRECTION_VECTOR_CATALOG_ENTRY,
    "fea.Direction.vector",
    DIRECTION_DOCUMENTATION,
    FEA_DIRECTION_VECTOR_DESCRIPTOR,
    inference::direction,
    &FEA_DIRECTION_VECTOR_INTEGER_CAPABILITIES,
    None
);
entry!(
    FEA_PERIODIC_PAIRING_MATCHING_CATALOG_ENTRY,
    "fea.PeriodicPairing.matching",
    MATCHING_DOCUMENTATION,
    FEA_PERIODIC_PAIRING_MATCHING_DESCRIPTOR,
    inference::matching,
    &[],
    Some(&FEA_TYPED_CONSTRAINT_INTEGER_AUDIT)
);
entry!(
    FEA_PERIODIC_PAIRING_GEOMETRIC_CATALOG_ENTRY,
    "fea.PeriodicPairing.geometric",
    GEOMETRIC_DOCUMENTATION,
    FEA_PERIODIC_PAIRING_GEOMETRIC_DESCRIPTOR,
    inference::geometric,
    &[],
    Some(&FEA_TYPED_CONSTRAINT_INTEGER_AUDIT)
);
entry!(
    FEA_PERIODIC_TRANSFORM_IDENTITY_CATALOG_ENTRY,
    "fea.PeriodicTransform.identity",
    IDENTITY_DOCUMENTATION,
    FEA_PERIODIC_TRANSFORM_IDENTITY_DESCRIPTOR,
    inference::identity,
    &[],
    Some(&FEA_TYPED_CONSTRAINT_INTEGER_AUDIT)
);
entry!(
    FEA_PERIODIC_TRANSFORM_TRANSLATION_CATALOG_ENTRY,
    "fea.PeriodicTransform.translation",
    TRANSLATION_DOCUMENTATION,
    FEA_PERIODIC_TRANSFORM_TRANSLATION_DESCRIPTOR,
    inference::translation,
    &[],
    Some(&FEA_TYPED_CONSTRAINT_INTEGER_AUDIT)
);
entry!(
    FEA_PERIODIC_TRANSFORM_ROTATION_CATALOG_ENTRY,
    "fea.PeriodicTransform.rotation",
    ROTATION_DOCUMENTATION,
    FEA_PERIODIC_TRANSFORM_ROTATION_DESCRIPTOR,
    inference::rotation,
    &[],
    Some(&FEA_TYPED_CONSTRAINT_INTEGER_AUDIT)
);
