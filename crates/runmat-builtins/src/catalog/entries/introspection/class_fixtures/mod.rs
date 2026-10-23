use crate::*;
mod contract;
mod inference;
#[cfg(test)]
mod tests;
mod documentation;
pub use contract::*;
const fn declaration(
    handler: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
) -> BuiltinContractDeclaration {
    BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(handler)),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[runmat_types::EffectKind::MayThrow],
        capabilities: &[],
    }
}
const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Dynamic,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Dynamic,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: runmat_types::ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};
const INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor {
        form: "native scalar integers in fixture-specific controls or payload",
        inputs: &[BuiltinIntegerInputCapability {
            name: "fixture arguments",
            classes: &ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: "Scalar arithmetic operands use existing f64 conversion and may round 64-bit values. Structural properties and unary identity preserve exact payloads; method-specific receiver/name restrictions still apply. Arithmetic integer arrays reject without gather; structural payload/identity routes preserve their original class and backing.",
        }],
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Internal fixture semantics, not a general integer overload claim; see the identity documentation and inference for exact conversions and outputs.",
    },
];
pub const POINT_MOVE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "Point.move" },
    category: "introspection",
    documentation: documentation::POINT_MOVE_DOCUMENTATION,
    descriptor: &POINT_MOVE_DESCRIPTOR,
    contract: declaration(inference::point_move),
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub const POINT_ORIGIN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "Point.origin",
    },
    category: "introspection",
    documentation: documentation::POINT_ORIGIN_DOCUMENTATION,
    descriptor: &POINT_ORIGIN_DESCRIPTOR,
    contract: declaration(inference::point_origin),
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&NO_INTEGER_OPERANDS),
    suppress_auto_output: false,
};
pub const SHAPE_AREA_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "Shape.area" },
    category: "introspection",
    documentation: documentation::SHAPE_AREA_DOCUMENTATION,
    descriptor: &SHAPE_AREA_DESCRIPTOR,
    contract: declaration(inference::shape_area),
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub const CIRCLE_AREA_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "Circle.area",
    },
    category: "introspection",
    documentation: documentation::CIRCLE_AREA_DOCUMENTATION,
    descriptor: &CIRCLE_AREA_DESCRIPTOR,
    contract: declaration(inference::circle_area),
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub const CTOR_CTOR_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "Ctor.Ctor" },
    category: "introspection",
    documentation: documentation::CTOR_CTOR_DOCUMENTATION,
    descriptor: &CTOR_CTOR_DESCRIPTOR,
    contract: declaration(inference::ctor_ctor),
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub const PKGF_FOO_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "PkgF.foo" },
    category: "introspection",
    documentation: documentation::PKGF_FOO_DOCUMENTATION,
    descriptor: &PKGF_FOO_DESCRIPTOR,
    contract: declaration(inference::pkgf_foo),
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&NO_INTEGER_OPERANDS),
    suppress_auto_output: false,
};
pub const PKGG_FOO_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Fixture,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "PkgG.foo" },
    category: "introspection",
    documentation: documentation::PKGG_FOO_DOCUMENTATION,
    descriptor: &PKGG_FOO_DESCRIPTOR,
    contract: declaration(inference::pkgg_foo),
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&NO_INTEGER_OPERANDS),
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &POINT_MOVE_CATALOG_ENTRY,
    &POINT_ORIGIN_CATALOG_ENTRY,
    &SHAPE_AREA_CATALOG_ENTRY,
    &CIRCLE_AREA_CATALOG_ENTRY,
    &CTOR_CTOR_CATALOG_ENTRY,
    &PKGF_FOO_CATALOG_ENTRY,
    &PKGG_FOO_CATALOG_ENTRY,
];
const NO_INTEGER_OPERANDS:BuiltinIntegerAuditDescriptor=BuiltinIntegerAuditDescriptor{
kind:BuiltinIntegerAuditKind::NotApplicable,
canonical_builtin:None,
notes:"This zero-input fixture has no integer operand domain; its object or double output is specified by identity."}
;
