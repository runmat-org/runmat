//! Canonical contracts for public object save and load hooks.

use crate::*;
use runmat_types::{CallContract, CallInference, CallRequest, DynamicReason};

const SAVE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Object to serialize.",
}];
const SAVE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Serialized object representation.",
}];
const SAVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "b = saveobj(a)",
    inputs: &SAVE_INPUT,
    outputs: &SAVE_OUTPUT,
}];
const LOAD_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Serialized object representation.",
}];
const LOAD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Deserialized object.",
}];
const LOAD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "b = loadobj(a)",
    inputs: &LOAD_INPUT,
    outputs: &LOAD_OUTPUT,
}];

pub const SERIALIZATION_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OBJECT_SERIALIZATION.INVALID_ARGUMENT",
    identifier: Some("RunMat:ObjectSerialization:InvalidArgument"),
    when: "The input is outside the supported object or payload forms.",
    message: "object serialization: invalid argument",
};
pub const SERIALIZATION_ERROR_DISPATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OBJECT_SERIALIZATION.DISPATCH",
    identifier: Some("RunMat:ObjectSerialization:Dispatch"),
    when: "A saveobj or loadobj method cannot be resolved or executed.",
    message: "object serialization: method dispatch failed",
};
pub const SERIALIZATION_ERROR_RECURSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OBJECT_SERIALIZATION.RECURSION",
    identifier: Some("RunMat:ObjectSerialization:RecursionLimit"),
    when: "Object serialization nesting exceeds the supported recursion limit.",
    message: "object serialization: recursion limit exceeded",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    SERIALIZATION_ERROR_ARGUMENT,
    SERIALIZATION_ERROR_DISPATCH,
    SERIALIZATION_ERROR_RECURSION,
];
pub const SAVEOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SAVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const LOADOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const SAVEOBJ_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "saveobj dispatches only on value or handle objects; direct typed-integer input is invalid. Integer-valued object properties remain opaque nested payloads and are copied through authoritative storage without becoming a direct integer call form." };
pub const LOADOBJ_PASSTHROUGH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "loadobj-plain-payload-passthrough",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "loadobj passthrough for a plain non-object, non-structure payload is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LoadobjPlainPayloadPassthroughExtension"),
};
pub const LOADOBJ_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LOADOBJ_PASSTHROUGH_EXTENSION];
const LOAD_INTEGER_PAYLOAD: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "a.integer_payload", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "An object or structure supplied to loadobj may contain fields of every integer class; nested payloads retain exact class, shape, and value through recursive restoration and class loadobj dispatch." }];
const LOAD_INTEGER_DIRECT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "integer_a", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "A direct integer payload is outside the documented object-or-structure surface and is admitted only in RunMat mode as an exact passthrough." }];
pub const LOADOBJ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "b = loadobj(a_with_integer_payload)", inputs: &LOAD_INTEGER_PAYLOAD, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Integer fields are opaque serialized payloads rather than numeric controls and are never materialized as floating point by the default restoration path." },
    BuiltinIntegerCapabilityDescriptor { form: "b = loadobj(integer_a)", inputs: &LOAD_INTEGER_DIRECT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "RunMat mode preserves a direct integer scalar or array without conversion; MATLAB-compatible modes reject this undocumented plain-payload form before recursive restoration." },
];

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::DynamicDispatch);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    runmat_types::infer_call(&contract, request)
}

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Object serialization runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/object_serialization.rs") };
const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Object save/load recursion and overloads",
        location: "builtins::introspection::object_serialization::tests",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "Object saveobj/loadobj script surface",
        location: "crates/runmat-vm/tests/functions.rs::object_saveobj_loadobj_script_surface",
    },
];
const POINT_CLASS: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "point-class" },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File {
        relative_path: "Point.m",
        content: BuiltinFixtureContent::Utf8("classdef Point\n  properties\n    x\n  end\n  methods\n    function obj = Point(x)\n      obj.x = x;\n    end\n  end\nend\n"),
    }],
});
const RESTORING_POINT_CLASS: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "restoring-point-class" },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File {
        relative_path: "RestoringPoint.m",
        content: BuiltinFixtureContent::Utf8("classdef RestoringPoint\n  properties\n    x\n  end\n  methods\n    function obj = RestoringPoint(x)\n      obj.x = x;\n    end\n  end\n  methods(Static)\n    function obj = loadobj(payload)\n      obj = RestoringPoint(payload.x + 1);\n    end\n  end\nend\n"),
    }],
});
const SAVE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("saveobj"), slug: Some("saveobj"),
    summary: "Return the MAT-file serialization payload for an object.",
    description: "`saveobj(a)` returns the payload that `save` should write for object `a`. If the object's class registers a `saveobj` method, RunMat dispatches to that method. Otherwise RunMat returns a scalar struct containing the object's stored public runtime properties.",
    keywords: &["saveobj", "loadobj", "object serialization", "mat", "classdef"], related: &["loadobj", "save", "load"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The public signature is `b = saveobj(a)`.",
            "Object and handle-object inputs dispatch to a registered `Class.saveobj` method when one is available.",
            "When no registered overload exists, RunMat serializes the object's stored property map as a scalar struct.",
            "Nested objects inside structs or cells are serialized recursively before MAT-file encoding.",
            "`save` automatically applies this object serialization path before writing MAT-file variables.",
            "Recursion is bounded and reports `RunMat:ObjectSerialization:RecursionLimit` for cyclic or excessively nested object graphs.",
            "Non-object inputs to direct `saveobj`, including typed integers and gpuArray values, raise `RunMat:ObjectSerialization:InvalidArgument` before serialization. Integer-valued properties nested inside an object remain opaque payloads and preserve authoritative native storage.",
        ] },
        BuiltinDocumentationSection { heading: "Host and GPU behavior", paragraphs: &[
            "Object serialization is host-side metadata and property traversal. GPU-resident property values are gathered later by MAT-file encoding when needed.",
            "`saveobj` does not launch provider kernels. `save` remains the residency sink for any GPU values that appear inside the serialized payload.",
            "The operation is not elementwise or a reduction, does not broadcast, and ends fusion. It requires no optional feature.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "default-property-payload", title: "Return an object's default property payload", program: "obj = Point(7);\npayload = saveobj(obj);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: POINT_CLASS, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(payload, 'struct')); assert(payload.x == 7);" } },
        BuiltinExample { id: "save-object-mat", title: "Save an object to an isolated MAT-file", program: "obj = Point(7);\nsave('object_state.mat', 'obj');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: POINT_CLASS, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "stored = load('object_state.mat'); assert(isa(stored.obj, 'Point')); assert(stored.obj.x == 7);" } },
        BuiltinExample { id: "reject-non-object", title: "Reject a direct integer input", program: "saveobj(uint64(7));", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::ExpectedError { identifier: "MATLAB:ObjectSerialization:InvalidArgument" } },
    ], example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `save` call `saveobj` automatically?", answer: "Yes. Before MAT-file encoding, RunMat walks selected values and applies registered `saveobj` methods for object and handle-object values." },
        BuiltinDocumentationFaq { question: "What is saved when no class-specific method exists?", answer: "RunMat writes a typed Level-5 object record carrying the class name and a struct of stored properties, so `load` can reconstruct the object." },
        BuiltinDocumentationFaq { question: "Are handle identities preserved?", answer: "No. Current support serializes the referenced object's properties. It does not preserve handle identity, listener state, or aliasing relationships across save/load." },
    ],
    links: &[BuiltinDocumentationLink { label: "loadobj", target: BuiltinDocumentationLinkTarget::Documentation("./loadobj") }, BuiltinDocumentationLink { label: "save", target: BuiltinDocumentationLinkTarget::Documentation("./save") }, BuiltinDocumentationLink { label: "load", target: BuiltinDocumentationLinkTarget::Documentation("./load") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: VERIFICATION, notes: &[] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
const LOAD_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("loadobj"), slug: Some("loadobj"),
    summary: "Restore an object payload produced during MAT-file loading.",
    description: "`loadobj(a)` accepts an object or structure payload. During MAT-file loading, typed Level-5 object records are restored separately: a registered static `Class.loadobj` method receives the saved payload when available.",
    keywords: &["loadobj", "saveobj", "object serialization", "mat", "classdef"], related: &["saveobj", "load"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "The public signature is `b = loadobj(a)`.",
            "`load` recognizes typed Level-5 object records written by `save`. Ordinary structs are never treated as object records because of their field names.",
            "If the saved class registers `Class.loadobj`, that method receives the restored payload and its return value becomes the loaded variable.",
            "If no registered `loadobj` method exists and the payload is a scalar struct, RunMat reconstructs an object with the saved class name and payload fields as properties.",
            "In RunMat mode, plain scalar and cell inputs pass through recursive restoration unchanged; MATLAB-compatible modes reject this undocumented convenience through the `loadobj-plain-payload-passthrough` compatibility gate.",
            "All eight integer classes nested in object or structure payloads retain their exact values, shape, signedness, and width through recursive restoration and class-specific `loadobj` dispatch.",
            "Recursion is bounded and reports `RunMat:ObjectSerialization:RecursionLimit` for cyclic or excessively nested payloads.",
        ] },
        BuiltinDocumentationSection { heading: "Host and GPU behavior", paragraphs: &[
            "Object restoration is host-side metadata and property construction. Loaded values start CPU-resident like other `load` results.",
            "`loadobj` does not launch provider kernels. Loaded object properties are CPU-resident after MAT-file loading; downstream acceleration may later promote numeric properties through the usual RunMat Accelerate paths.",
            "The operation is not elementwise or a reduction, does not broadcast, and ends fusion. It requires no optional feature.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "load-object-mat", title: "Restore an object from an isolated MAT-file", program: "obj = Point(7);\nsave('object_state.mat', 'obj');\nclear obj;\nload('object_state.mat');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: POINT_CLASS, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(obj, 'Point')); assert(obj.x == 7);" } },
        BuiltinExample { id: "static-class-restoration", title: "Call a class-specific static loadobj method", program: "obj = RestoringPoint(7);\npayload = saveobj(obj);\nrestored = RestoringPoint.loadobj(payload);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: RESTORING_POINT_CLASS, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(restored, 'RestoringPoint')); assert(restored.x == 8);" } },
        BuiltinExample { id: "nested-integer", title: "Retain exact nested integer payloads", program: "payload = struct('x', uint64(7)); restored = loadobj(payload);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(restored.x,'uint64')); assert(restored.x == uint64(7));" } },
    ], example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `load` call `loadobj` automatically?", answer: "Yes for typed Level-5 object records written by RunMat. `load` restores nested payloads and calls registered static `Class.loadobj` methods when available." },
        BuiltinDocumentationFaq { question: "What happens without a class-specific `loadobj` method?", answer: "RunMat reconstructs a value object with the saved class name and property struct fields. If the saved payload was not a struct, the payload itself is returned." },
        BuiltinDocumentationFaq { question: "Does this support arbitrary MATLAB object MAT-files?", answer: "No. This support covers RunMat's typed Level-5 object records and runtime object model. MATLAB v7.3/HDF5 class storage, listener state, and handle alias preservation remain unsupported." },
    ],
    links: &[BuiltinDocumentationLink { label: "saveobj", target: BuiltinDocumentationLinkTarget::Builtin("saveobj") }, BuiltinDocumentationLink { label: "load", target: BuiltinDocumentationLinkTarget::Builtin("load") }], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: VERIFICATION, notes: &[] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

macro_rules! entry {
    ($name:literal, $doc:expr, $descriptor:expr, $extensions:expr, $integers:expr, $audit:expr) => {
        BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "introspection",
            documentation: $doc,
            descriptor: $descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::DynamicByDesign,
                inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Impure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[
                    runmat_types::EffectKind::HostCallback,
                    runmat_types::EffectKind::MaySuspend,
                    runmat_types::EffectKind::MayThrow,
                    runmat_types::EffectKind::Unknown,
                ],
                capabilities: &[],
            },
            placement: BuiltinPlacementContract {
                preparation: BuiltinOperandPreparation::RuntimeOwned,
                portability: BuiltinPortability::NativeAndWasm,
                accelerator: BuiltinAcceleratorPolicy::Optional,
                residency: BuiltinResidencyPolicy::Dynamic,
                fusion: BuiltinFusionPolicy::Boundary,
                distributed: BuiltinDistributedPolicy::Unsupported,
            },
            link: BuiltinLinkContract {
                reachability: BuiltinReachability::Dynamic,
                policy: BuiltinLinkPolicy::PortableRuntime,
                execution_stack: runmat_types::ExecutionStackRequirement::Process,
                artifact_dependencies: &[],
            },
            bindings: &REQUIRED_DEFAULT_BINDING,
            extensions: $extensions,
            integer_capabilities: $integers,
            integer_audit: $audit,
            suppress_auto_output: false,
        }
    };
}
pub const SAVEOBJ_CATALOG_ENTRY: BuiltinCatalogEntry = entry!(
    "saveobj",
    SAVE_DOCUMENTATION,
    &SAVEOBJ_DESCRIPTOR,
    &[],
    &[],
    Some(&SAVEOBJ_INTEGER_AUDIT)
);
pub const LOADOBJ_CATALOG_ENTRY: BuiltinCatalogEntry = entry!(
    "loadobj",
    LOAD_DOCUMENTATION,
    &LOADOBJ_DESCRIPTOR,
    &LOADOBJ_EXTENSIONS,
    &LOADOBJ_INTEGER_CAPABILITIES,
    None
);
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&SAVEOBJ_CATALOG_ENTRY, &LOADOBJ_CATALOG_ENTRY];
