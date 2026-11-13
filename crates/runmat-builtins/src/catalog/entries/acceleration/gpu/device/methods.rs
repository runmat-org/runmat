//! Canonical executable members installed by the actual GPUDevice class owner.
use super::GpuDeviceProperty;
use crate::*;
use runmat_types::*;

pub const LIFECYCLE_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.LIFECYCLE", identifier: Some("RunMat:gpuDevice:Lifecycle"),
    when: "The handle/session/lease is invalid, a backend fact or control capability is unsupported, or physical reset/wait fails.",
    message: "GPUDevice lifecycle or property access failed",
};
const INPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "device",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A session-owned parallel.gpu.GPUDevice handle.",
}];
const OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Backend-reported property value, or double zero after successful wait/reset.",
}];
const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "value = member(device)",
        inputs: INPUT,
        outputs: OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[LIFECYCLE_ERROR],
};
const AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "The input is an issued GPUDevice handle, never a numeric resource identifier; properties retain their canonical logical, character or double class.",
};
fn infer_member(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    output: ValueFact,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM.GPU_DEVICE.MEMBER_ARITY",
            "GPUDevice member requires one handle",
            0,
        ));
    } else if !match &request.arguments[0].kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => {
            object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class.is(super::GPU_DEVICE_CLASS))
                && object.handle_semantics != Some(false)
                && request.arguments[0]
                    .shape
                    .element_count()
                    .is_none_or(|count| count == 1)
        }
        _ => false,
    } {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM.GPU_DEVICE.INVALID_OBJECT",
            "GPUDevice member requires a GPUDevice handle",
            0,
        ));
    }
    crate::catalog::inference::finish_fixed(entry, request, output, diagnostics)
}
fn status(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_member(request, entry, GpuDeviceProperty::Index.fact())
}
macro_rules! member {
    ($entry:ident, $infer:ident, $name:literal, $property:ident) => {
        fn $infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
            infer_member(request, entry, GpuDeviceProperty::$property.fact())
        }
        pub(super) const $entry: BuiltinCatalogEntry = make_entry($name, $infer, false);
    };
}
member!(NAME, infer_name, "__runmat_gpudevice_name", Name);
member!(INDEX, infer_index, "__runmat_gpudevice_index", Index);
member!(
    TOTAL_MEMORY,
    infer_total,
    "__runmat_gpudevice_total_memory",
    TotalMemory
);
member!(
    AVAILABLE_MEMORY,
    infer_free,
    "__runmat_gpudevice_available_memory",
    AvailableMemory
);
member!(
    SUPPORTS_DOUBLE,
    infer_double,
    "__runmat_gpudevice_supports_double",
    SupportsDouble
);
member!(
    AVAILABLE,
    infer_available,
    "__runmat_gpudevice_available",
    DeviceAvailable
);
member!(
    SELECTED,
    infer_selected,
    "__runmat_gpudevice_selected",
    DeviceSelected
);
pub(super) const RESET: BuiltinCatalogEntry = make_entry("__runmat_gpudevice_reset", status, true);
pub(super) const WAIT: BuiltinCatalogEntry = make_entry("__runmat_gpudevice_wait", status, true);

const fn make_entry(
    name: &'static str,
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    control: bool,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name }, category: "acceleration/gpu",
        documentation: BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog, title: Some(name), slug: Some(name),
            summary: "Executable GPUDevice property/control member.",
            description: "MethodOnly executable target of the canonical parallel.gpu.GPUDevice class. The session-owned object identifies the actual provider; reads use provider capabilities, and unsupported measurements are errors. Reset awaits physical completion and retires only the requesting session's allocation epochs; failed disposition remains fenced. Wait preserves peer allocations.",
            keywords: &["gpuDevice", "GPUDevice", "device property", "reset", "wait"],
            related: &["gpuDevice", "gpuInfo"],
            example_exemption: Some("MethodOnly GPUDevice member; exercised through public gpuDevice property/reset/wait examples and owner isolation fixtures."),
            ..BuiltinDocumentation::EMPTY
        },
        descriptor: &DESCRIPTOR,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: if control { BuiltinAsyncBehavior::MaySuspend } else { BuiltinAsyncBehavior::NeverSuspends },
            purity: if control { BuiltinPurity::Impure } else { BuiltinPurity::DeterministicReadOnly },
            semantic_kind: BuiltinSemanticKind::General, workspace_effect: None, environment_effect: None,
            effects: if control { &[EffectKind::EnvironmentRead, EffectKind::EnvironmentWrite, EffectKind::MaySuspend, EffectKind::MayThrow] }
                else { &[EffectKind::EnvironmentRead, EffectKind::MayThrow] },
            capabilities: &[CapabilityRequirement::Accelerator],
        },
        placement: super::GPU_DEVICE_CATALOG_ENTRY.placement,
        link: super::GPU_DEVICE_CATALOG_ENTRY.link,
        bindings: &REQUIRED_DEFAULT_BINDING, extensions: &[], integer_capabilities: &[],
        integer_audit: Some(&AUDIT), suppress_auto_output: true,
    }
}
