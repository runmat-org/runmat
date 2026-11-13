use super::{helper_documentation as docs, helper_inference as infer, *};
use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};
pub(super) const fn entry(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    inference: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
    random: bool,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "math/optim",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Complete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference)),
            compatibility: BuiltinCompatibility::Matlab,
            async_behavior: BuiltinAsyncBehavior::MaySuspend,
            purity: if random {
                BuiltinPurity::Impure
            } else {
                BuiltinPurity::Pure
            },
            semantic_kind: BuiltinSemanticKind::General,
            workspace_effect: None,
            environment_effect: None,
            effects: if random {
                &[
                    EffectKind::Randomness,
                    EffectKind::MaySuspend,
                    EffectKind::MayThrow,
                ]
            } else {
                &[EffectKind::MaySuspend, EffectKind::MayThrow]
            },
            capabilities: &[],
        },
        placement: BuiltinPlacementContract {
            preparation: BuiltinOperandPreparation::RuntimeOwned,
            portability: BuiltinPortability::NativeAndWasm,
            accelerator: BuiltinAcceleratorPolicy::Forbidden,
            residency: BuiltinResidencyPolicy::GatherToHost,
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
        extensions: super::descriptor::EXTENSIONS,
        integer_capabilities: super::helper_descriptors::INTEGERS,
        integer_audit: None,
        suppress_auto_output: false,
    }
}
pub(super) const fn creator(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    documentation: BuiltinDocumentation,
    inference: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
    clock: bool,
    nonlinear: bool,
) -> BuiltinCatalogEntry {
    let mut result = entry(name, descriptor, documentation, inference, true);
    result.integer_capabilities = super::descriptor::INTEGERS;
    result.contract.effects = if nonlinear {
        &[
            EffectKind::FilesystemWrite,
            EffectKind::UserInterface,
            EffectKind::Randomness,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown,
        ]
    } else if clock {
        &[
            EffectKind::Randomness,
            EffectKind::Clock,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    } else {
        &[
            EffectKind::Randomness,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    if nonlinear {
        result.contract.maturity = BuiltinContractMaturity::DynamicByDesign;
        result.link.execution_stack = ExecutionStackRequirement::Process;
    }
    result
}

pub const SELECTIONSTOCHUNIF_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "selectionstochunif",
    &GA_SELECTION_HELPER_DESCRIPTOR,
    docs::doc(
        "selectionstochunif",
        "Sample equally spaced parent positions from scaled expectations.",
        docs::SELECTION_SECTIONS,
        &[docs::example(
            "p=selectionstochunif([0;3],5,[]);",
            "assert(isequal(p,2*ones(1,5)));",
        )],
    ),
    infer::stochastic,
    true,
);

pub const SELECTIONREMAINDER_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "selectionremainder",
    &GA_SELECTION_HELPER_DESCRIPTOR,
    docs::doc(
        "selectionremainder",
        "Combine deterministic offspring counts with fractional roulette selection.",
        docs::SELECTION_SECTIONS,
        &[docs::example(
            "p=selectionremainder([0;2;1],3,[]);",
            "assert(isequal(p,[2 2 3]));",
        )],
    ),
    infer::remainder,
    true,
);

pub const SELECTIONUNIFORM_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "selectionuniform",
    &GA_SELECTION_HELPER_DESCRIPTOR,
    docs::doc(
        "selectionuniform",
        "Select parent indices uniformly from a population.",
        docs::SELECTION_SECTIONS,
        &[docs::example(
            "p=selectionuniform([1;1;1],8,[]);",
            "assert(isequal(size(p),[1 8])); assert(all(p>=1 & p<=3 & p==floor(p)));",
        )],
    ),
    infer::uniform,
    true,
);

pub const SELECTIONROULETTE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "selectionroulette",
    &GA_SELECTION_HELPER_DESCRIPTOR,
    docs::doc(
        "selectionroulette",
        "Select independent parents from scaled roulette weights.",
        docs::SELECTION_SECTIONS,
        &[docs::example(
            "p=selectionroulette([0;1;0],4,[]);",
            "assert(isequal(p,[2 2 2 2]));",
        )],
    ),
    infer::roulette,
    true,
);

pub const SELECTIONTOURNAMENT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "selectiontournament",
    &GA_TOURNAMENT_HELPER_DESCRIPTOR,
    docs::doc(
        "selectiontournament",
        "Choose each parent from a tournament of scaled expectations.",
        docs::SELECTION_SECTIONS,
        &[docs::example(
            "p=selectiontournament(1,4,[],2);",
            "assert(isequal(p,[1 1 1 1]));",
        )],
    ),
    infer::tournament,
    true,
);

pub const FITSCALINGRANK_CATALOG_ENTRY:BuiltinCatalogEntry=entry("fitscalingrank",&GA_SCALING_HELPER_DESCRIPTOR,
    docs::doc("fitscalingrank","Scale sorted fitness by inverse square-root rank.",docs::SCALING_SECTIONS,
        &[docs::example("e=fitscalingrank([1;4;9],6);","assert(isequal(size(e),[3 1])); assert(abs(sum(e)-6)<1e-12 && e(1)>e(2) && e(2)>e(3));")]),infer::rank,false);

pub const FITSCALINGPROP_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "fitscalingprop",
    &GA_SCALING_HELPER_DESCRIPTOR,
    docs::doc(
        "fitscalingprop",
        "Normalize finite nonnegative raw fitness proportionally.",
        docs::SCALING_SECTIONS,
        &[docs::example(
            "e=fitscalingprop([1;2;3],6);",
            "assert(all(abs(e-[1;2;3])<1e-12));",
        )],
    ),
    infer::proportional,
    false,
);

pub const FITSCALINGTOP_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "fitscalingtop",
    &GA_PARAMETRIC_SCALING_HELPER_DESCRIPTOR,
    docs::doc(
        "fitscalingtop",
        "Assign equal offspring expectations to the best selected scores.",
        docs::SCALING_SECTIONS,
        &[docs::example(
            "e=fitscalingtop([4;1;3;2],6,0.5);",
            "assert(isequal(e,[0;3;0;3]));",
        )],
    ),
    infer::top,
    false,
);

pub const FITSCALINGSHIFTLINEAR_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "fitscalingshiftlinear",
    &GA_PARAMETRIC_SCALING_HELPER_DESCRIPTOR,
    docs::doc(
        "fitscalingshiftlinear",
        "Use nonnegative affine scaling with a bounded best-to-mean ratio.",
        docs::SCALING_SECTIONS,
        &[docs::example(
            "e=fitscalingshiftlinear([0;1;2],3,2);",
            "assert(all(abs(e-[2;1;0])<1e-12));",
        )],
    ),
    infer::shift_linear,
    false,
);
