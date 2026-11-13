//! One shared contract shape for the timer family's catalog identities.

macro_rules! timer_effects {
    (TIMER) => {
        &[
            EffectKind::EnvironmentWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (TIMERFIND) => {
        &[
            EffectKind::EnvironmentRead,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (TIMERFINDALL) => {
        &[
            EffectKind::EnvironmentRead,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (START) => {
        &[
            EffectKind::EnvironmentWrite,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (STARTAT) => {
        &[
            EffectKind::EnvironmentWrite,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (STOP) => {
        &[
            EffectKind::EnvironmentWrite,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (WAIT) => {
        &[
            EffectKind::EnvironmentRead,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    (DELETE) => {
        &[
            EffectKind::EnvironmentWrite,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
    ($setter:ident) => {
        &[
            EffectKind::EnvironmentWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ]
    };
}

macro_rules! timer_entry {
    ($id:ident, $name:literal, $doc:expr, $descriptor:expr, $infer:expr,
     $stack:expr, $extensions:expr, $integers:expr, $auto_output:expr) => {
        pub const $id: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "timing",
            documentation: $doc,
            descriptor: $descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Incomplete,
                inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new($infer)),
                // MATLAB absolute startat forms coexist with explicitly gated
                // RunMat relative/status forms on the same method identity.
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                purity: BuiltinPurity::Impure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: timer_effects!($id),
                // HostRuntime currently represents the timer, local-task,
                // timing, and source-evaluation ports together.
                capabilities: &[CapabilityRequirement::HostRuntime],
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
                execution_stack: $stack,
                artifact_dependencies: &[],
            },
            bindings: &REQUIRED_DEFAULT_BINDING,
            extensions: $extensions,
            integer_capabilities: $integers,
            integer_audit: None,
            suppress_auto_output: $auto_output,
        };
    };
}
