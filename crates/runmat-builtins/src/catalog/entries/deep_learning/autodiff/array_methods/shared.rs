use crate::*;

macro_rules! array_method_entry {
    ($constant:ident, $name:literal, $descriptor:ident, $documentation:expr, $inference:path) => {
        pub const $constant: BuiltinCatalogEntry = BuiltinCatalogEntry {
            audience: crate::BuiltinCatalogAudience::Public,
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            identity: BuiltinCatalogIdentity { name: $name },
            category: "deep_learning",
            documentation: $documentation,
            descriptor: &super::contract::$descriptor,
            contract: BuiltinContractDeclaration {
                maturity: BuiltinContractMaturity::Incomplete,
                inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
                    $inference
                        as fn(
                            &runmat_types::CallRequest,
                            &BuiltinCatalogEntry,
                        ) -> runmat_types::CallInference,
                )),
                compatibility: BuiltinCompatibility::Matlab,
                async_behavior: BuiltinAsyncBehavior::NeverSuspends,
                purity: BuiltinPurity::Impure,
                semantic_kind: BuiltinSemanticKind::General,
                workspace_effect: None,
                environment_effect: None,
                effects: &[
                    runmat_types::EffectKind::EnvironmentRead,
                    runmat_types::EffectKind::EnvironmentWrite,
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
            integer_capabilities: &super::contract::INTEGER_CAPABILITIES,
            integer_audit: None,
            suppress_auto_output: false,
        };
    };
}
pub(super) use array_method_entry;

macro_rules! array_method_documentation {
    ($title:literal, $summary:literal, $description:literal, $program:literal, $verification:literal) => {
        BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($title),
            slug: Some($title),
            summary: $summary,
            description: $description,
            keywords: &["dlarray", "deep learning", "automatic differentiation", $title],
            related: &["dlarray", "dlgradient", "dlfeval"],
            sections: &[
                BuiltinDocumentationSection {
                    heading: "Values and precision",
                    paragraphs: &["This host method returns a dlarray. Real double and single Data are supported. Logical Data are read as 0/1 double data at this arithmetic boundary; native integer and complex Data are not admitted. A single operand makes the floating result single, regardless of operand order; otherwise the result is double. The dlarray operand supplies Format and Labels when paired with an ordinary numeric operand."],
                },
                BuiltinDocumentationSection {
                    heading: "Tracing and placement",
                    paragraphs: &["Inside dlfeval the host reverse-mode tape records supported arithmetic for dlgradient. Provider-resident dlarray Data are not silently downloaded for this host tape; the operation reports that resident tape kernels are unavailable. The current gradient subset includes scalar broadcasting, not general higher-rank implicit-expansion reduction."],
                },
            ],
            examples: &[BuiltinExample {
                id: "typed-host-arithmetic",
                title: "Compute with a typed dlarray",
                program: $program,
                display_output: None,
                compatibility: BuiltinExampleCompatibility::RunMat,
                harness: BuiltinExampleHarness::Portable,
                fixture: BuiltinExampleFixture::None,
                requirements: BuiltinExampleRequirements::NONE,
                verification: BuiltinExampleVerification::Assertions { source: $verification },
            }],
            example_exemption: None,
            faqs: &[
                BuiltinDocumentationFaq { question: "Does the method preserve single precision?", answer: "Yes. The result uses single when either real floating input uses single, in either operand order." },
                BuiltinDocumentationFaq { question: "Can this method differentiate a GPU-backed dlarray?", answer: "No. The current reverse-mode tape runs on the host and rejects resident dlarray Data rather than implicitly downloading it." },
            ],
            links: &[shared::SOURCE, BuiltinDocumentationLink { label: "dlgradient", target: BuiltinDocumentationLinkTarget::Builtin("dlgradient") }],
            media: &[],
            evidence: shared::EVIDENCE,
            introduced: None,
            status: Some(BuiltinDocumentationStatus::Partial),
        }
    };
}
pub(super) use array_method_documentation;

pub(super) const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Host dlarray arithmetic and tape recording",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/autodiff/array_methods.rs",
    ),
};

pub(super) const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[SOURCE],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Host dlarray arithmetic, precision and gradients",
        location: "crates/runmat-runtime/src/builtins/deep_learning/autodiff/tests.rs",
    }],
    notes: &["Native, JIT, AOT, WASM, and standalone-example execution remain integrated qualification gates."],
};
