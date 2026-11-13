mod contract;
mod documentation;
mod examples;
mod inference;
use crate::*;
pub use contract::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};
pub const UIGETDIR_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
 audience: crate::BuiltinCatalogAudience::Public,
 provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
 identity: BuiltinCatalogIdentity { name: "uigetdir" },
 category: "io/repl_fs",
 documentation: documentation::DOCUMENTATION,
 descriptor: &UIGETDIR_DESCRIPTOR,
 contract: BuiltinContractDeclaration {
 maturity: BuiltinContractMaturity::Complete,
 inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::infer)),
 compatibility: BuiltinCompatibility::Matlab,
 async_behavior: BuiltinAsyncBehavior::MaySuspend,
 purity: BuiltinPurity::Impure,
 semantic_kind: BuiltinSemanticKind::General,
 workspace_effect: None, environment_effect: None,
 effects: &[EffectKind::EnvironmentRead, EffectKind::UserInterface, EffectKind::HostCallback,
 EffectKind::MaySuspend, EffectKind::MayThrow],
 capabilities: &[CapabilityRequirement::HostRuntime, CapabilityRequirement::UserInterface],
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
 reachability: BuiltinReachability::Always, policy: BuiltinLinkPolicy::PortableRuntime,
 execution_stack: ExecutionStackRequirement::Any, artifact_dependencies: &[],
 },
 bindings: &REQUIRED_DEFAULT_BINDING,
 extensions: &[], integer_capabilities: &[],
 integer_audit: Some(&BuiltinIntegerAuditDescriptor {
 kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
 notes: "The optional path and title are scalar text. Numeric values reject before provider access. Cancellation is represented by a host double zero.",
 }),
 suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&UIGETDIR_CATALOG_ENTRY];
