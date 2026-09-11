use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

use super::{
    documentation, MAT2CELL_DESCRIPTOR, MAT2CELL_EXTENSIONS, MAT2CELL_INTEGER_CAPABILITIES,
};

const EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];

pub const MAT2CELL_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "mat2cell" },
    category: "cells/core",
    documentation: documentation::DOCUMENTATION,
    descriptor: &MAT2CELL_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(super::infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: MAT2CELL_EXTENSIONS,
    integer_capabilities: MAT2CELL_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
