mod descriptor;
mod documentation;

pub use descriptor::*;

use crate::*;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, NumericDomain,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-GRAY2RGB-ARITY",
            "gray2rgb requires one grayscale image",
            0,
        ));
    }
    let input = request.arguments.first();
    if let Some(input) = input {
        let supported = match input.kind {
            ValueKindFact::Numeric(fact) => fact.domain == NumericDomain::Real,
            ValueKindFact::Logical | ValueKindFact::Unknown => true,
            _ => false,
        };
        if !supported || input.storage == StorageFact::Sparse {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-GRAY2RGB-CLASS",
                "expected a dense real numeric or logical image",
                0,
            ));
        }
    }
    let shape = match input.map(|fact| &fact.shape) {
        Some(ShapeFact::Scalar) => ShapeFact::from(vec![Some(1), Some(1), Some(3)]),
        Some(ShapeFact::Shaped { dims }) if dims.len() == 2 => ShapeFact::Shaped {
            dims: vec![dims[0].clone(), dims[1].clone(), DimensionFact::Known(3)],
        },
        Some(ShapeFact::Ranked { rank: 2 }) => ShapeFact::from(vec![None, None, Some(3)]),
        Some(ShapeFact::Unknown) | None => ShapeFact::Unknown,
        Some(_) => {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-GRAY2RGB-SHAPE",
                "expected an MxN grayscale image",
                0,
            ));
            ShapeFact::Unknown
        }
    };
    let kind = match input.map(|fact| &fact.kind) {
        Some(ValueKindFact::Numeric(fact)) if fact.domain == NumericDomain::Real => {
            ValueKindFact::Numeric(*fact)
        }
        Some(ValueKindFact::Logical) => ValueKindFact::Logical,
        _ => ValueKindFact::Unknown,
    };
    let mut output = ValueFact::proven(kind, shape, StorageFact::Dense);
    if output.kind == ValueKindFact::Unknown {
        output.certainty =
            runmat_types::CertaintyFact::Dynamic(runmat_types::DynamicReason::RuntimeValue);
    }
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub const GRAY2RGB_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "gray2rgb" },
    category: "image/color",
    documentation: documentation::DOCUMENTATION,
    descriptor: &GRAY2RGB_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
        compatibility: BuiltinCompatibility::Matlab,
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
        preparation: BuiltinOperandPreparation::Forms(&[BuiltinPreparationForm {
            id: BuiltinPreparationFormId(0),
            binding_variant: "default",
            signatures: &[0],
            data_arguments: &[0],
            operation: BuiltinPreparationOperation::Sink(BuiltinSinkPreparation::Gather),
            provider_requirement: BuiltinPreparationProviderRequirement::InputSupported,
        }]),
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Host,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &GRAY2RGB_EXTENSIONS,
    integer_capabilities: &GRAY2RGB_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GRAY2RGB_CATALOG_ENTRY];
