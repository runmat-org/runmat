mod descriptor;
mod documentation;

pub use descriptor::*;

use crate::*;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-HSV2RGB-ARITY",
            "hsv2rgb requires one input",
            0,
        ));
    }
    let input = request.arguments.first();
    let shape = input.map_or(ShapeFact::Unknown, |fact| fact.shape.clone());
    let layout = match &shape {
        ShapeFact::Shaped { dims } if dims.len() == 3 => {
            if dims[..2]
                .iter()
                .any(|dim| matches!(dim, DimensionFact::Known(0)))
                || matches!(dims[2], DimensionFact::Known(value) if value != 3)
            {
                Some(false)
            } else {
                Some(true)
            }
        }
        ShapeFact::Shaped { dims } if dims.len() == 2 => {
            if matches!(dims[0], DimensionFact::Known(0))
                || matches!(dims[1], DimensionFact::Known(value) if value != 3)
            {
                Some(false)
            } else {
                Some(true)
            }
        }
        ShapeFact::Ranked { rank: 2 | 3 } | ShapeFact::Unknown => None,
        _ => Some(false),
    };
    if layout == Some(false) {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-HSV2RGB-SHAPE",
            "expected nonempty MxNx3 HSV image or Nx3 colormap",
            0,
        ));
    }
    let kind = match input.map(|fact| &fact.kind) {
        Some(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        })) => {
            if !matches!(class, NumericClass::Single | NumericClass::Double) {
                diagnostics.push(crate::catalog::inference::argument_error(
                    "RM-CATALOG-HSV2RGB-CLASS",
                    "expected single, double, or logical HSV",
                    0,
                ));
            }
            ValueKindFact::Numeric(NumericFact {
                class: *class,
                domain: NumericDomain::Real,
            })
        }
        Some(ValueKindFact::Logical) => {
            if matches!(&shape, ShapeFact::Shaped { dims } if dims.len() == 2)
                || matches!(&shape, ShapeFact::Ranked { rank: 2 })
            {
                diagnostics.push(crate::catalog::inference::argument_error(
                    "RM-CATALOG-HSV2RGB-CLASS",
                    "logical colormap is unsupported",
                    0,
                ));
            }
            ValueKindFact::Logical
        }
        Some(ValueKindFact::Unknown) | None => ValueKindFact::Unknown,
        _ => {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-HSV2RGB-CLASS",
                "expected real numeric or logical HSV",
                0,
            ));
            ValueKindFact::Unknown
        }
    };
    if input.is_some_and(|fact| fact.storage == StorageFact::Sparse) {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-HSV2RGB-STORAGE",
            "HSV must use dense storage",
            0,
        ));
    }
    let output_kind = match kind {
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        }) => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => ValueKindFact::Unknown,
    };
    let mut output = ValueFact::proven(output_kind, shape, StorageFact::Dense);
    if output.kind == ValueKindFact::Unknown {
        output.certainty =
            runmat_types::CertaintyFact::Dynamic(runmat_types::DynamicReason::RuntimeValue);
    }
    if !matches!(input.map(|fact| &fact.residency), Some(ResidencyFact::Host)) {
        output.residency = ResidencyFact::Unknown;
    }
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub const HSV2RGB_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "hsv2rgb" },
    category: "image/color",
    documentation: documentation::DOCUMENTATION,
    descriptor: &HSV2RGB_DESCRIPTOR,
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
        residency: BuiltinResidencyPolicy::Dynamic,
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
    extensions: &[],
    integer_capabilities: &HSV2RGB_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&HSV2RGB_CATALOG_ENTRY];

#[cfg(test)]
mod tests;
