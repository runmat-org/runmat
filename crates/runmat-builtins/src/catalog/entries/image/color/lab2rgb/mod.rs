mod descriptor;
mod documentation;
mod output_type;

pub use descriptor::*;
pub use output_type::LabOutputType;

use crate::*;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 1 | 3) {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-LAB2RGB-ARITY",
            "expected LAB and optional OutputType name-value pair",
            0,
        ));
    }
    let input = request.arguments.first();
    let shape = input.map_or(ShapeFact::Unknown, |fact| fact.shape.clone());
    let valid_shape = match &shape {
        ShapeFact::Shaped { dims } if dims.len() == 3 => {
            !dims[..2]
                .iter()
                .any(|dim| matches!(dim, DimensionFact::Known(0)))
                && !matches!(dims[2], DimensionFact::Known(value) if value != 3)
        }
        ShapeFact::Shaped { dims } if dims.len() == 2 => {
            !matches!(dims[0], DimensionFact::Known(0))
                && !matches!(dims[1], DimensionFact::Known(value) if value != 3)
        }
        ShapeFact::Ranked { rank: 2 | 3 } | ShapeFact::Unknown => true,
        _ => false,
    };
    if !valid_shape {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-LAB2RGB-SHAPE",
            "expected nonempty MxNx3 Lab image or Nx3 colormap",
            0,
        ));
    }
    let input_class = match input.map(|fact| &fact.kind) {
        Some(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single | NumericClass::Double,
            domain: NumericDomain::Real,
        })) => input.and_then(|fact| match fact.kind {
            ValueKindFact::Numeric(numeric) => Some(numeric.class),
            _ => None,
        }),
        Some(ValueKindFact::Unknown) | None => None,
        _ => {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-LAB2RGB-CLASS",
                "expected real single or double Lab",
                0,
            ));
            None
        }
    };
    if input.is_some_and(|fact| fact.storage == StorageFact::Sparse) {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-LAB2RGB-STORAGE",
            "Lab must use dense storage",
            0,
        ));
    }
    let option_class = if request.arguments.len() == 3 {
        let name = request.literals.literal_string_at(1);
        if name
            .as_deref()
            .is_some_and(|name| !name.eq_ignore_ascii_case("OutputType"))
        {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-LAB2RGB-OPTION",
                "expected OutputType option name",
                1,
            ));
        }
        match request.literals.literal_string_at(2) {
            Some(value) => match LabOutputType::from_name(&value) {
                Some(output) => Some(output.class()),
                None => {
                    diagnostics.push(crate::catalog::inference::argument_error(
                        "RM-CATALOG-LAB2RGB-OUTPUT-TYPE",
                        "unsupported OutputType",
                        2,
                    ));
                    None
                }
            },
            None => None,
        }
    } else {
        input_class
    };
    let kind = option_class.map_or(ValueKindFact::Unknown, |class| {
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        })
    });
    let mut output = ValueFact::proven(kind, shape, StorageFact::Dense);
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

pub const LAB2RGB_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "lab2rgb" },
    category: "image/color",
    documentation: documentation::DOCUMENTATION,
    descriptor: &LAB2RGB_DESCRIPTOR,
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
    extensions: &LAB2RGB_EXTENSIONS,
    integer_capabilities: &LAB2RGB_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&LAB2RGB_CATALOG_ENTRY];

#[cfg(test)]
mod tests;
