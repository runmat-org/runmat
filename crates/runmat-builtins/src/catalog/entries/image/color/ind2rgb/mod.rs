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
    if request.arguments.len() != 2 {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-IND2RGB-ARITY",
            "ind2rgb requires indexed image and colormap",
            0,
        ));
    }
    let indexed = request.arguments.first();
    let map = request.arguments.get(1);
    if let Some(indexed) = indexed {
        match indexed.kind {
            ValueKindFact::Numeric(NumericFact {
                class:
                    NumericClass::Single
                    | NumericClass::Double
                    | NumericClass::UInt8
                    | NumericClass::UInt16,
                domain: NumericDomain::Real,
            })
            | ValueKindFact::Unknown => {}
            _ => diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-IND2RGB-INDEX-CLASS",
                "expected real single, double, uint8, or uint16 indexed image",
                0,
            )),
        }
        if indexed.storage == StorageFact::Sparse {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-IND2RGB-INDEX-STORAGE",
                "indexed image must use dense storage",
                0,
            ));
        }
    }
    let shape = match indexed.map(|fact| &fact.shape) {
        Some(ShapeFact::Scalar) => ShapeFact::from(vec![Some(1), Some(1), Some(3)]),
        Some(ShapeFact::Shaped { dims }) if dims.len() == 2 => ShapeFact::Shaped {
            dims: vec![dims[0].clone(), dims[1].clone(), DimensionFact::Known(3)],
        },
        Some(ShapeFact::Ranked { rank: 2 }) => ShapeFact::from(vec![None, None, Some(3)]),
        Some(ShapeFact::Unknown) | None => ShapeFact::Unknown,
        Some(_) => {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-IND2RGB-INDEX-SHAPE",
                "indexed image must be an MxN matrix",
                0,
            ));
            ShapeFact::Unknown
        }
    };
    if let Some(map) = map {
        match map.kind {
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
            | ValueKindFact::Unknown => {}
            _ => diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-IND2RGB-MAP-CLASS",
                "expected a real double colormap",
                1,
            )),
        }
        match &map.shape {
            ShapeFact::Shaped { dims } if dims.len() == 2 => {
                if matches!(dims[0], DimensionFact::Known(0))
                    || matches!(dims[1], DimensionFact::Known(value) if value != 3)
                {
                    diagnostics.push(crate::catalog::inference::argument_error(
                        "RM-CATALOG-IND2RGB-MAP-SHAPE",
                        "map must be nonempty Nx3",
                        1,
                    ));
                }
            }
            ShapeFact::Ranked { rank: 2 } | ShapeFact::Unknown => {}
            _ => diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-IND2RGB-MAP-SHAPE",
                "map must be nonempty Nx3",
                1,
            )),
        }
        if map.storage == StorageFact::Sparse {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-IND2RGB-MAP-STORAGE",
                "map must use dense storage",
                1,
            ));
        }
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    );
    if !matches!(
        indexed.map(|fact| &fact.residency),
        Some(ResidencyFact::Host)
    ) || !matches!(map.map(|fact| &fact.residency), Some(ResidencyFact::Host))
    {
        output.residency = ResidencyFact::Unknown;
    }
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub const IND2RGB_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "ind2rgb" },
    category: "image/color",
    documentation: documentation::DOCUMENTATION,
    descriptor: &IND2RGB_DESCRIPTOR,
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
            data_arguments: &[0, 1],
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
    integer_capabilities: &IND2RGB_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&IND2RGB_CATALOG_ENTRY];

#[cfg(test)]
mod tests;
