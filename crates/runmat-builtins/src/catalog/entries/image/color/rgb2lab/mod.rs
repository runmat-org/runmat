mod descriptor;
mod documentation;
mod options;

pub use descriptor::*;
pub use options::{Rgb2LabOptionName, RgbColorSpace, RgbWhitePoint};

use crate::*;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, NumericClass,
    NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn valid_shape(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Shaped { dims } if dims.len() == 2 => {
            !matches!(dims[0], DimensionFact::Known(0))
                && !matches!(dims[1], DimensionFact::Known(value) if value != 3)
        }
        ShapeFact::Shaped { dims } if dims.len() >= 3 => {
            !dims[..2]
                .iter()
                .any(|dim| matches!(dim, DimensionFact::Known(0)))
                && !matches!(dims[2], DimensionFact::Known(value) if value != 3)
        }
        ShapeFact::Ranked { rank } if *rank >= 2 => true,
        ShapeFact::Unknown => true,
        _ => false,
    }
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty()
        || request.arguments.len() > 5
        || !(request.arguments.len() - 1).is_multiple_of(2)
    {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-RGB2LAB-ARITY",
            "expected RGB and up to two name-value pairs",
            0,
        ));
    }
    let input = request.arguments.first();
    let shape = input.map_or(ShapeFact::Unknown, |fact| fact.shape.clone());
    if !valid_shape(&shape) {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-RGB2LAB-SHAPE",
            "expected nonempty MxNx3 RGB stack or Nx3 colormap",
            0,
        ));
    }
    let class = match input.map(|fact| &fact.kind) {
        Some(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        })) => {
            if !matches!(
                class,
                NumericClass::Single
                    | NumericClass::Double
                    | NumericClass::UInt8
                    | NumericClass::UInt16
            ) {
                diagnostics.push(crate::catalog::inference::argument_error(
                    "RM-CATALOG-RGB2LAB-CLASS",
                    "expected real single, double, uint8, or uint16 RGB",
                    0,
                ));
            }
            Some(*class)
        }
        Some(ValueKindFact::Unknown) | None => None,
        _ => {
            diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-RGB2LAB-CLASS",
                "expected real numeric RGB",
                0,
            ));
            None
        }
    };
    if input.is_some_and(|fact| fact.storage == StorageFact::Sparse) {
        diagnostics.push(crate::catalog::inference::argument_error(
            "RM-CATALOG-RGB2LAB-STORAGE",
            "RGB must use dense storage",
            0,
        ));
    }
    for index in (1..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            continue;
        };
        match Rgb2LabOptionName::from_name(&name) {
            Some(Rgb2LabOptionName::ColorSpace) => {
                if let Some(value) = request.literals.literal_string_at(index + 1) {
                    if RgbColorSpace::from_name(&value).is_none() {
                        diagnostics.push(crate::catalog::inference::argument_error(
                            "RM-CATALOG-RGB2LAB-COLORSPACE",
                            "unsupported ColorSpace",
                            index + 1,
                        ));
                    }
                }
            }
            Some(Rgb2LabOptionName::WhitePoint) => {
                if let Some(value) = request.literals.literal_string_at(index + 1) {
                    if RgbWhitePoint::from_name(&value).is_none() {
                        diagnostics.push(crate::catalog::inference::argument_error(
                            "RM-CATALOG-RGB2LAB-WHITEPOINT",
                            "unsupported WhitePoint",
                            index + 1,
                        ));
                    }
                }
            }
            None => diagnostics.push(crate::catalog::inference::argument_error(
                "RM-CATALOG-RGB2LAB-OPTION",
                "expected ColorSpace or WhitePoint",
                index,
            )),
        }
    }
    let output_class = class.map(|class| {
        if class == NumericClass::Single {
            NumericClass::Single
        } else {
            NumericClass::Double
        }
    });
    let kind = output_class.map_or(ValueKindFact::Unknown, |class| {
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

pub const RGB2LAB_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "rgb2lab" },
    category: "image/color",
    documentation: documentation::DOCUMENTATION,
    descriptor: &RGB2LAB_DESCRIPTOR,
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
    extensions: &RGB2LAB_EXTENSIONS,
    integer_capabilities: &RGB2LAB_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&RGB2LAB_CATALOG_ENTRY];

#[cfg(test)]
mod tests;
