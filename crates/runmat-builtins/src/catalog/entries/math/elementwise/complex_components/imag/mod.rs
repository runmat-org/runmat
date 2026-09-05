mod documentation;

use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, NumericComponentRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use super::projection_contract::define_component_projection;
use documentation::DOCUMENTATION;

const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

define_component_projection!(
    "imag",
    "IMAG",
    NumericComponentRule::ImaginaryPart,
    DOCUMENTATION,
    "Imaginary component of X.",
    "All eight real and componentwise-complex integer classes retain their class.",
    "Real integer input produces exact same-class zeros. Paired complex-integer input projects its exact imaginary storage without arithmetic or overflow; supported resident forms preserve class, shape, owner, and residency."
);

pub use CATALOG_ENTRY as IMAG_CATALOG_ENTRY;
pub use DESCRIPTOR as IMAG_DESCRIPTOR;
pub use ERROR_INTERNAL as IMAG_ERROR_INTERNAL;
pub use ERROR_INVALID_INPUT as IMAG_ERROR_INVALID_INPUT;
pub use INTEGER_CAPABILITIES as IMAG_INTEGER_CAPABILITIES;
