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
    "real",
    "REAL",
    NumericComponentRule::RealPart,
    DOCUMENTATION,
    "Real component of X.",
    "All eight integer classes retain their class while projecting the real component.",
    "Real integer input is an exact same-class identity. Paired complex-integer input projects its same-class real storage without arithmetic; supported resident forms preserve class, shape, owner, and residency."
);

pub use CATALOG_ENTRY as REAL_CATALOG_ENTRY;
pub use DESCRIPTOR as REAL_DESCRIPTOR;
pub use ERROR_INTERNAL as REAL_ERROR_INTERNAL;
pub use ERROR_INVALID_INPUT as REAL_ERROR_INVALID_INPUT;
pub use INTEGER_CAPABILITIES as REAL_INTEGER_CAPABILITIES;
