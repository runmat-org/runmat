mod documentation;

use crate::{
    BuiltinDescriptor, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, FloatingLimitKind, NumericLimitRule,
};

const VALUE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single- or double-precision real or complex input array.",
}];
const EPS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "spacing",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real spacing scalar or same-shaped floating array, according to the input form.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "d = eps()",
        inputs: &[],
        outputs: &EPS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "d = eps(x)",
        inputs: &VALUE_INPUT,
        outputs: &EPS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "d = eps(typename)",
        inputs: &super::contract::CLASS_INPUT,
        outputs: &EPS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "d = eps(\"like\", prototype)",
        inputs: &super::contract::LIKE_INPUTS,
        outputs: &EPS_OUTPUT,
    },
];
const DESCRIPTOR: BuiltinDescriptor = super::contract::descriptor(&SIGNATURES);

const EFFECTS: &[runmat_types::EffectKind] = &[
    runmat_types::EffectKind::MaySuspend,
    runmat_types::EffectKind::MayThrow,
];

pub const EPS_CATALOG_ENTRY: crate::BuiltinCatalogEntry = {
    let mut entry = super::contract::entry(
        crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        "eps",
        documentation::DOCUMENTATION,
        &DESCRIPTOR,
        NumericLimitRule::Floating(FloatingLimitKind::SpacingAtOne),
        &[],
    );
    entry.contract.async_behavior = crate::BuiltinAsyncBehavior::MaySuspend;
    entry.contract.effects = EFFECTS;
    // Duration is a documented MATLAB input but is not yet represented by a
    // first-class RunMat Value carrier. Keep the public contract partial.
    entry.contract.maturity = crate::BuiltinContractMaturity::Incomplete;
    entry.placement.distributed = crate::BuiltinDistributedPolicy::MapUnaryOrScalarLikePrototype;
    entry
};
