mod documentation;

use crate::{BuiltinDescriptor, BuiltinSignatureDescriptor, FloatingLimitKind, NumericLimitRule};

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "value = realmin()",
        inputs: &[],
        outputs: &super::contract::VALUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = realmin(typename)",
        inputs: &super::contract::CLASS_INPUT,
        outputs: &super::contract::VALUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = realmin(\"like\", prototype)",
        inputs: &super::contract::LIKE_INPUTS,
        outputs: &super::contract::VALUE_OUTPUT,
    },
];
const DESCRIPTOR: BuiltinDescriptor = super::contract::descriptor(&SIGNATURES);

pub const REALMIN_CATALOG_ENTRY: crate::BuiltinCatalogEntry = super::contract::entry(
    "realmin",
    documentation::DOCUMENTATION,
    &DESCRIPTOR,
    NumericLimitRule::Floating(FloatingLimitKind::SmallestNormal),
    &[],
);
