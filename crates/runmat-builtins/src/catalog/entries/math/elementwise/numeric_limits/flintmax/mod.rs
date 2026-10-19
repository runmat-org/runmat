mod documentation;

use crate::{BuiltinDescriptor, BuiltinSignatureDescriptor, FloatingLimitKind, NumericLimitRule};

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "value = flintmax()",
        inputs: &[],
        outputs: &super::contract::VALUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = flintmax(typename)",
        inputs: &super::contract::CLASS_INPUT,
        outputs: &super::contract::VALUE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "value = flintmax(\"like\", prototype)",
        inputs: &super::contract::LIKE_INPUTS,
        outputs: &super::contract::VALUE_OUTPUT,
    },
];
const DESCRIPTOR: BuiltinDescriptor = super::contract::descriptor(&SIGNATURES);

pub const FLINTMAX_CATALOG_ENTRY: crate::BuiltinCatalogEntry = super::contract::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "flintmax",
    documentation::DOCUMENTATION,
    &DESCRIPTOR,
    NumericLimitRule::Floating(FloatingLimitKind::LargestConsecutiveInteger),
    &[],
);
