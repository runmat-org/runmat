use crate::*;
use runmat_types::{CallInference, CallRequest};

pub const GETVAROPTS_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::shared::modifier_entry(
    "getvaropts",
    &super::contract::GETVAROPTS_DESCRIPTOR,
    super::documentation::GETVAROPTS_DOCUMENTATION,
    super::inference::getvaropts as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    file!(),
    module_path!(),
);
