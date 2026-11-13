use crate::*;
use runmat_types::{CallInference, CallRequest};

pub const SETVAROPTS_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::shared::modifier_entry(
    "setvaropts",
    &super::contract::SETVAROPTS_DESCRIPTOR,
    super::documentation::SETVAROPTS_DOCUMENTATION,
    super::inference::setvaropts as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    file!(),
    module_path!(),
);
