use crate::*;
use runmat_types::{CallInference, CallRequest};

pub const SETVARTYPE_CATALOG_ENTRY: BuiltinCatalogEntry = super::super::shared::modifier_entry(
    "setvartype",
    &super::contract::SETVARTYPE_DESCRIPTOR,
    super::documentation::SETVARTYPE_DOCUMENTATION,
    super::inference::setvartype as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
    file!(),
    module_path!(),
);
