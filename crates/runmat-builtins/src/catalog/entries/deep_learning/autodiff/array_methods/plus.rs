use super::shared::{array_method_documentation, array_method_entry};
use super::{inference, shared};
use crate::*;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::binary(request, entry, inference::BinaryGeometry::Elementwise)
}

array_method_entry!(DLARRAY_PLUS_CATALOG_ENTRY, "dlarray.plus", DLARRAY_PLUS_DESCRIPTOR,
    array_method_documentation!("dlarray.plus", "Add real floating dlarray values.",
        "Elementwise addition with scalar expansion or equal-shaped host operands. The output retains the dlarray wrapper and is traced inside dlfeval.",
        "A = dlarray(single([1 2 3])); B = A + 2; assert(isequal(B.Data, single([3 4 5])));",
        "assert(strcmp(class(B.Data), 'single')); assert(isequal(size(B.Data), [1 3])); assert(isequal(B.Data, single([3 4 5])));"), infer);
