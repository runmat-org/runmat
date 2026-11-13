use super::shared::{array_method_documentation, array_method_entry};
use super::{inference, shared};
use crate::*;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::binary(request, entry, inference::BinaryGeometry::Elementwise)
}

array_method_entry!(DLARRAY_RDIVIDE_CATALOG_ENTRY, "dlarray.rdivide", DLARRAY_RDIVIDE_DESCRIPTOR,
    array_method_documentation!("dlarray.rdivide", "Divide real floating dlarray values elementwise.",
        "Elementwise right division with scalar expansion or equal-shaped host operands. Division by zero follows floating-point arithmetic and may produce Inf or NaN.",
        "A = dlarray(single([2 4 8])); B = A ./ 2; assert(isequal(B.Data, single([1 2 4])));",
        "assert(strcmp(class(B.Data), 'single')); assert(isequal(size(B.Data), [1 3])); assert(isequal(B.Data, single([1 2 4])));"), infer);
