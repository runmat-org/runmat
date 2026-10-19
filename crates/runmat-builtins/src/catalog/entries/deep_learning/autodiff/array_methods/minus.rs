use super::shared::{array_method_documentation, array_method_entry};
use super::{inference, shared};
use crate::*;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::binary(request, entry, inference::BinaryGeometry::Elementwise)
}

array_method_entry!(DLARRAY_MINUS_CATALOG_ENTRY, "dlarray.minus", DLARRAY_MINUS_DESCRIPTOR,
    array_method_documentation!("dlarray.minus", "Subtract real floating dlarray values.",
        "Elementwise subtraction with scalar expansion or equal-shaped host operands. Operand order determines the value but not the floating-class rule.",
        "A = dlarray(single([3 4 5])); B = 6 - A; assert(isequal(B.Data, single([3 2 1])));",
        "assert(strcmp(class(B.Data), 'single')); assert(isequal(size(B.Data), [1 3])); assert(isequal(B.Data, single([3 2 1])));"), infer);
