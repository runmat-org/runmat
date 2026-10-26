//! Actual termination enum conversion/comparison methods.
use super::*;
use crate::*;
use runmat_types::*;
mod documentation;
mod termination;
pub use termination::OptimizationExitflag;
fn number(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    crate::catalog::inference::finish_fixed(
        entry,
        request,
        super::super::program_facts::double(ShapeFact::Scalar, StorageFact::Scalar),
        Vec::new(),
    )
}
fn text(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    crate::catalog::inference::finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::String),
        Vec::new(),
    )
}
fn logical(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    crate::catalog::inference::finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Logical),
        Vec::new(),
    )
}
const fn parameter(name: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Original OptimizationExitflag scalar.",
    }
}
const INPUTS: &[BuiltinParamDescriptor] = &[parameter("left"), parameter("right")];
const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric equivalent, member string or logical equality.",
}];
const fn descriptor(signatures: &'static [BuiltinSignatureDescriptor]) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &[OPTIMPROBLEM_ERROR_INVALID_INPUT],
    }
}
const UNARY: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "value = terminationMethod(exitflag)",
    inputs: INPUTS.split_at(1).0,
    outputs: OUTPUTS,
}]);
const BINARY: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "value = terminationEquality(left,right)",
    inputs: INPUTS,
    outputs: OUTPUTS,
}]);
pub const OPTIMEXITFLAG_DOUBLE_CATALOG_ENTRY: BuiltinCatalogEntry = super::entry::entry(
    "optim.problemdef.OptimizationExitflag.double",
    &UNARY,
    documentation::method(
        "OptimizationExitflag.double",
        "optimization-exitflag-double",
    ),
    number,
);
pub const OPTIMEXITFLAG_STRING_CATALOG_ENTRY: BuiltinCatalogEntry = super::entry::entry(
    "optim.problemdef.OptimizationExitflag.string",
    &UNARY,
    documentation::method(
        "OptimizationExitflag.string",
        "optimization-exitflag-string",
    ),
    text,
);
pub const OPTIMEXITFLAG_EQ_CATALOG_ENTRY: BuiltinCatalogEntry = super::entry::entry(
    "optim.problemdef.OptimizationExitflag.eq",
    &BINARY,
    documentation::method("OptimizationExitflag.eq", "optimization-exitflag-eq"),
    logical,
);
pub const OPTIMEXITFLAG_NE_CATALOG_ENTRY: BuiltinCatalogEntry = super::entry::entry(
    "optim.problemdef.OptimizationExitflag.ne",
    &BINARY,
    documentation::method("OptimizationExitflag.ne", "optimization-exitflag-ne"),
    logical,
);
