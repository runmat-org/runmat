//! Static argument rules for `fea.StepObjectRef` constructors.

use crate::fea_contract::model_types::classes;
use crate::fea_contract::model_types::inference::{infer_model_object, ModelArg, ModelObjectCall};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub(super) const CONSTRAINT: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STEP_OBJECT_REF_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn constraint(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CONSTRAINT)
}

pub(super) const BOUNDARY: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STEP_OBJECT_REF_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn boundary(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &BOUNDARY)
}

pub(super) const CONNECTOR: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STEP_OBJECT_REF_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn connector(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &CONNECTOR)
}

pub(super) const INTERFACE: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STEP_OBJECT_REF_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn interface(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &INTERFACE)
}

pub(super) const EXCITATION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STEP_OBJECT_REF_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn excitation(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_model_object(request, entry, &EXCITATION)
}

pub(super) const INITIAL_CONDITION: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STEP_OBJECT_REF_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn initial_condition(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &INITIAL_CONDITION)
}

pub(super) const MATERIAL_PARAMETER: ModelObjectCall = ModelObjectCall {
    result: classes::FEA_STEP_OBJECT_REF_CLASS,
    arguments: &[ModelArg::Text],
};

pub(super) fn material_parameter(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer_model_object(request, entry, &MATERIAL_PARAMETER)
}
