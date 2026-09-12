use crate::analysis::load_fea_document_from_path_async;
use crate::analysis::AnalysisStudySpec;
use crate::analysis::AnalysisStudySweepSpec;
use crate::analysis::FeaResolvedDocument;
use crate::builtins::fea::contracts::descriptors::{ERROR_INPUT, ERROR_LOAD};
use crate::builtins::fea::contracts::identities::{
    FEA_STUDY_CLASS, FEA_STUDY_SPEC_JSON_PROPERTY, FEA_SWEEP_CLASS, FEA_SWEEP_SPEC_JSON_PROPERTY,
    LOAD_NAME,
};
use crate::builtins::fea::errors::builtin_error;
use crate::builtins::fea::geometry::object_json_property;
use crate::builtins::fea::output::resolved_document_to_object;
use crate::BuiltinResult;
use runmat_value::Value;
use std::path::PathBuf;

pub(in crate::builtins::fea) async fn load_document_object(path: PathBuf) -> BuiltinResult<Value> {
    let document = load_fea_document_from_path_async(&path)
        .await
        .map_err(|err| builtin_error(LOAD_NAME, &ERROR_LOAD, err))?;
    resolved_document_to_object(document)
}

pub(in crate::builtins::fea) async fn resolve_document_input(
    input: Value,
    builtin: &'static str,
) -> BuiltinResult<FeaResolvedDocument> {
    match input {
        Value::Object(object) if object.class_name.is(FEA_STUDY_CLASS) => {
            let spec: AnalysisStudySpec =
                object_json_property(builtin, &object, FEA_STUDY_SPEC_JSON_PROPERTY, &ERROR_INPUT)?;
            Ok(FeaResolvedDocument::Study(Box::new(spec)))
        }
        Value::Object(object) if object.class_name.is(FEA_SWEEP_CLASS) => {
            let spec: AnalysisStudySweepSpec =
                object_json_property(builtin, &object, FEA_SWEEP_SPEC_JSON_PROPERTY, &ERROR_INPUT)?;
            Ok(FeaResolvedDocument::Sweep(spec))
        }
        Value::String(path) => load_fea_document_from_path_async(&PathBuf::from(path))
            .await
            .map_err(|err| builtin_error(builtin, &ERROR_LOAD, err)),
        Value::CharArray(chars) if chars.rows == 1 => {
            let path: String = chars.data.iter().collect();
            load_fea_document_from_path_async(&PathBuf::from(path))
                .await
                .map_err(|err| builtin_error(builtin, &ERROR_LOAD, err))
        }
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected .fea path, {FEA_STUDY_CLASS}, or {FEA_SWEEP_CLASS}; got {other:?}"),
        )),
    }
}
