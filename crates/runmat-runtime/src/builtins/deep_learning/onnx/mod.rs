use runmat_builtins::{ExportOnnxOptionKey, ExportOnnxOptionValueRole};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericStorage, ObjectInstance, Tensor, Value};
use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::BuiltinResult;

use super::layer_identity::ForwardLayerKind;
use super::{
    deep_learning_error, layer_names, logical_scalar, model, numeric_scalar, parse_name_values,
    positive_i64, scalar_text, unsupported_error, DLARRAY_CLASS, FEATURE_INPUT_LAYER_CLASS,
};

const EXPORT_ONNX: &str = "exportONNXNetwork";
const DEFAULT_OPSET_VERSION: i64 = 14;
const MIN_SUPPORTED_OPSET_VERSION: i64 = 13;
const MAX_SUPPORTED_OPSET_VERSION: i64 = 20;
const ONNX_FLOAT: i32 = 1;
const ONNX_DOUBLE: i32 = 11;

mod admission;
mod encode;
mod graph;
mod options;
mod types;

use admission::admit_export_call;
use encode::encode_model;
#[cfg(test)]
use encode::{encode_initializer, put_varint};
use graph::build_export_graph;
#[cfg(test)]
use graph::{merge_export_dtype, tensor_to_row_major_2d};
#[cfg(test)]
pub(super) use options::integer_scalar;
use options::{parse_export_options, parse_option_inputs, unsupported_option, validate_option};
use types::*;

#[runtime_builtin(
    name = "exportONNXNetwork",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::onnx"
)]
pub(super) async fn export_onnx_network_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 0) {
        return Err(deep_learning_error(
            EXPORT_ONNX,
            "exportONNXNetwork does not return output arguments",
        ));
    }

    let admitted = admit_export_call(args).await?;
    let graph = build_export_graph(&admitted.network, &admitted.layers, &admitted.options)?;
    let bytes = encode_model(&graph, admitted.options.opset_version);
    runmat_filesystem::write_async(PathBuf::from(&admitted.filename), &bytes)
        .await
        .map_err(|err| {
            crate::build_runtime_error(format!(
                "exportONNXNetwork: failed to write '{}': {err}",
                admitted.filename
            ))
            .with_builtin(EXPORT_ONNX)
            .with_identifier("RunMat:deepLearning:InvalidInput")
            .with_source(err)
            .build()
        })?;
    Ok(Value::output_list(Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_length_delimited_field(bytes: &[u8], field: u32, payload: &[u8]) -> bool {
        let key = ((field as u64) << 3) | 2;
        let mut expected = Vec::new();
        put_varint(&mut expected, key);
        put_varint(&mut expected, payload.len() as u64);
        expected.extend_from_slice(payload);
        bytes
            .windows(expected.len())
            .any(|window| window == expected.as_slice())
    }

    fn has_varint_field(bytes: &[u8], field: u32, value: u64) -> bool {
        let mut expected = Vec::new();
        put_varint(&mut expected, (field as u64) << 3);
        put_varint(&mut expected, value);
        bytes
            .windows(expected.len())
            .any(|window| window == expected.as_slice())
    }

    #[test]
    fn initializer_name_uses_tensor_proto_name_field() {
        let initializer = Initializer {
            name: "fc.Weights".into(),
            dims: vec![2, 3],
            data: NumericStorage::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let encoded = encode_initializer(&initializer, NumericDType::F64);

        assert!(has_length_delimited_field(&encoded, 8, b"fc.Weights"));
        assert!(!has_length_delimited_field(&encoded, 4, b"fc.Weights"));
    }

    #[test]
    fn initializer_encodes_native_single_type_and_raw_width() {
        let initializer = Initializer {
            name: "fc.Bias".into(),
            dims: vec![2],
            data: NumericStorage::F32(vec![1.25, -2.5]),
        };
        let encoded = encode_initializer(&initializer, NumericDType::F32);
        let raw = [1.25_f32.to_le_bytes(), (-2.5_f32).to_le_bytes()].concat();

        assert!(has_varint_field(&encoded, 2, ONNX_FLOAT as u64));
        assert!(has_length_delimited_field(&encoded, 9, &raw));
    }

    #[test]
    fn mixed_floating_initializers_promote_graph_to_double() {
        assert_eq!(
            merge_export_dtype(Some(NumericDType::F32), NumericDType::F64),
            NumericDType::F64
        );
        assert_eq!(
            merge_export_dtype(Some(NumericDType::F64), NumericDType::F32),
            NumericDType::F64
        );
        assert_eq!(
            merge_export_dtype(Some(NumericDType::F32), NumericDType::F32),
            NumericDType::F32
        );
    }

    #[test]
    fn row_major_conversion_preserves_native_single_storage() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let storage = tensor_to_row_major_2d(tensor).unwrap();

        assert_eq!(
            storage,
            NumericStorage::F32(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
        );
    }
}
