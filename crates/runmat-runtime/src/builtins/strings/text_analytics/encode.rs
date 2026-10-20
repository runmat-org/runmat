//! Count-matrix encoding for Text Analytics bag models.

use std::collections::{BTreeMap, HashMap, HashSet};

use runmat_builtins::{
    ENCODE_ERROR_INVALID_INPUT, ENCODE_NUMERIC_FORCE_CELL_OUTPUT_EXTENSION,
    ENCODE_RESIDENT_FORCE_CELL_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, ObjectInstance, SparseTensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ReductionNaN, ResidencyPolicy,
};
use crate::builtins::strings::common::{scalar_text_input, text_input_admission};
use crate::builtins::strings::text_analytics::documents::{
    documents_from_object, vocabulary_from_bag, words_from_word_vector, BAG_OF_WORDS_CLASS,
    TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::ngrams::{ngrams_from_bag, BAG_OF_NGRAMS_CLASS};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::strings::text_analytics::encode"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "encode",
    op_kind: GpuOpKind::Custom("text-analytics-encode"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The builtin owns resident arguments so object/text rejection and ForceCellOutput compatibility gates run before provider access; admitted scalar controls gather explicitly and count outputs remain host sparse values.",
};

fn encode_error(message: impl Into<String>) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("encode");
    if let Some(identifier) = ENCODE_ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "encode",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::text_analytics::encode",
    context = true
)]
async fn encode_builtin(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let (bag, input, options) = parse_args(args, runtime.as_ref()).await?;
    let sparse = match bag {
        Value::Object(object) if object.is_class(BAG_OF_WORDS_CLASS) => {
            encode_words(&object, input, options.documents_in)?
        }
        Value::Object(object) if object.is_class(BAG_OF_NGRAMS_CLASS) => {
            encode_ngrams(&object, input, options.documents_in)?
        }
        Value::Object(object) => {
            return Err(encode_error(format!(
                "encode: expected bagOfWords or bagOfNgrams object, got {}",
                object.class_name()
            )))
        }
        other => {
            return Err(encode_error(format!(
                "encode: expected bagOfWords or bagOfNgrams object, got {other:?}"
            )))
        }
    };

    let output = Value::SparseTensor(sparse);
    if options.force_cell_output {
        return CellArray::new(vec![output], 1, 1)
            .map(Value::Cell)
            .map_err(encode_error);
    }
    Ok(output)
}

#[derive(Clone, Copy)]
enum DocumentsIn {
    Rows,
    Columns,
}

struct EncodeOptions {
    documents_in: DocumentsIn,
    force_cell_output: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            documents_in: DocumentsIn::Rows,
            force_cell_output: false,
        }
    }
}

async fn parse_args(
    mut args: crate::call::BuiltinArguments,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<(Value, Value, EncodeOptions)> {
    if args.len() < 2 {
        return Err(encode_error(
            "encode: expected bag model and documents or words input",
        ));
    }
    if !(args.len() - 2).is_multiple_of(2) {
        return Err(encode_error(
            "encode: name-value options must appear in pairs",
        ));
    }
    let bag = args.pop_front().expect("validated argument prefix");
    let input = args.pop_front().expect("validated argument prefix");
    if crate::dispatcher::value_contains_gpu(&bag) {
        return Err(encode_error(
            "encode: bag model must be a host bagOfWords or bagOfNgrams object",
        ));
    }
    if crate::dispatcher::value_contains_gpu(&input) {
        return Err(encode_error(
            "encode: documents or words must be host text or tokenizedDocument values",
        ));
    }
    match &bag {
        Value::Object(object)
            if object.is_class(BAG_OF_WORDS_CLASS) || object.is_class(BAG_OF_NGRAMS_CLASS) => {}
        Value::Object(object) => {
            return Err(encode_error(format!(
                "encode: expected bagOfWords or bagOfNgrams object, got {}",
                object.class_name()
            )))
        }
        other => {
            return Err(encode_error(format!(
                "encode: expected bagOfWords or bagOfNgrams object, got {other:?}"
            )))
        }
    }
    validate_documents_outer_type(&input)?;
    let mut options = EncodeOptions::default();
    let mut idx = 0;
    while idx < args.len() {
        let name = scalar_text_input(&args[idx], runtime, |runtime, cause| {
            text_input_admission("encode", runtime, cause)
        })?
        .ok_or_else(|| encode_error("encode: option name must be scalar text"))?;
        if name.as_str().eq_ignore_ascii_case("documentsin") {
            let value = scalar_text_input(&args[idx + 1], runtime, |runtime, cause| {
                text_input_admission("encode", runtime, cause)
            })?
            .ok_or_else(|| encode_error("encode: DocumentsIn must be scalar text"))?;
            options.documents_in = if value.as_str().eq_ignore_ascii_case("rows") {
                DocumentsIn::Rows
            } else if value.as_str().eq_ignore_ascii_case("columns") {
                DocumentsIn::Columns
            } else {
                return Err(encode_error(format!(
                    "encode: DocumentsIn must be 'rows' or 'columns', got '{}'",
                    value.as_str()
                )));
            };
        } else if name.as_str().eq_ignore_ascii_case("forcecelloutput") {
            let raw = &args[idx + 1];
            let resident = crate::dispatcher::value_contains_gpu(raw);
            if resident {
                validate_scalar_control_shape(raw)?;
                crate::compatibility::ensure_builtin_extension_enabled(
                    &ENCODE_RESIDENT_FORCE_CELL_OUTPUT_EXTENSION,
                    "encode",
                )?;
            }
            let host = if resident {
                gather_if_needed_async(raw).await.map_err(|err| {
                    encode_error(format!("encode: failed to gather ForceCellOutput: {err}"))
                })?
            } else {
                raw.clone()
            };
            let parsed = parse_bool_scalar(&host)?;
            if is_numeric_bool_value(&host) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &ENCODE_NUMERIC_FORCE_CELL_OUTPUT_EXTENSION,
                    "encode",
                )?;
            }
            options.force_cell_output = parsed;
        } else {
            return Err(encode_error(format!(
                "encode: unsupported option '{}'",
                name.as_str()
            )));
        }
        idx += 2;
    }
    Ok((bag, input, options))
}

fn validate_documents_outer_type(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => Ok(()),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => Ok(()),
        other => Err(encode_error(format!(
            "encode: expected tokenizedDocument or word vector, got {other:?}"
        ))),
    }
}

fn validate_scalar_control_shape(value: &Value) -> BuiltinResult<()> {
    let len = match value {
        Value::GpuTensor(handle) => handle
            .shape()
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .unwrap_or(usize::MAX),
        Value::Tensor(tensor) => tensor.len(),
        Value::LogicalArray(array) => array.data.len(),
        _ => 1,
    };
    if len != 1 {
        return Err(encode_error(
            "encode: ForceCellOutput must be a logical scalar",
        ));
    }
    Ok(())
}

fn is_numeric_bool_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
}

fn parse_bool_scalar(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Int(value) => Ok(!value.is_zero()),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return Ok(!value.is_zero());
            }
            let value = crate::builtins::common::tensor::tensor_value_f64(tensor, 0);
            if value == 0.0 || value == 1.0 {
                Ok(value != 0.0)
            } else {
                Err(encode_error(
                    "encode: ForceCellOutput numeric values must be 0 or 1",
                ))
            }
        }
        other => Err(encode_error(format!(
            "encode: ForceCellOutput must be a logical scalar, got {other:?}"
        ))),
    }
}

fn encode_words(
    object: &ObjectInstance,
    input: Value,
    documents_in: DocumentsIn,
) -> BuiltinResult<SparseTensor> {
    let vocabulary = vocabulary_from_bag(object, "encode").map_err(|err| {
        encode_error(format!(
            "encode: failed to read bagOfWords Vocabulary property: {err}"
        ))
    })?;
    let documents = documents_from_input(input, "bagOfWords")?;
    let positions = vocabulary
        .iter()
        .enumerate()
        .map(|(idx, word)| (word.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let counts = documents
        .iter()
        .map(|document| {
            let mut row = BTreeMap::new();
            for token in document {
                if let Some(&col) = positions.get(token.as_str()) {
                    *row.entry(col).or_insert(0.0) += 1.0;
                }
            }
            row
        })
        .collect::<Vec<_>>();
    sparse_from_document_counts(counts, vocabulary.len(), documents_in)
}

fn encode_ngrams(
    object: &ObjectInstance,
    input: Value,
    documents_in: DocumentsIn,
) -> BuiltinResult<SparseTensor> {
    let ngrams = ngrams_from_bag(object, "encode").map_err(|err| {
        encode_error(format!(
            "encode: failed to read bagOfNgrams Ngrams property: {err}"
        ))
    })?;
    let lengths = unique_ngram_lengths(&ngrams);
    let documents = documents_from_input(input, "bagOfNgrams")?;
    let positions = ngrams
        .iter()
        .enumerate()
        .map(|(idx, ngram)| (ngram.as_slice(), idx))
        .collect::<HashMap<_, _>>();
    let counts = documents
        .iter()
        .map(|document| {
            let mut row = BTreeMap::new();
            for &length in &lengths {
                if length > document.len() {
                    continue;
                }
                for start in 0..=document.len() - length {
                    let key = &document[start..start + length];
                    if let Some(&col) = positions.get(key) {
                        *row.entry(col).or_insert(0.0) += 1.0;
                    }
                }
            }
            row
        })
        .collect::<Vec<_>>();
    sparse_from_document_counts(counts, ngrams.len(), documents_in)
}

fn unique_ngram_lengths(ngrams: &[Vec<String>]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut lengths = Vec::new();
    for ngram in ngrams {
        let length = ngram.len();
        if seen.insert(length) {
            lengths.push(length);
        }
    }
    lengths
}

fn documents_from_input(input: Value, model_name: &str) -> BuiltinResult<Vec<Vec<String>>> {
    match input {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => {
            documents_from_object(&object, "encode").map_err(|err| {
                encode_error(format!(
                    "encode: failed to read tokenizedDocument input: {err}"
                ))
            })
        }
        Value::Object(object) => Err(encode_error(format!(
            "encode: expected tokenizedDocument or word vector for {model_name}, got {}",
            object.class_name()
        ))),
        other => {
            validate_row_word_vector(&other, model_name)?;
            Ok(vec![words_from_word_vector(&other, "encode").map_err(
                |err| encode_error(format!("encode: failed to read word vector input: {err}")),
            )?])
        }
    }
}

fn validate_row_word_vector(value: &Value, model_name: &str) -> BuiltinResult<()> {
    match value {
        Value::String(_) => Ok(()),
        Value::StringArray(array) if array.rows <= 1 => Ok(()),
        Value::CharArray(array) if array.rows <= 1 => Ok(()),
        Value::Cell(cell) if cell.rows <= 1 => Ok(()),
        Value::StringArray(array) => Err(encode_error(format!(
            "encode: non-tokenized {model_name} input must be a row word vector; got string array with shape {}x{}",
            array.rows, array.cols
        ))),
        Value::CharArray(array) => Err(encode_error(format!(
            "encode: non-tokenized {model_name} input must be a row word vector; got char array with shape {}x{}",
            array.rows, array.cols
        ))),
        Value::Cell(cell) => Err(encode_error(format!(
            "encode: non-tokenized {model_name} input must be a row word vector; got cell array with shape {}x{}",
            cell.rows, cell.cols
        ))),
        other => Err(encode_error(format!(
            "encode: expected tokenizedDocument or word vector for {model_name}, got {other:?}"
        ))),
    }
}

fn sparse_from_document_counts(
    counts: Vec<BTreeMap<usize, f64>>,
    term_count: usize,
    documents_in: DocumentsIn,
) -> BuiltinResult<SparseTensor> {
    match documents_in {
        DocumentsIn::Rows => sparse_rows(counts, term_count),
        DocumentsIn::Columns => sparse_columns(counts, term_count),
    }
}

fn sparse_rows(
    counts: Vec<BTreeMap<usize, f64>>,
    term_count: usize,
) -> BuiltinResult<SparseTensor> {
    let rows = counts.len();
    let cols = term_count;
    let col_ptr_capacity = cols
        .checked_add(1)
        .ok_or_else(|| encode_error("encode: sparse output column count overflows"))?;
    let mut columns = vec![Vec::<(usize, f64)>::new(); cols];
    for (doc_idx, doc_counts) in counts.iter().enumerate() {
        for (&term_idx, &value) in doc_counts {
            if term_idx >= term_count {
                return Err(encode_error(
                    "encode: internal sparse term index exceeds model size",
                ));
            }
            if value != 0.0 {
                columns[term_idx].push((doc_idx, value));
            }
        }
    }
    let mut col_ptrs = Vec::with_capacity(col_ptr_capacity);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for entries in columns {
        for (row, value) in entries {
            row_indices.push(row);
            values.push(value);
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new(rows, cols, col_ptrs, row_indices, values).map_err(encode_error)
}

fn sparse_columns(
    counts: Vec<BTreeMap<usize, f64>>,
    term_count: usize,
) -> BuiltinResult<SparseTensor> {
    let rows = term_count;
    let cols = counts.len();
    let col_ptr_capacity = cols
        .checked_add(1)
        .ok_or_else(|| encode_error("encode: sparse output column count overflows"))?;
    let mut col_ptrs = Vec::with_capacity(col_ptr_capacity);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for doc_counts in &counts {
        for (&row, &value) in doc_counts {
            if row >= term_count {
                return Err(encode_error(
                    "encode: internal sparse term index exceeds model size",
                ));
            }
            if value != 0.0 {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new(rows, cols, col_ptrs, row_indices, values).map_err(encode_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{IntValue, IntegerStorage, StringArray, Tensor};

    fn run_encode(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(encode_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn sparse(value: Value) -> SparseTensor {
        match value {
            Value::SparseTensor(sparse) => sparse,
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    fn string_array(values: &[&str], rows: usize, cols: usize) -> Value {
        Value::StringArray(
            StringArray::new(
                values.iter().map(|value| (*value).to_string()).collect(),
                vec![rows, cols],
            )
            .expect("string array"),
        )
    }

    fn tokenized(docs: &[&[&str]]) -> Value {
        let mut data = Vec::with_capacity(docs.len());
        for doc in docs {
            let row = doc
                .iter()
                .map(|token| Value::from(*token))
                .collect::<Vec<_>>();
            data.push(Value::Cell(
                CellArray::new(row, 1, doc.len()).expect("row cell"),
            ));
        }
        let mut object = ObjectInstance::new(TOKENIZED_DOCUMENT_CLASS.to_string());
        object.replace_property_unbounded(
            "Documents".to_string(),
            Value::Cell(CellArray::new(data, docs.len(), 1).expect("documents cell")),
        );
        object
            .replace_property_unbounded("NumDocuments".to_string(), Value::Num(docs.len() as f64));
        Value::Object(object)
    }

    fn bag_of_words(vocabulary: &[&str]) -> Value {
        let mut object = ObjectInstance::new(BAG_OF_WORDS_CLASS.to_string());
        object.replace_property_unbounded(
            "Vocabulary".to_string(),
            string_array(vocabulary, 1, vocabulary.len()),
        );
        object.replace_property_unbounded(
            "Counts".to_string(),
            Value::Tensor(Tensor::zeros(vec![0, vocabulary.len()])),
        );
        object.replace_property_unbounded(
            "NumWords".to_string(),
            Value::Num(vocabulary.len() as f64),
        );
        object.replace_property_unbounded("NumDocuments".to_string(), Value::Num(0.0));
        Value::Object(object)
    }

    fn bag_of_ngrams(ngrams: &[&[&str]], lengths: &[usize]) -> Value {
        let rows = ngrams.len();
        let cols = ngrams.iter().map(|ngram| ngram.len()).max().unwrap_or(0);
        let mut data = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            for ngram in ngrams {
                data.push(ngram.get(col).copied().unwrap_or_default().to_string());
            }
        }
        let mut object = ObjectInstance::new(BAG_OF_NGRAMS_CLASS.to_string());
        object.replace_property_unbounded(
            "Ngrams".to_string(),
            Value::StringArray(StringArray::new(data, vec![rows, cols]).expect("ngrams")),
        );
        object.replace_property_unbounded(
            "NgramLengths".to_string(),
            Value::Tensor(
                Tensor::new(
                    lengths.iter().map(|length| *length as f64).collect(),
                    vec![1, lengths.len()],
                )
                .expect("lengths"),
            ),
        );
        object.replace_property_unbounded(
            "Counts".to_string(),
            Value::Tensor(Tensor::zeros(vec![0, ngrams.len()])),
        );
        object.replace_property_unbounded("NumNgrams".to_string(), Value::Num(ngrams.len() as f64));
        object.replace_property_unbounded("NumDocuments".to_string(), Value::Num(0.0));
        Value::Object(object)
    }

    #[test]
    fn encodes_tokenized_documents_against_bag_of_words_rows() {
        let bag = bag_of_words(&["alpha", "beta", "gamma"]);
        let docs = tokenized(&[&["beta", "beta", "delta"], &["alpha", "gamma"]]);

        let out = sparse(run_encode(vec![bag, docs]).expect("encode"));
        assert_eq!((out.rows, out.cols), (2, 3));
        let dense = out.to_dense().unwrap();
        assert_eq!(dense.shape(), &[2, 3]);
        assert_eq!(dense.materialize_f64(), vec![0.0, 1.0, 2.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn encodes_word_vector_with_documents_in_columns() {
        let bag = bag_of_words(&["alpha", "beta", "gamma"]);

        let out = sparse(
            run_encode(vec![
                bag,
                string_array(&["beta", "gamma", "beta"], 1, 3),
                Value::from("DocumentsIn"),
                Value::from("columns"),
            ])
            .expect("encode"),
        );
        assert_eq!((out.rows, out.cols), (3, 1));
        let dense = out.to_dense().unwrap();
        assert_eq!(dense.shape(), &[3, 1]);
        assert_eq!(dense.materialize_f64(), vec![0.0, 2.0, 1.0]);
    }

    #[test]
    fn encodes_multiple_documents_in_columns() {
        let bag = bag_of_words(&["alpha", "beta", "gamma"]);
        let docs = tokenized(&[&["beta", "beta", "delta"], &["alpha", "gamma"]]);

        let out = sparse(
            run_encode(vec![
                bag,
                docs,
                Value::from("DocumentsIn"),
                Value::from("columns"),
            ])
            .expect("encode"),
        );
        assert_eq!((out.rows, out.cols), (3, 2));
        let dense = out.to_dense().unwrap();
        assert_eq!(dense.shape(), &[3, 2]);
        assert_eq!(dense.materialize_f64(), vec![0.0, 2.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn returns_sparse_zeros_for_empty_bag_and_unknown_terms() {
        let empty = sparse(run_encode(vec![bag_of_words(&[]), tokenized(&[&["alpha"]])]).unwrap());
        assert_eq!((empty.rows, empty.cols), (1, 0));
        assert_eq!(empty.col_ptrs, vec![0]);
        assert!(empty.row_indices.is_empty());
        assert!(empty.materialize_f64().is_empty());

        let unknown = sparse(
            run_encode(vec![bag_of_words(&["alpha", "beta"]), Value::from("gamma")]).unwrap(),
        );
        assert_eq!((unknown.rows, unknown.cols), (1, 2));
        assert_eq!(unknown.col_ptrs, vec![0, 0, 0]);
        assert!(unknown.row_indices.is_empty());
        assert!(unknown.materialize_f64().is_empty());
    }

    #[test]
    fn force_cell_output_wraps_sparse_result() {
        let bag = bag_of_words(&["alpha", "beta"]);
        let out = run_encode(vec![
            bag,
            Value::from("alpha"),
            Value::from("ForceCellOutput"),
            Value::Bool(true),
        ])
        .expect("encode");
        let Value::Cell(cell) = out else {
            panic!("expected cell");
        };
        assert_eq!((cell.rows, cell.cols), (1, 1));
        let Value::SparseTensor(sparse) = &cell.data()[0] else {
            panic!("expected sparse cell element");
        };
        assert_eq!((sparse.rows, sparse.cols), (1, 2));
        let dense = sparse.to_dense().unwrap();
        assert_eq!(dense.shape(), &[1, 2]);
        assert_eq!(dense.materialize_f64(), vec![1.0, 0.0]);
    }

    #[test]
    fn encodes_bag_of_ngrams_documents() {
        let bag = bag_of_ngrams(&[&["a"], &["b"], &["a", "b"], &["b", "a"]], &[1, 2]);
        let docs = tokenized(&[&["a", "b", "a", "b"]]);

        let out = sparse(run_encode(vec![bag, docs]).expect("encode"));
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, 4);
        let dense = out.to_dense().unwrap();
        assert_eq!(dense.shape(), &[1, 4]);
        assert_eq!(dense.materialize_f64(), vec![2.0, 2.0, 2.0, 1.0]);
    }

    #[test]
    fn rejects_malformed_bag_of_ngrams_metadata() {
        let err = run_encode(vec![bag_of_ngrams(&[&[]], &[]), tokenized(&[&["a"]])])
            .expect_err("expected empty ngram rejection");
        assert!(err.to_string().contains("empty n-gram"));

        let err = run_encode(vec![
            bag_of_ngrams(&[&["a"], &["a"]], &[1]),
            tokenized(&[&["a"]]),
        ])
        .expect_err("expected duplicate ngram rejection");
        assert!(err.to_string().contains("duplicate n-gram"));
    }

    #[test]
    fn rejects_bad_options_and_column_word_vectors() {
        let bag = bag_of_words(&["alpha"]);
        let err = run_encode(vec![
            bag.clone(),
            Value::from("alpha"),
            Value::from("DocumentsIn"),
            Value::from("pages"),
        ])
        .expect_err("expected bad option");
        assert!(err.to_string().contains("DocumentsIn"));

        let err = run_encode(vec![bag, string_array(&["alpha", "beta"], 2, 1)])
            .expect_err("expected column rejection");
        assert!(err.to_string().contains("row word vector"));
    }

    #[test]
    fn rejects_invalid_force_cell_output_and_odd_options() {
        let bag = bag_of_words(&["alpha"]);
        let err = run_encode(vec![
            bag.clone(),
            Value::from("alpha"),
            Value::from("ForceCellOutput"),
            Value::from("yes"),
        ])
        .expect_err("expected invalid force cell output");
        assert!(err.to_string().contains("ForceCellOutput"));

        let err = run_encode(vec![bag, Value::from("alpha"), Value::from("DocumentsIn")])
            .expect_err("expected odd options rejection");
        assert!(err.to_string().contains("name-value options"));
    }

    #[test]
    fn integer_force_cell_output_accepts_all_classes_exactly_in_runmat_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for flag in [
            IntValue::I8(-1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(i64::MAX),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ] {
            let out = run_encode(vec![
                bag_of_words(&["alpha"]),
                Value::from("alpha"),
                Value::from("ForceCellOutput"),
                Value::Int(flag),
            ])
            .unwrap();
            assert!(matches!(out, Value::Cell(_)));
        }
        let zero = Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1]).unwrap();
        let out = run_encode(vec![
            bag_of_words(&["alpha"]),
            Value::from("alpha"),
            Value::from("ForceCellOutput"),
            Value::Tensor(zero),
        ])
        .unwrap();
        assert!(matches!(out, Value::SparseTensor(_)));
    }

    #[test]
    fn integer_force_cell_output_is_gated_in_matlab_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = run_encode(vec![
            bag_of_words(&["alpha"]),
            Value::from("alpha"),
            Value::from("ForceCellOutput"),
            Value::Int(IntValue::U64(u64::MAX)),
        ])
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:EncodeNumericForceCellOutputExtension")
        );
    }

    #[test]
    fn resident_numeric_documents_reject_before_provider_access() {
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("resident numeric admission fixture")
                .handle,
        );
        let err = run_encode(vec![bag_of_words(&["alpha"]), resident]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:encode:InvalidInput"));
    }

    #[test]
    fn resident_force_cell_output_rejects_before_provider_access_in_matlab_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("resident numeric admission fixture")
                .handle,
        );
        let err = run_encode(vec![
            bag_of_words(&["alpha"]),
            Value::from("alpha"),
            Value::from("ForceCellOutput"),
            resident,
        ])
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:EncodeResidentForceCellOutputExtension")
        );
    }

    #[test]
    fn encode_dispatch_preserves_residency_until_builtin_preflight() {
        assert_eq!(GPU_SPEC.residency, ResidencyPolicy::NewHandle);
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("resident numeric admission fixture");
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
                .is_none()
        );
        let resident = Value::GpuTensor(allocation.handle);
        let prepared = futures::executor::block_on(runmat_accelerate::prepare_builtin_args(
            "encode",
            &[resident],
        ))
        .expect("dispatcher must retain resident argument");
        assert!(matches!(prepared.as_slice(), [Value::GpuTensor(_)]));
        assert_eq!(allocation.probe.retirements(), 0);
        let Value::GpuTensor(retained) = &prepared[0] else {
            unreachable!()
        };
        retained.allocation_lease().validate_live().unwrap();
        drop(prepared);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
