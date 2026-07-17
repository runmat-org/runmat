//! Count-matrix encoding for Text Analytics bag models.

use std::collections::{BTreeMap, HashMap, HashSet};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ObjectInstance, ResolveContext, SparseTensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    documents_from_object, vocabulary_from_bag, words_from_word_vector, BAG_OF_WORDS_CLASS,
    TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::ngrams::{ngrams_from_bag, BAG_OF_NGRAMS_CLASS};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

const OUT_COUNTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "counts",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse word or n-gram count matrix.",
}];

const IN_BAG_INPUT_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "bag",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "bagOfWords or bagOfNgrams model.",
    },
    BuiltinParamDescriptor {
        name: "documentsOrWords",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object or row word vector.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: DocumentsIn, ForceCellOutput.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENCODE.INVALID_INPUT",
    identifier: Some("RunMat:encode:InvalidInput"),
    when: "Inputs do not match a supported Text Analytics encode form.",
    message: "encode: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const ENCODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "counts = encode(bag, documentsOrWords, Name, Value, ...)",
        inputs: &IN_BAG_INPUT_REST,
        outputs: &OUT_COUNTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn encode_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("encode")
        .with_identifier("RunMat:encode:InvalidInput")
        .build()
}

#[runtime_builtin(
    name = "encode",
    category = "strings/text_analytics",
    summary = "Encode documents as sparse word or n-gram count matrices.",
    keywords = "encode,text analytics,bagOfWords,bagOfNgrams,count matrix",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::encode::ENCODE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::encode"
)]
async fn encode_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args).await?;
    let (bag, input, options) = parse_args(gathered)?;
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
                object.class_name
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

async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(
            gather_if_needed_async(&arg)
                .await
                .map_err(|err| encode_error(format!("encode: failed to gather input: {err}")))?,
        );
    }
    Ok(out)
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

fn parse_args(mut args: Vec<Value>) -> BuiltinResult<(Value, Value, EncodeOptions)> {
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
    let bag = args.remove(0);
    let input = args.remove(0);
    let mut options = EncodeOptions::default();
    let mut idx = 0;
    while idx < args.len() {
        let name =
            scalar_text(&args[idx], "encode").map_err(|err| encode_error(err.to_string()))?;
        match name.to_ascii_lowercase().as_str() {
            "documentsin" => {
                let value = scalar_text(&args[idx + 1], "encode")
                    .map_err(|err| encode_error(err.to_string()))?;
                options.documents_in = match value.to_ascii_lowercase().as_str() {
                    "rows" => DocumentsIn::Rows,
                    "columns" => DocumentsIn::Columns,
                    other => {
                        return Err(encode_error(format!(
                            "encode: DocumentsIn must be 'rows' or 'columns', got '{other}'"
                        )))
                    }
                };
            }
            "forcecelloutput" => {
                options.force_cell_output = parse_bool_scalar(&args[idx + 1])?;
            }
            other => {
                return Err(encode_error(format!(
                    "encode: unsupported option '{other}'"
                )))
            }
        }
        idx += 2;
    }
    Ok((bag, input, options))
}

fn parse_bool_scalar(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
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
            object.class_name
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
    use runmat_builtins::{StringArray, Tensor};

    fn run_encode(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(encode_builtin(args))
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
        object.properties.insert(
            "Documents".to_string(),
            Value::Cell(CellArray::new(data, docs.len(), 1).expect("documents cell")),
        );
        object
            .properties
            .insert("NumDocuments".to_string(), Value::Num(docs.len() as f64));
        Value::Object(object)
    }

    fn bag_of_words(vocabulary: &[&str]) -> Value {
        let mut object = ObjectInstance::new(BAG_OF_WORDS_CLASS.to_string());
        object.properties.insert(
            "Vocabulary".to_string(),
            string_array(vocabulary, 1, vocabulary.len()),
        );
        object.properties.insert(
            "Counts".to_string(),
            Value::Tensor(Tensor::zeros(vec![0, vocabulary.len()])),
        );
        object
            .properties
            .insert("NumWords".to_string(), Value::Num(vocabulary.len() as f64));
        object
            .properties
            .insert("NumDocuments".to_string(), Value::Num(0.0));
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
        object.properties.insert(
            "Ngrams".to_string(),
            Value::StringArray(StringArray::new(data, vec![rows, cols]).expect("ngrams")),
        );
        object.properties.insert(
            "NgramLengths".to_string(),
            Value::Tensor(
                Tensor::new(
                    lengths.iter().map(|length| *length as f64).collect(),
                    vec![1, lengths.len()],
                )
                .expect("lengths"),
            ),
        );
        object.properties.insert(
            "Counts".to_string(),
            Value::Tensor(Tensor::zeros(vec![0, ngrams.len()])),
        );
        object
            .properties
            .insert("NumNgrams".to_string(), Value::Num(ngrams.len() as f64));
        object
            .properties
            .insert("NumDocuments".to_string(), Value::Num(0.0));
        Value::Object(object)
    }

    #[test]
    fn encodes_tokenized_documents_against_bag_of_words_rows() {
        let bag = bag_of_words(&["alpha", "beta", "gamma"]);
        let docs = tokenized(&[&["beta", "beta", "delta"], &["alpha", "gamma"]]);

        let out = sparse(run_encode(vec![bag, docs]).expect("encode"));
        assert_eq!((out.rows, out.cols), (2, 3));
        let dense = out.to_dense().unwrap();
        assert_eq!(dense.shape, vec![2, 3]);
        assert_eq!(dense.data, vec![0.0, 1.0, 2.0, 0.0, 0.0, 1.0]);
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
        assert_eq!(dense.shape, vec![3, 1]);
        assert_eq!(dense.data, vec![0.0, 2.0, 1.0]);
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
        assert_eq!(dense.shape, vec![3, 2]);
        assert_eq!(dense.data, vec![0.0, 2.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn returns_sparse_zeros_for_empty_bag_and_unknown_terms() {
        let empty = sparse(run_encode(vec![bag_of_words(&[]), tokenized(&[&["alpha"]])]).unwrap());
        assert_eq!((empty.rows, empty.cols), (1, 0));
        assert_eq!(empty.col_ptrs, vec![0]);
        assert!(empty.row_indices.is_empty());
        assert!(empty.values.is_empty());

        let unknown = sparse(
            run_encode(vec![bag_of_words(&["alpha", "beta"]), Value::from("gamma")]).unwrap(),
        );
        assert_eq!((unknown.rows, unknown.cols), (1, 2));
        assert_eq!(unknown.col_ptrs, vec![0, 0, 0]);
        assert!(unknown.row_indices.is_empty());
        assert!(unknown.values.is_empty());
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
        let Value::SparseTensor(sparse) = &cell.data[0] else {
            panic!("expected sparse cell element");
        };
        assert_eq!((sparse.rows, sparse.cols), (1, 2));
        let dense = sparse.to_dense().unwrap();
        assert_eq!(dense.shape, vec![1, 2]);
        assert_eq!(dense.data, vec![1.0, 0.0]);
    }

    #[test]
    fn encodes_bag_of_ngrams_documents() {
        let bag = bag_of_ngrams(&[&["a"], &["b"], &["a", "b"], &["b", "a"]], &[1, 2]);
        let docs = tokenized(&[&["a", "b", "a", "b"]]);

        let out = sparse(run_encode(vec![bag, docs]).expect("encode"));
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, 4);
        let dense = out.to_dense().unwrap();
        assert_eq!(dense.shape, vec![1, 4]);
        assert_eq!(dense.data, vec![2.0, 2.0, 2.0, 1.0]);
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
}
