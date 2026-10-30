//! Bag-of-n-grams compatibility object for Text Analytics workflows.
use runmat_types::MemberAccess;

use std::collections::{HashMap, HashSet};

use runmat_builtins::BAG_OF_NGRAMS_ERROR_INVALID_INPUT;
pub use runmat_builtins::{BAG_OF_NGRAMS_DESCRIPTOR, BAG_OF_NGRAMS_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, IntegerStorage, NumericScalar, ObjectInstance, StringArray, Tensor, Value,
};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::common::is_missing_string;
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    checked_count_len, documents_from_object, is_nonnegative_integer_count, words_from_word_vector,
    words_from_word_vector_preserving_missing, TOKENIZED_DOCUMENT_CLASS,
};
use crate::{build_runtime_error, BuiltinResult};

pub const BAG_OF_NGRAMS_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("bagOfNgrams");

static BAG_OF_NGRAMS_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(BAG_OF_NGRAMS_CLASS);

fn ngrams_error(message: impl Into<String>) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("bagOfNgrams");
    if let Some(identifier) = BAG_OF_NGRAMS_ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ensure_bag_of_ngrams_class_registered() {
    BAG_OF_NGRAMS_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in [
            "Counts",
            "Ngrams",
            "NgramLengths",
            "Vocabulary",
            "NumNgrams",
            "NumDocuments",
        ] {
            properties.insert(name.into(), property_def(name));
        }
        crate::class_registry::register_builtin_class(
            BAG_OF_NGRAMS_CLASS,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: BAG_OF_NGRAMS_CLASS.into(),
                parent: None,
                properties,
                methods: HashMap::new(),
            },
        );
    });
}

fn property_def(name: &str) -> crate::class_registry::RuntimeProperty {
    crate::class_registry::RuntimeProperty {
        name: name.into(),
        is_static: false,
        is_constant: false,
        is_dependent: false,
        get_access: MemberAccess::Public,
        set_access: MemberAccess::Public,
        default_value: None,
    }
}

#[runtime_builtin(
    name = "bagOfNgrams",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::text_analytics::ngrams"
)]
async fn bag_of_ngrams_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if args.iter().any(|arg| matches!(arg, Value::GpuTensor(_))) {
        return Err(ngrams_error(
            "bagOfNgrams: GPU-resident inputs are not supported",
        ));
    }
    let parsed = parse_args(args)?;
    match parsed.source {
        NgramSource::Empty => {
            bag_object(runtime.as_ref(), Vec::new(), parsed.lengths, Vec::new(), 0)
        }
        NgramSource::Documents(documents) => {
            bag_from_documents(runtime.as_ref(), documents, parsed.lengths)
        }
        NgramSource::Unique {
            ngrams,
            counts,
            requested_lengths,
        } => bag_from_unique_ngrams(runtime.as_ref(), ngrams, counts, requested_lengths),
    }
}

struct ParsedArgs {
    source: NgramSource,
    lengths: Vec<usize>,
}

enum NgramSource {
    Empty,
    Documents(Vec<Vec<String>>),
    Unique {
        ngrams: Vec<Vec<runmat_value::StringElement>>,
        counts: Tensor,
        requested_lengths: Option<Vec<usize>>,
    },
}

fn parse_args(args: crate::call::BuiltinArguments) -> BuiltinResult<ParsedArgs> {
    if args.is_empty() {
        return Ok(ParsedArgs {
            source: NgramSource::Empty,
            lengths: vec![2],
        });
    }

    let first_is_option = is_option_name(&args[0], "NgramLengths");
    if first_is_option {
        let lengths = parse_options(args.view(), 0)?.unwrap_or_else(|| vec![2]);
        return Ok(ParsedArgs {
            source: NgramSource::Empty,
            lengths,
        });
    }

    if args.len() >= 2 && !is_option_name(&args[1], "NgramLengths") {
        let counts = match &args[1] {
            Value::Tensor(tensor) => tensor.clone(),
            Value::Num(value) => Tensor::new(vec![*value], vec![1, 1]).map_err(ngrams_error)?,
            Value::Int(value) => {
                Tensor::new_integer(IntegerStorage::from_scalar(value.clone()), vec![1, 1])
                    .map_err(ngrams_error)?
            }
            other => {
                return Err(ngrams_error(format!(
                    "bagOfNgrams: counts must be a numeric matrix, got {other:?}"
                )))
            }
        };
        let lengths = parse_options(args.view(), 2)?;
        return Ok(ParsedArgs {
            lengths: lengths.clone().unwrap_or_else(|| vec![2]),
            source: NgramSource::Unique {
                ngrams: unique_ngrams_from_value(&args[0], count_matrix_dimensions(&counts).1)?,
                counts,
                requested_lengths: lengths,
            },
        });
    }

    let lengths = parse_options(args.view(), 1)?.unwrap_or_else(|| vec![2]);
    Ok(ParsedArgs {
        source: NgramSource::Documents(documents_from_value(&args[0])?),
        lengths,
    })
}

fn is_option_name(value: &Value, expected: &str) -> bool {
    scalar_text(value, "bagOfNgrams")
        .map(|text| text.eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
    start: usize,
) -> BuiltinResult<Option<Vec<usize>>> {
    if start >= args.len() {
        return Ok(None);
    }
    if !(args.len() - start).is_multiple_of(2) {
        return Err(ngrams_error(
            "bagOfNgrams: name-value options must appear in pairs",
        ));
    }
    let mut lengths = None;
    let mut idx = start;
    while idx < args.len() {
        let name =
            scalar_text(&args[idx], "bagOfNgrams").map_err(|err| ngrams_error(err.to_string()))?;
        match name.to_ascii_lowercase().as_str() {
            "ngramlengths" => {
                lengths = Some(parse_lengths(&args[idx + 1])?);
            }
            other => {
                return Err(ngrams_error(format!(
                    "bagOfNgrams: unsupported option '{other}'"
                )));
            }
        }
        idx += 2;
    }
    Ok(lengths)
}

fn parse_lengths(value: &Value) -> BuiltinResult<Vec<usize>> {
    let raw = match value {
        Value::Num(n) => vec![NumericScalar::F64(*n)],
        Value::Int(value) => vec![NumericScalar::from(value.clone())],
        Value::Tensor(tensor) if tensor_utils::tensor_element_len(tensor) != 0 => (0..tensor.len())
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .expect("tensor storage length matches shape")
            })
            .collect(),
        other => {
            return Err(ngrams_error(format!(
            "bagOfNgrams: NgramLengths must be a positive integer scalar or vector, got {other:?}"
        )))
        }
    };
    let mut lengths = Vec::with_capacity(raw.len());
    let mut seen = HashSet::new();
    for n in raw {
        let Some(len) = positive_length_usize(n) else {
            return Err(ngrams_error(
                "bagOfNgrams: NgramLengths must contain positive integers".to_string(),
            ));
        };
        if seen.insert(len) {
            lengths.push(len);
        }
    }
    Ok(lengths)
}

fn positive_length_usize(value: NumericScalar) -> Option<usize> {
    match value {
        NumericScalar::F64(value) => positive_length_f64(value),
        NumericScalar::F32(value) => positive_length_f64(f64::from(value)),
        value => value
            .into_int_value()?
            .try_to_usize()
            .filter(|value| *value > 0),
    }
}

fn positive_length_f64(value: f64) -> Option<usize> {
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

fn documents_from_value(value: &Value) -> BuiltinResult<Vec<Vec<String>>> {
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => {
            documents_from_object(object, "bagOfNgrams")
        }
        Value::Object(object) => Err(ngrams_error(format!(
            "bagOfNgrams: expected tokenizedDocument object, got {}",
            object.class_name()
        ))),
        other => {
            validate_row_word_vector(other)?;
            Ok(vec![words_from_word_vector(other, "bagOfNgrams")?])
        }
    }
}

fn validate_row_word_vector(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::String(_) | Value::Num(_) => Ok(()),
        Value::StringArray(array) if array.rows <= 1 => Ok(()),
        Value::CharArray(CharArray { rows, .. }) if *rows <= 1 => Ok(()),
        Value::Cell(cell) if cell.rows <= 1 => Ok(()),
        Value::StringArray(array) => Err(ngrams_error(format!(
            "bagOfNgrams: non-tokenized documents input must be a row word vector; got string array with shape {}x{}",
            array.rows, array.cols
        ))),
        Value::CharArray(CharArray { rows, cols, .. }) => Err(ngrams_error(format!(
            "bagOfNgrams: non-tokenized documents input must be a row word vector; got char array with shape {rows}x{cols}"
        ))),
        Value::Cell(cell) => Err(ngrams_error(format!(
            "bagOfNgrams: non-tokenized documents input must be a row word vector; got cell array with shape {}x{}",
            cell.rows, cell.cols
        ))),
        _ => Ok(()),
    }
}

fn bag_from_documents(
    runtime: Option<&crate::context::RuntimeContext>,
    documents: Vec<Vec<String>>,
    lengths: Vec<usize>,
) -> BuiltinResult<Value> {
    let mut ngrams = Vec::new();
    let mut positions: HashMap<Vec<String>, usize> = HashMap::new();
    let rows = documents.len();
    let mut counts = Vec::<f64>::new();

    for (doc_idx, document) in documents.iter().enumerate() {
        for &length in &lengths {
            if length > document.len() {
                continue;
            }
            for start in 0..=document.len() - length {
                let ngram = document[start..start + length].to_vec();
                let col = if let Some(col) = positions.get(&ngram) {
                    *col
                } else {
                    let col = ngrams.len();
                    positions.insert(ngram.clone(), col);
                    ngrams.push(ngram);
                    counts.resize(checked_count_len(rows, ngrams.len(), "bagOfNgrams")?, 0.0);
                    col
                };
                counts[doc_idx + col * rows] += 1.0;
            }
        }
    }

    bag_object(runtime, ngrams, lengths, counts, rows)
}

fn bag_from_unique_ngrams(
    runtime: Option<&crate::context::RuntimeContext>,
    raw_ngrams: Vec<Vec<runmat_value::StringElement>>,
    counts: Tensor,
    requested_lengths: Option<Vec<usize>>,
) -> BuiltinResult<Value> {
    let (count_rows, count_cols) = count_matrix_dimensions(&counts);
    if count_cols != raw_ngrams.len() {
        return Err(ngrams_error(format!(
            "bagOfNgrams: counts columns ({}) must match uniqueNgrams rows ({})",
            count_cols,
            raw_ngrams.len()
        )));
    }
    if (0..counts.len()).any(|index| {
        !is_nonnegative_integer_count(
            counts
                .numeric_value_at(index)
                .expect("tensor storage length matches shape"),
        )
    }) {
        return Err(ngrams_error(
            "bagOfNgrams: counts must be nonnegative integers",
        ));
    }

    let requested = requested_lengths
        .as_ref()
        .map(|lengths| lengths.iter().copied().collect::<HashSet<_>>());
    let mut seen = HashSet::new();
    let mut keep_cols = Vec::new();
    let mut ngrams = Vec::new();
    for (col, ngram) in raw_ngrams.iter().enumerate() {
        if ngram.iter().any(|word| is_missing_string(word)) {
            continue;
        }
        let ngram = ngram
            .iter()
            .filter_map(|word| word.as_str().map(str::to_owned))
            .collect::<Vec<_>>();
        if ngram.is_empty() {
            return Err(ngrams_error(
                "bagOfNgrams: each n-gram must contain at least one word",
            ));
        }
        if requested
            .as_ref()
            .is_some_and(|lengths| !lengths.contains(&ngram.len()))
        {
            continue;
        }
        if !seen.insert(ngram.clone()) {
            return Err(ngrams_error(format!(
                "bagOfNgrams: uniqueNgrams contains duplicate n-gram '{}'",
                ngram.join(" ")
            )));
        }
        keep_cols.push(col);
        ngrams.push(ngram.clone());
    }

    let mut filtered_counts = Vec::with_capacity(checked_count_len(
        count_rows,
        keep_cols.len(),
        "bagOfNgrams",
    )?);
    let values = tensor_utils::tensor_values_f64_cow(&counts);
    for col in keep_cols {
        for row in 0..count_rows {
            filtered_counts.push(values[row + col * count_rows]);
        }
    }
    let lengths = requested_lengths.unwrap_or_else(|| infer_ngram_lengths(&ngrams));
    bag_object(runtime, ngrams, lengths, filtered_counts, count_rows)
}

/// `bagOfNgrams` treats a numeric vector of counts as one document row. This
/// is the matrix projection historically carried by dense tensor metadata,
/// rather than the public vector projection used by general array APIs.
fn count_matrix_dimensions(counts: &Tensor) -> (usize, usize) {
    let shape = counts.shape();
    match shape {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

fn infer_ngram_lengths(ngrams: &[Vec<String>]) -> Vec<usize> {
    let mut lengths = Vec::new();
    let mut seen = HashSet::new();
    for ngram in ngrams {
        if seen.insert(ngram.len()) {
            lengths.push(ngram.len());
        }
    }
    if lengths.is_empty() {
        lengths.push(2);
    }
    lengths
}

fn unique_ngrams_from_value(
    value: &Value,
    expected_rows: usize,
) -> BuiltinResult<Vec<Vec<runmat_value::StringElement>>> {
    match value {
        Value::StringArray(array) => {
            if array.rows != expected_rows {
                return Err(ngrams_error(format!(
                    "bagOfNgrams: uniqueNgrams rows ({}) must match counts columns ({expected_rows})",
                    array.rows
                )));
            }
            let mut out = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let mut ngram = Vec::new();
                let mut row_has_missing = false;
                for col in 0..array.cols {
                    let word = array.data()[row + col * array.rows].clone();
                    if is_missing_string(&word) {
                        row_has_missing = true;
                        break;
                    }
                    if word.as_str().is_some_and(|word| !word.is_empty()) {
                        ngram.push(word);
                    }
                }
                if row_has_missing {
                    out.push(vec![runmat_value::StringElement::Missing]);
                } else {
                    out.push(ngram);
                }
            }
            Ok(out)
        }
        Value::Cell(cell) => {
            if cell.rows != expected_rows {
                return Err(ngrams_error(format!(
                    "bagOfNgrams: uniqueNgrams rows ({}) must match counts columns ({expected_rows})",
                    cell.rows
                )));
            }
            let mut out = Vec::with_capacity(cell.rows);
            for row in 0..cell.rows {
                let mut ngram = Vec::new();
                let mut row_has_missing = false;
                for col in 0..cell.cols {
                    let idx = row + col * cell.rows;
                    let word = match &cell.data()[idx] {
                        Value::String(text) => text.clone(),
                        Value::StringArray(array) if array.data().len() == 1 => {
                            array.data()[0].clone()
                        }
                        value => scalar_text(value, "bagOfNgrams")
                            .map(runmat_value::StringElement::text)
                            .map_err(|err| ngrams_error(err.to_string()))?,
                    };
                    if is_missing_string(&word) {
                        row_has_missing = true;
                        break;
                    }
                    if word.as_str().is_some_and(|word| !word.is_empty()) {
                        ngram.push(word);
                    }
                }
                if row_has_missing {
                    out.push(vec![runmat_value::StringElement::Missing]);
                } else {
                    out.push(ngram);
                }
            }
            Ok(out)
        }
        other => {
            let words = words_from_word_vector_preserving_missing(other, "bagOfNgrams")?;
            if expected_rows != 1 {
                return Err(ngrams_error(format!(
                    "bagOfNgrams: uniqueNgrams rows (1) must match counts columns ({expected_rows})"
                )));
            }
            Ok(vec![words
                .into_iter()
                .filter(|word| word.as_str().is_none_or(|word| !word.is_empty()))
                .collect()])
        }
    }
}

fn bag_object(
    runtime: Option<&crate::context::RuntimeContext>,
    ngrams: Vec<Vec<String>>,
    lengths: Vec<usize>,
    counts: Vec<f64>,
    rows: usize,
) -> BuiltinResult<Value> {
    ensure_bag_of_ngrams_class_registered();
    let cols = ngrams.len();
    let expected = checked_count_len(rows, cols, "bagOfNgrams")?;
    if counts.len() != expected {
        return Err(ngrams_error(format!(
            "bagOfNgrams: count storage has {} values but expected {} for a {}x{} model",
            counts.len(),
            expected,
            rows,
            cols
        )));
    }
    let max_len = ngrams.iter().map(Vec::len).max().unwrap_or(0);
    let fields = [
        ("Ngrams", Value::StringArray(ngram_array(&ngrams, max_len)?)),
        (
            "Counts",
            Value::Tensor(Tensor::new(counts, vec![rows, cols]).map_err(|err| ngrams_error(err))?),
        ),
        (
            "NgramLengths",
            Value::Tensor(
                Tensor::new(
                    lengths.iter().map(|length| *length as f64).collect(),
                    vec![1, lengths.len()],
                )
                .map_err(|err| ngrams_error(err))?,
            ),
        ),
        ("Vocabulary", Value::StringArray(vocabulary_array(&ngrams)?)),
        ("NumNgrams", Value::Num(cols as f64)),
        ("NumDocuments", Value::Num(rows as f64)),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        BAG_OF_NGRAMS_CLASS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}

pub(in crate::builtins::strings::text_analytics) fn ngrams_from_bag(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Vec<Vec<String>>> {
    match object.properties().get("Ngrams") {
        Some(Value::StringArray(array)) => {
            let mut ngrams = Vec::with_capacity(array.rows);
            let mut seen = HashSet::new();
            for row in 0..array.rows {
                let mut ngram = Vec::new();
                for col in 0..array.cols {
                    let word = &array.data()[row + col * array.rows];
                    if let Some(word) = word.as_str().filter(|word| !word.is_empty()) {
                        ngram.push(word.to_owned());
                    }
                }
                if ngram.is_empty() {
                    return Err(ngrams_error(format!(
                        "{fn_name}: bagOfNgrams object contains an empty n-gram"
                    )));
                }
                if !seen.insert(ngram.clone()) {
                    return Err(ngrams_error(format!(
                        "{fn_name}: bagOfNgrams object contains duplicate n-gram '{}'",
                        ngram.join(" ")
                    )));
                }
                ngrams.push(ngram);
            }
            Ok(ngrams)
        }
        Some(other) => Err(ngrams_error(format!(
            "{fn_name}: bagOfNgrams Ngrams property must be a string array, got {other:?}"
        ))),
        None => Err(ngrams_error(format!(
            "{fn_name}: bagOfNgrams object missing Ngrams property"
        ))),
    }
}

fn ngram_array(ngrams: &[Vec<String>], max_len: usize) -> BuiltinResult<StringArray> {
    let rows = ngrams.len();
    let mut data = Vec::with_capacity(rows * max_len);
    for col in 0..max_len {
        for ngram in ngrams {
            data.push(ngram.get(col).cloned().unwrap_or_default());
        }
    }
    StringArray::new(data, vec![rows, max_len]).map_err(|err| ngrams_error(err))
}

fn vocabulary_array(ngrams: &[Vec<String>]) -> BuiltinResult<StringArray> {
    let mut seen = HashSet::new();
    let mut words = Vec::new();
    for word in ngrams.iter().flatten() {
        if seen.insert(word.clone()) {
            words.push(word.clone());
        }
    }
    StringArray::new(words.clone(), vec![1, words.len()]).map_err(|err| ngrams_error(err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::strings::text_analytics::documents::TOKENIZED_DOCUMENT_CLASS;
    use runmat_value::{CellArray, IntegerStorage, NumericStorage};

    fn run(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(bag_of_ngrams_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn object(value: Value) -> ObjectInstance {
        let Value::Object(object) = value else {
            panic!("expected object");
        };
        object
    }

    fn tokenized(documents: Vec<Vec<&str>>) -> Value {
        let values = documents
            .into_iter()
            .map(|doc| {
                let len = doc.len();
                Value::StringArray(
                    StringArray::new(
                        doc.into_iter().map(str::to_string).collect::<Vec<_>>(),
                        vec![1, len],
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let rows = values.len();
        let mut object = ObjectInstance::new(TOKENIZED_DOCUMENT_CLASS.to_string());
        object.replace_property_unbounded(
            "Documents".to_string(),
            Value::Cell(CellArray::new(values, rows, 1).unwrap()),
        );
        Value::Object(object)
    }

    #[test]
    fn parse_lengths_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 3]), vec![1, 2])
            .expect("integer tensor");

        assert_eq!(
            parse_lengths(&Value::Tensor(tensor)).expect("lengths"),
            vec![1, 3]
        );
    }

    #[test]
    fn accepts_all_integer_count_and_length_classes_with_double_model_metadata() {
        let ngrams = || {
            StringArray::new(
                vec!["a".into(), "b".into(), "b".into(), "c".into()],
                vec![2, 2],
            )
            .expect("ngrams")
        };
        let counts = [
            IntegerStorage::I8(vec![2, 3]),
            IntegerStorage::I16(vec![2, 3]),
            IntegerStorage::I32(vec![2, 3]),
            IntegerStorage::I64(vec![2, 3]),
            IntegerStorage::U8(vec![2, 3]),
            IntegerStorage::U16(vec![2, 3]),
            IntegerStorage::U32(vec![2, 3]),
            IntegerStorage::U64(vec![2, 3]),
        ];
        let lengths = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];
        for (count_storage, length_storage) in counts.into_iter().zip(lengths) {
            let count_tensor =
                Tensor::new_integer(count_storage, vec![1, 2]).expect("integer counts");
            let length_tensor =
                Tensor::new_integer(length_storage, vec![1, 1]).expect("integer length");
            let bag = object(
                run(vec![
                    Value::StringArray(ngrams()),
                    Value::Tensor(count_tensor),
                    Value::String("NgramLengths".into()),
                    Value::Tensor(length_tensor),
                ])
                .expect("integer bag"),
            );
            let actual_counts = tensor_property(&bag, "Counts");
            assert_eq!(
                actual_counts.numeric_dtype(),
                runmat_value::NumericDType::F64
            );
            assert_eq!(actual_counts.materialize_f64(), vec![2.0, 3.0]);
            let actual_lengths = tensor_property(&bag, "NgramLengths");
            assert_eq!(
                actual_lengths.numeric_dtype(),
                runmat_value::NumericDType::F64
            );
            assert_eq!(actual_lengths.materialize_f64(), vec![2.0]);
        }
    }

    #[test]
    fn exact_wide_counts_cross_only_the_documented_double_property_boundary() {
        let ngrams = StringArray::new(vec!["wide".into()], vec![1, 1]).expect("ngrams");
        let wide = (1_u64 << 53) + 1;
        let counts =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("counts");
        let bag =
            object(run(vec![Value::StringArray(ngrams), Value::Tensor(counts)]).expect("bag"));
        let actual = tensor_property(&bag, "Counts");
        assert_eq!(actual.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(actual.materialize_f64(), vec![wide as f64]);
    }

    #[test]
    fn rejects_resident_inputs_before_provider_access() {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("resident numeric admission fixture");
        let caller_alias = allocation.handle.clone();
        let error = run(vec![Value::GpuTensor(allocation.handle)])
            .expect_err("resident text-model input must reject");
        assert_eq!(error.identifier(), Some("RunMat:bagOfNgrams:InvalidInput"));
        assert_eq!(allocation.probe.retirements(), 0);
        caller_alias.allocation_lease().validate_live().unwrap();
        drop(error);
        drop(caller_alias);
        assert_eq!(allocation.probe.retirements(), 1);
    }

    #[test]
    fn integer_capabilities_cover_counts_and_lengths() {
        assert_eq!(BAG_OF_NGRAMS_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            BAG_OF_NGRAMS_INTEGER_CAPABILITIES[0].inputs[0].classes,
            crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
        );
        assert_eq!(
            BAG_OF_NGRAMS_INTEGER_CAPABILITIES[1].inputs[0].classes,
            crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
        );
    }

    #[test]
    fn parse_lengths_rejects_empty_typed_integer_storage_without_mirror() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 0]).expect("empty");

        let err = parse_lengths(&Value::Tensor(tensor)).expect_err("empty should reject");
        assert!(
            err.message()
                .contains("NgramLengths must be a positive integer scalar or vector"),
            "{}",
            err.message()
        );
    }

    fn string_array_property(object: &ObjectInstance, name: &str) -> StringArray {
        let Some(Value::StringArray(array)) = object.properties().get(name) else {
            panic!("expected string array property {name}");
        };
        array.clone()
    }

    fn tensor_property(object: &ObjectInstance, name: &str) -> Tensor {
        let Some(Value::Tensor(tensor)) = object.properties().get(name) else {
            panic!("expected tensor property {name}");
        };
        tensor.clone()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn counts_default_bigrams_from_tokenized_documents() {
        let bag = object(
            run(vec![tokenized(vec![
                vec!["a", "b", "a"],
                vec!["a", "b", "c"],
            ])])
            .expect("bag"),
        );
        assert!(bag.class_name().is(BAG_OF_NGRAMS_CLASS));
        assert_eq!(bag.properties().get("NumDocuments"), Some(&Value::Num(2.0)));
        assert_eq!(bag.properties().get("NumNgrams"), Some(&Value::Num(3.0)));

        let ngrams = string_array_property(&bag, "Ngrams");
        assert_eq!(ngrams.shape, vec![3, 2]);
        assert_eq!(
            ngrams.data(),
            vec!["a", "b", "b", "b", "a", "c"]
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        );
        let counts = tensor_property(&bag, "Counts");
        assert_eq!(counts.shape(), [2, 3]);
        assert_eq!(counts.materialize_f64(), vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accepts_ngram_lengths_vector() {
        let lengths = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let bag = object(
            run(vec![
                Value::StringArray(
                    StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![1, 3]).unwrap(),
                ),
                Value::String("NgramLengths".to_string().into()),
                Value::Tensor(lengths),
            ])
            .expect("bag"),
        );
        let ngram_lengths = tensor_property(&bag, "NgramLengths");
        assert_eq!(ngram_lengths.materialize_f64(), vec![1.0, 3.0]);
        assert_eq!(bag.properties().get("NumNgrams"), Some(&Value::Num(4.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accepts_unique_ngram_matrix_and_counts() {
        let ngrams = StringArray::new(
            vec!["a".into(), "b".into(), "b".into(), "c".into()],
            vec![2, 2],
        )
        .unwrap();
        let counts = Tensor::new(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let bag =
            object(run(vec![Value::StringArray(ngrams), Value::Tensor(counts)]).expect("bag"));
        assert_eq!(bag.properties().get("NumDocuments"), Some(&Value::Num(2.0)));
        assert_eq!(bag.properties().get("NumNgrams"), Some(&Value::Num(2.0)));
        assert_eq!(
            tensor_property(&bag, "NgramLengths").materialize_f64(),
            vec![2.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn accepts_native_single_and_rejects_negative_exact_integer_counts() {
        let ngrams = StringArray::new(
            vec!["a".into(), "b".into(), "b".into(), "c".into()],
            vec![2, 2],
        )
        .unwrap();
        let single = Tensor::from_numeric_storage(NumericStorage::F32(vec![2.0, 3.0]), vec![1, 2])
            .expect("single counts");
        let bag = object(
            run(vec![
                Value::StringArray(ngrams.clone()),
                Value::Tensor(single),
            ])
            .expect("single bag"),
        );
        assert_eq!(
            tensor_property(&bag, "Counts").materialize_f64(),
            vec![2.0, 3.0]
        );

        let negative = Tensor::new_integer(IntegerStorage::I64(vec![1, -1]), vec![1, 2]).unwrap();
        assert!(run(vec![Value::StringArray(ngrams), Value::Tensor(negative),]).is_err());

        let one_ngram = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let rank_zero = Tensor::new(vec![1.0], Vec::new()).expect("rank-zero counts");
        let error = run(vec![
            Value::StringArray(one_ngram),
            Value::Tensor(rank_zero),
        ])
        .expect_err("rank-zero counts retain zero-column matrix geometry");
        assert!(
            error.message().contains("counts columns (0)"),
            "unexpected error: {error}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn infers_and_filters_unique_ngram_lengths() {
        let ngrams = StringArray::new(
            vec![
                "a".into(),
                "b".into(),
                "d".into(),
                "".into(),
                "c".into(),
                "e".into(),
            ],
            vec![3, 2],
        )
        .unwrap();
        let counts = Tensor::new(vec![2.0, 0.0, 4.0], vec![1, 3]).unwrap();
        let bag = object(
            run(vec![
                Value::StringArray(ngrams.clone()),
                Value::Tensor(counts.clone()),
            ])
            .expect("bag"),
        );
        assert_eq!(
            tensor_property(&bag, "NgramLengths").materialize_f64(),
            vec![1.0, 2.0]
        );

        let filtered = object(
            run(vec![
                Value::StringArray(ngrams),
                Value::Tensor(counts),
                Value::String("NgramLengths".to_string().into()),
                Value::Num(2.0),
            ])
            .expect("bag"),
        );
        assert_eq!(
            filtered.properties().get("NumNgrams"),
            Some(&Value::Num(2.0))
        );
        assert_eq!(
            tensor_property(&filtered, "NgramLengths").materialize_f64(),
            vec![2.0]
        );
        assert_eq!(
            tensor_property(&filtered, "Counts").materialize_f64(),
            vec![0.0, 4.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accepts_cell_unique_ngram_matrix_in_column_major_order() {
        let cells = vec![
            Value::String("a".into()),
            Value::String("c".into()),
            Value::String("e".into()),
            Value::String("b".into()),
            Value::String("d".into()),
            Value::String("f".into()),
        ];
        let ngrams = CellArray::new(cells, 3, 2).unwrap();
        let counts = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let bag = object(run(vec![Value::Cell(ngrams), Value::Tensor(counts)]).expect("bag"));
        let ngrams = string_array_property(&bag, "Ngrams");
        assert_eq!(
            ngrams.data(),
            vec!["a", "c", "e", "b", "d", "f"]
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_tokenized_column_word_vectors() {
        let err = run(vec![Value::StringArray(
            StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap(),
        )])
        .expect_err("expected column word vector rejection");
        assert!(
            err.to_string().contains("row word vector"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reports_bag_of_ngrams_identifier() {
        let err = run(vec![
            Value::String("a b c".to_string().into()),
            Value::String("NgramLengths".to_string().into()),
            Value::Num(0.0),
        ])
        .expect_err("expected bad length rejection");
        assert_eq!(
            err.identifier.as_deref(),
            Some("RunMat:bagOfNgrams:InvalidInput")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn drops_missing_unique_ngram_and_count_column() {
        for element in [runmat_value::StringElement::Missing, "<missing>".into()] {
            let missing = element.is_missing();
            let ngrams = StringArray::from_elements_unbounded(
                vec!["a".into(), element, "b".into(), "ignored".into()],
                vec![2, 2],
            )
            .unwrap();
            let original = vec![4.0, 9.0, 5.0, 9.0];
            let counts = Tensor::new(original.clone(), vec![2, 2]).unwrap();
            let bag =
                object(run(vec![Value::StringArray(ngrams), Value::Tensor(counts)]).expect("bag"));
            let columns = if missing { 1 } else { 2 };
            assert_eq!(
                bag.properties().get("NumNgrams"),
                Some(&Value::Num(columns as f64))
            );
            let counts = tensor_property(&bag, "Counts");
            assert_eq!(counts.shape(), [2, columns]);
            assert_eq!(
                counts.materialize_f64(),
                if missing { vec![4.0, 9.0] } else { original }
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_duplicate_unique_ngrams_and_bad_lengths() {
        let ngrams = StringArray::new(
            vec!["a".into(), "a".into(), "b".into(), "b".into()],
            vec![2, 2],
        )
        .unwrap();
        let counts = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let err = run(vec![Value::StringArray(ngrams), Value::Tensor(counts)])
            .expect_err("expected duplicate ngram rejection");
        assert!(err.to_string().contains("duplicate"));

        let err = run(vec![
            Value::String("a b c".to_string().into()),
            Value::String("NgramLengths".to_string().into()),
            Value::Num(0.0),
        ])
        .expect_err("expected bad length rejection");
        assert!(err.to_string().contains("positive integers"));
    }
}
