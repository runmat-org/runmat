//! Cosine similarity helpers for Text Analytics workflows.

use std::collections::{BTreeMap, HashMap, HashSet};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, ObjectInstance, SparseTensor, Tensor, Value};

use crate::builtins::common::tensor::{tensor_into_values_f64, tensor_values_f64_cow};
use crate::builtins::strings::common::is_missing_string;
use crate::builtins::strings::text_analytics::documents::{
    checked_count_len, counts_from_bag, documents_from_object, vocabulary_from_bag,
    words_from_word_vector, BAG_OF_WORDS_CLASS, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::ngrams::BAG_OF_NGRAMS_CLASS;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

const NAME: &str = "cosineSimilarity";
pub const COSINE_SIMILARITY_INTEGER_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cosine-similarity-integer-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cosineSimilarity with typed-integer matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CosineSimilarityIntegerMatrixExtension"),
    };
pub const COSINE_SIMILARITY_SINGLE_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cosine-similarity-single-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cosineSimilarity with single matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CosineSimilaritySingleMatrixExtension"),
    };
pub const COSINE_SIMILARITY_LOGICAL_MATRIX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cosine-similarity-logical-matrix",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cosineSimilarity with logical matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CosineSimilarityLogicalMatrixExtension"),
    };
pub const COSINE_SIMILARITY_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cosine-similarity-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cosineSimilarity with resident matrix input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CosineSimilarityResidentInputExtension"),
    };
pub const COSINE_SIMILARITY_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    COSINE_SIMILARITY_INTEGER_MATRIX_EXTENSION,
    COSINE_SIMILARITY_SINGLE_MATRIX_EXTENSION,
    COSINE_SIMILARITY_LOGICAL_MATRIX_EXTENSION,
    COSINE_SIMILARITY_RESIDENT_INPUT_EXTENSION,
];
const COSINE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "M1", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Real or typed-complex integer matrix values must be exactly representable at the binary64 norm and inner-product boundary." },
    BuiltinIntegerInputCapability { name: "M2", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The optional second matrix follows the same exact binary64 boundary and column-count rule." },
];
pub const COSINE_SIMILARITY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "similarities = cosineSimilarity(integer_M1, integer_M2?)", inputs: &COSINE_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat mode validates authoritative integer components before double norm/dot evaluation; real output is host sparse double, while complex output remains a dense complex-double representation gap." }];

const OUT_SIMILARITIES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "similarities",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pairwise cosine-similarity matrix.",
}];

const IN_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documentsOrBagOrMatrix",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object, bag model, word vector, or numeric matrix.",
}];

const IN_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documentsOrBagOrMatrix",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object, bag model, word vector, or numeric matrix.",
    },
    BuiltinParamDescriptor {
        name: "queriesOrMatrix",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query tokenizedDocument object, word vector, or numeric matrix.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSINE_SIMILARITY.INVALID_INPUT",
    identifier: Some("RunMat:cosineSimilarity:InvalidInput"),
    when: "Inputs do not match a supported cosineSimilarity form.",
    message: "cosineSimilarity: invalid input",
};

const ERROR_DIMENSION_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSINE_SIMILARITY.DIMENSION_MISMATCH",
    identifier: Some("RunMat:cosineSimilarity:DimensionMismatch"),
    when: "Matrix inputs have different numbers of columns.",
    message: "cosineSimilarity: matrix dimensions are not compatible",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_DIMENSION_MISMATCH];

pub const COSINE_SIMILARITY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(documents)",
            inputs: &IN_ONE,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(documents, queries)",
            inputs: &IN_TWO,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(bag)",
            inputs: &IN_ONE,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(bag, queries)",
            inputs: &IN_TWO,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(M)",
            inputs: &IN_ONE,
            outputs: &OUT_SIMILARITIES,
        },
        BuiltinSignatureDescriptor {
            label: "similarities = cosineSimilarity(M1, M2)",
            inputs: &IN_TWO,
            outputs: &OUT_SIMILARITIES,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn similarity_error(message: impl Into<String>) -> crate::RuntimeError {
    descriptor_error(message, &ERROR_INVALID_INPUT)
}

fn dimension_error(message: impl Into<String>) -> crate::RuntimeError {
    descriptor_error(message, &ERROR_DIMENSION_MISMATCH)
}

fn descriptor_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "cosineSimilarity",
    category = "strings/text_analytics",
    summary = "Compute cosine similarity between documents, bag models, or numeric matrix rows.",
    keywords = "cosineSimilarity,text analytics,bagOfWords,bagOfNgrams,tf-idf,similarity",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::similarity::COSINE_SIMILARITY_DESCRIPTOR),
    extensions(COSINE_SIMILARITY_EXTENSIONS),
    integer_capabilities(COSINE_SIMILARITY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::strings::text_analytics::similarity"
)]
async fn cosine_similarity_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !(1..=2).contains(&args.len()) {
        return Err(similarity_error(
            "cosineSimilarity: expected one or two input arguments",
        ));
    }
    ensure_extensions(&args)?;
    let args = gather_args(args).await?;
    for arg in &args {
        ensure_integer_exact(arg)?;
    }
    match args.as_slice() {
        [single] => cosine_similarity_one(single),
        [lhs, rhs] => cosine_similarity_two(lhs, rhs),
        _ => Err(similarity_error(
            "cosineSimilarity: expected one or two input arguments",
        )),
    }
}

fn ensure_extensions(args: &[Value]) -> BuiltinResult<()> {
    for value in args {
        if value_has_integer(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COSINE_SIMILARITY_INTEGER_MATRIX_EXTENSION,
                NAME,
            )?;
        }
        if value_is_single(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COSINE_SIMILARITY_SINGLE_MATRIX_EXTENSION,
                NAME,
            )?;
        }
        if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
            || matches!(value, Value::SparseTensor(s) if s.numeric_dtype().is_none())
            || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_is_logical(h))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COSINE_SIMILARITY_LOGICAL_MATRIX_EXTENSION,
                NAME,
            )?;
        }
        if matches!(value, Value::GpuTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COSINE_SIMILARITY_RESIDENT_INPUT_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}
fn value_has_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(t) if t.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(t) if t.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(s)
            if matches!(s.numeric_dtype(), Some(dtype) if !matches!(dtype,
                runmat_value::NumericDType::F64 | runmat_value::NumericDType::F32)))
        || matches!(value, Value::GpuTensor(h) if runmat_accelerate_api::handle_integer_type(h).is_some())
}
fn value_is_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if t.numeric_dtype() == runmat_value::NumericDType::F32)
        || matches!(value, Value::ComplexTensor(t) if t.numeric_dtype() == runmat_value::NumericDType::F32)
        || matches!(value, Value::SparseTensor(s)
            if s.numeric_dtype() == Some(runmat_value::NumericDType::F32))
        || matches!(value, Value::GpuTensor(h)
            if runmat_accelerate_api::handle_integer_type(h).is_none()
                && !runmat_accelerate_api::handle_is_logical(h)
                && runmat_accelerate_api::handle_precision(h)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32))
}
fn ensure_integer_exact(value: &Value) -> BuiltinResult<()> {
    let ok = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
    let valid = match value {
        Value::Int(v) => ok(v),
        Value::Tensor(t) => t
            .integer_storage()
            .is_none_or(|s| s.exact_values().iter().all(ok)),
        Value::ComplexTensor(t) => t.integer_storage().is_none_or(|s| {
            s.real
                .exact_values()
                .iter()
                .chain(s.imag.exact_values().iter())
                .all(ok)
        }),
        Value::SparseTensor(s) => s
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(ok)),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(similarity_error(
            "cosineSimilarity: integer matrix values must be exactly representable as double",
        ))
    }
}

async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            similarity_error(format!("cosineSimilarity: failed to gather input: {err}"))
        })?);
    }
    Ok(out)
}

fn cosine_similarity_one(value: &Value) -> BuiltinResult<Value> {
    match classify_primary(value)? {
        PrimaryInput::Real(matrix) => sparse_real_cosine(&matrix, &matrix).map(Value::SparseTensor),
        PrimaryInput::Complex(matrix) => complex_cosine(&matrix, &matrix).map(Value::ComplexTensor),
        PrimaryInput::Text(model) => {
            let weighted = model.tfidf_against_self()?;
            sparse_real_cosine(&weighted, &weighted).map(Value::SparseTensor)
        }
    }
}

fn cosine_similarity_two(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    match classify_primary(lhs)? {
        PrimaryInput::Real(lhs_matrix) => {
            if matches!(rhs, Value::Complex(..) | Value::ComplexTensor(_)) {
                let lhs_matrix = complex_matrix_input(lhs, NAME)?;
                let rhs_matrix = complex_matrix_input(rhs, NAME)?;
                complex_cosine(&lhs_matrix, &rhs_matrix).map(Value::ComplexTensor)
            } else {
                let rhs_matrix = real_matrix_input(rhs, NAME)?;
                sparse_real_cosine(&lhs_matrix, &rhs_matrix).map(Value::SparseTensor)
            }
        }
        PrimaryInput::Complex(lhs_matrix) => {
            let rhs_matrix = complex_matrix_input(rhs, NAME)?;
            complex_cosine(&lhs_matrix, &rhs_matrix).map(Value::ComplexTensor)
        }
        PrimaryInput::Text(model) => {
            let query_documents = query_documents(rhs)?;
            let lhs_weighted = model.tfidf_against_self()?;
            let rhs_counts = model.counts_for_queries(&query_documents)?;
            let rhs_weighted = model.apply_idf(rhs_counts)?;
            sparse_real_cosine(&lhs_weighted, &rhs_weighted).map(Value::SparseTensor)
        }
    }
}

enum PrimaryInput {
    Real(RealRows),
    Complex(ComplexRows),
    Text(TextModel),
}

fn classify_primary(value: &Value) -> BuiltinResult<PrimaryInput> {
    if is_text_like_input(value) {
        return text_model_input(value).map(PrimaryInput::Text);
    }
    if matches!(value, Value::Complex(..) | Value::ComplexTensor(_)) {
        return complex_matrix_input(value, NAME).map(PrimaryInput::Complex);
    }
    real_matrix_input(value, NAME).map(PrimaryInput::Real)
}

fn is_text_like_input(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_)
            | Value::StringArray(_)
            | Value::CharArray(_)
            | Value::Cell(_)
            | Value::Object(_)
    )
}

#[derive(Clone)]
struct RealRows {
    rows: usize,
    cols: usize,
    data: Vec<Vec<(usize, f64)>>,
    norms: Vec<f64>,
}

impl RealRows {
    fn new(rows: usize, cols: usize, data: Vec<Vec<(usize, f64)>>) -> BuiltinResult<Self> {
        if data.len() != rows {
            return Err(similarity_error(format!(
                "cosineSimilarity: row storage has {} rows but expected {rows}",
                data.len()
            )));
        }
        let norms = data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(_, value)| value * value)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        Ok(Self {
            rows,
            cols,
            data,
            norms,
        })
    }
}

#[derive(Clone)]
struct ComplexRows {
    rows: usize,
    cols: usize,
    data: Vec<Vec<(usize, (f64, f64))>>,
    norms: Vec<f64>,
}

impl ComplexRows {
    fn new(rows: usize, cols: usize, data: Vec<Vec<(usize, (f64, f64))>>) -> BuiltinResult<Self> {
        if data.len() != rows {
            return Err(similarity_error(format!(
                "cosineSimilarity: complex row storage has {} rows but expected {rows}",
                data.len()
            )));
        }
        let norms = data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(_, (re, im))| re * re + im * im)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        Ok(Self {
            rows,
            cols,
            data,
            norms,
        })
    }
}

fn real_matrix_input(value: &Value, fn_name: &str) -> BuiltinResult<RealRows> {
    match value {
        Value::Num(value) => RealRows::new(1, 1, nonzero_real_rows(1, 1, &[*value])),
        Value::Bool(value) => {
            let scalar = if *value { 1.0 } else { 0.0 };
            RealRows::new(1, 1, nonzero_real_rows(1, 1, &[scalar]))
        }
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let values = tensor_values_f64_cow(tensor);
            RealRows::new(rows, cols, nonzero_real_rows(rows, cols, &values))
        }
        Value::SparseTensor(sparse) => real_rows_from_sparse(sparse),
        Value::Complex(..) | Value::ComplexTensor(_) => Err(similarity_error(format!(
            "{fn_name}: complex matrix input requires both matrix inputs to be complex-compatible"
        ))),
        other => Err(similarity_error(format!(
            "{fn_name}: expected tokenizedDocument, bag model, word vector, or numeric matrix input, got {other:?}"
        ))),
    }
}

fn nonzero_real_rows(rows: usize, cols: usize, values: &[f64]) -> Vec<Vec<(usize, f64)>> {
    let mut out = vec![Vec::new(); rows];
    for col in 0..cols {
        for (row, row_values) in out.iter_mut().enumerate().take(rows) {
            let value = values[row + col * rows];
            if value != 0.0 || value.is_nan() {
                row_values.push((col, value));
            }
        }
    }
    out
}

fn real_rows_from_sparse(sparse: &SparseTensor) -> BuiltinResult<RealRows> {
    let mut rows = vec![Vec::new(); sparse.rows];
    let values = sparse.materialize_f64();
    for col in 0..sparse.cols {
        for idx in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
            let value = values[idx];
            if value != 0.0 || value.is_nan() {
                rows[sparse.row_indices[idx]].push((col, value));
            }
        }
    }
    RealRows::new(sparse.rows, sparse.cols, rows)
}

fn complex_matrix_input(value: &Value, fn_name: &str) -> BuiltinResult<ComplexRows> {
    match value {
        Value::Complex(re, im) => ComplexRows::new(1, 1, nonzero_complex_rows(1, 1, &[(*re, *im)])),
        Value::ComplexTensor(tensor) => ComplexRows::new(
            tensor.rows,
            tensor.cols,
            nonzero_complex_rows(tensor.rows, tensor.cols, &tensor.materialize_f64()),
        ),
        Value::Num(value) => ComplexRows::new(1, 1, nonzero_complex_rows(1, 1, &[(*value, 0.0)])),
        Value::Bool(value) => {
            let scalar = if *value { 1.0 } else { 0.0 };
            ComplexRows::new(1, 1, nonzero_complex_rows(1, 1, &[(scalar, 0.0)]))
        }
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let values = tensor_values_f64_cow(tensor);
            let data = values.iter().map(|value| (*value, 0.0)).collect::<Vec<_>>();
            ComplexRows::new(rows, cols, nonzero_complex_rows(rows, cols, &data))
        }
        Value::SparseTensor(sparse) => {
            let real = real_rows_from_sparse(sparse)?;
            let rows = real
                .data
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|(col, value)| (col, (value, 0.0)))
                        .collect()
                })
                .collect();
            ComplexRows::new(real.rows, real.cols, rows)
        }
        other => Err(similarity_error(format!(
            "{fn_name}: expected numeric matrix input, got {other:?}"
        ))),
    }
}

fn nonzero_complex_rows(
    rows: usize,
    cols: usize,
    values: &[(f64, f64)],
) -> Vec<Vec<(usize, (f64, f64))>> {
    let mut out = vec![Vec::new(); rows];
    for col in 0..cols {
        for (row, row_values) in out.iter_mut().enumerate().take(rows) {
            let value = values[row + col * rows];
            if value.0 != 0.0 || value.1 != 0.0 || value.0.is_nan() || value.1.is_nan() {
                row_values.push((col, value));
            }
        }
    }
    out
}

#[derive(Clone)]
struct TextModel {
    terms: Terms,
    counts: RealRows,
    idf: Vec<f64>,
}

#[derive(Clone)]
enum Terms {
    Words(Vec<String>),
    Ngrams(Vec<Vec<String>>),
}

impl TextModel {
    fn from_counts(terms: Terms, counts: Tensor, fn_name: &str) -> BuiltinResult<Self> {
        let term_len = terms.len();
        let rows = counts.rows();
        let cols = counts.cols();
        if cols != term_len {
            return Err(similarity_error(format!(
                "{fn_name}: count matrix columns ({}) must match term count ({term_len})",
                cols
            )));
        }
        let values = tensor_into_values_f64(counts);
        let real_rows = RealRows::new(rows, cols, nonzero_real_rows(rows, cols, &values))?;
        let idf = idf_from_counts(&values, rows, cols);
        Ok(Self {
            terms,
            counts: real_rows,
            idf,
        })
    }

    fn tfidf_against_self(&self) -> BuiltinResult<RealRows> {
        self.apply_idf(self.counts.clone())
    }

    fn apply_idf(&self, mut counts: RealRows) -> BuiltinResult<RealRows> {
        for row in &mut counts.data {
            for (col, value) in &mut *row {
                *value *= self.idf[*col];
            }
            row.retain(|(_, value)| *value != 0.0 || value.is_nan());
        }
        RealRows::new(counts.rows, counts.cols, counts.data)
    }

    fn counts_for_queries(&self, queries: &[Vec<String>]) -> BuiltinResult<RealRows> {
        match &self.terms {
            Terms::Words(words) => counts_for_word_queries(words, queries),
            Terms::Ngrams(ngrams) => counts_for_ngram_queries(ngrams, queries),
        }
    }
}

impl Terms {
    fn len(&self) -> usize {
        match self {
            Self::Words(words) => words.len(),
            Self::Ngrams(ngrams) => ngrams.len(),
        }
    }
}

fn idf_from_counts(counts: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    if rows == 0 {
        return vec![0.0; cols];
    }
    let row_count = rows as f64;
    (0..cols)
        .map(|col| {
            let mut df = 0usize;
            for row in 0..rows {
                if counts[row + col * rows] > 0.0 {
                    df += 1;
                }
            }
            if df == 0 {
                0.0
            } else {
                (row_count / df as f64).ln()
            }
        })
        .collect()
}

fn text_model_input(value: &Value) -> BuiltinResult<TextModel> {
    match value {
        Value::Object(object) if object.class_name == TOKENIZED_DOCUMENT_CLASS => {
            let documents = documents_from_object(object, NAME)?;
            text_model_from_documents(&documents)
        }
        Value::Object(object) if object.class_name == BAG_OF_WORDS_CLASS => TextModel::from_counts(
            Terms::Words(vocabulary_from_bag(object, NAME)?),
            counts_from_bag(object, NAME)?,
            NAME,
        ),
        Value::Object(object) if object.class_name == BAG_OF_NGRAMS_CLASS => {
            TextModel::from_counts(
                Terms::Ngrams(ngrams_from_bag(object)?),
                counts_from_ngram_bag(object)?,
                NAME,
            )
        }
        other => {
            let words = row_word_vector(other)?;
            text_model_from_documents(&[words])
        }
    }
}

fn text_model_from_documents(documents: &[Vec<String>]) -> BuiltinResult<TextModel> {
    let mut vocabulary = Vec::new();
    let mut positions = BTreeMap::new();
    for token in documents.iter().flatten() {
        if !positions.contains_key(token) {
            positions.insert(token.clone(), vocabulary.len());
            vocabulary.push(token.clone());
        }
    }
    let rows = documents.len();
    let cols = vocabulary.len();
    let mut counts = vec![0.0; checked_count_len(rows, cols, NAME)?];
    for (row, document) in documents.iter().enumerate() {
        for token in document {
            if let Some(col) = positions.get(token) {
                counts[row + col * rows] += 1.0;
            }
        }
    }
    let tensor = Tensor::new(counts, vec![rows, cols]).map_err(similarity_error)?;
    TextModel::from_counts(Terms::Words(vocabulary), tensor, NAME)
}

fn query_documents(value: &Value) -> BuiltinResult<Vec<Vec<String>>> {
    match value {
        Value::Object(object) if object.class_name == TOKENIZED_DOCUMENT_CLASS => {
            documents_from_object(object, NAME)
        }
        Value::Object(object) => Err(similarity_error(format!(
            "cosineSimilarity: query input must be tokenizedDocument or a row word vector, got {}",
            object.class_name
        ))),
        other => Ok(vec![row_word_vector(other)?]),
    }
}

fn row_word_vector(value: &Value) -> BuiltinResult<Vec<String>> {
    validate_row_word_vector(value)?;
    words_from_word_vector(value, NAME)
}

fn validate_row_word_vector(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::String(_) | Value::Bool(_) | Value::Num(_) => Ok(()),
        Value::StringArray(array) if array.rows <= 1 => Ok(()),
        Value::CharArray(array) if array.rows <= 1 => Ok(()),
        Value::Cell(cell) if cell.rows <= 1 => Ok(()),
        Value::CharArray(array) => Err(similarity_error(format!(
            "cosineSimilarity: non-tokenized text input must be a row word vector; got char array with shape {}x{}",
            array.rows, array.cols
        ))),
        Value::StringArray(array) => Err(similarity_error(format!(
            "cosineSimilarity: non-tokenized text input must be a row word vector; got string array with shape {}x{}",
            array.rows, array.cols
        ))),
        Value::Cell(cell) => Err(similarity_error(format!(
            "cosineSimilarity: non-tokenized text input must be a row word vector; got cell array with shape {}x{}",
            cell.rows, cell.cols
        ))),
        _ => Ok(()),
    }
}

fn counts_for_word_queries(
    vocabulary: &[String],
    queries: &[Vec<String>],
) -> BuiltinResult<RealRows> {
    let positions = vocabulary
        .iter()
        .enumerate()
        .map(|(idx, word)| (word.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut rows = vec![Vec::<(usize, f64)>::new(); queries.len()];
    for (row, query) in queries.iter().enumerate() {
        let mut counts = BTreeMap::<usize, f64>::new();
        for word in query {
            if let Some(&col) = positions.get(word.as_str()) {
                *counts.entry(col).or_default() += 1.0;
            }
        }
        rows[row] = counts.into_iter().collect();
    }
    RealRows::new(queries.len(), vocabulary.len(), rows)
}

fn counts_for_ngram_queries(
    ngrams: &[Vec<String>],
    queries: &[Vec<String>],
) -> BuiltinResult<RealRows> {
    let positions = ngrams
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, ngram)| (ngram, idx))
        .collect::<HashMap<_, _>>();
    let lengths = ngrams
        .iter()
        .map(Vec::len)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut rows = vec![Vec::<(usize, f64)>::new(); queries.len()];
    for (row, query) in queries.iter().enumerate() {
        let mut counts = BTreeMap::<usize, f64>::new();
        for &length in &lengths {
            if length == 0 || length > query.len() {
                continue;
            }
            for start in 0..=query.len() - length {
                if let Some(&col) = positions.get(&query[start..start + length]) {
                    *counts.entry(col).or_default() += 1.0;
                }
            }
        }
        rows[row] = counts.into_iter().collect();
    }
    RealRows::new(queries.len(), ngrams.len(), rows)
}

fn counts_from_ngram_bag(object: &ObjectInstance) -> BuiltinResult<Tensor> {
    match object.properties.get("Counts") {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        _ => Err(similarity_error(
            "cosineSimilarity: bagOfNgrams object missing Counts property",
        )),
    }
}

fn ngrams_from_bag(object: &ObjectInstance) -> BuiltinResult<Vec<Vec<String>>> {
    let value = object.properties.get("Ngrams").ok_or_else(|| {
        similarity_error("cosineSimilarity: bagOfNgrams object missing Ngrams property")
    })?;
    let Value::StringArray(array) = value else {
        return Err(similarity_error(format!(
            "cosineSimilarity: bagOfNgrams Ngrams property must be a string array, got {value:?}"
        )));
    };
    let mut out = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let mut ngram = Vec::new();
        for col in 0..array.cols {
            let word = array.data[row + col * array.rows].clone();
            if !word.is_empty() && !is_missing_string(&word) {
                ngram.push(word);
            }
        }
        if ngram.is_empty() {
            return Err(similarity_error(
                "cosineSimilarity: bagOfNgrams object contains an empty n-gram",
            ));
        }
        out.push(ngram);
    }
    Ok(out)
}

fn sparse_real_cosine(lhs: &RealRows, rhs: &RealRows) -> BuiltinResult<SparseTensor> {
    if lhs.cols != rhs.cols {
        return Err(dimension_error(format!(
            "cosineSimilarity: matrix inputs must have the same number of columns, got {} and {}",
            lhs.cols, rhs.cols
        )));
    }
    let mut col_ptrs = Vec::with_capacity(rhs.rows + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for rhs_row in 0..rhs.rows {
        for lhs_row in 0..lhs.rows {
            let value = real_cosine_value(lhs, lhs_row, rhs, rhs_row);
            if value != 0.0 || value.is_nan() {
                row_indices.push(lhs_row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    SparseTensor::new(lhs.rows, rhs.rows, col_ptrs, row_indices, values)
        .map_err(|err| similarity_error(format!("cosineSimilarity: {err}")))
}

fn real_cosine_value(lhs: &RealRows, lhs_row: usize, rhs: &RealRows, rhs_row: usize) -> f64 {
    let denom = lhs.norms[lhs_row] * rhs.norms[rhs_row];
    if denom == 0.0 {
        return f64::NAN;
    }
    real_dot(&lhs.data[lhs_row], &rhs.data[rhs_row]) / denom
}

fn real_dot(lhs: &[(usize, f64)], rhs: &[(usize, f64)]) -> f64 {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut dot = 0.0;
    while i < lhs.len() && j < rhs.len() {
        match lhs[i].0.cmp(&rhs[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += lhs[i].1 * rhs[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    dot
}

fn complex_cosine(lhs: &ComplexRows, rhs: &ComplexRows) -> BuiltinResult<ComplexTensor> {
    if lhs.cols != rhs.cols {
        return Err(dimension_error(format!(
            "cosineSimilarity: matrix inputs must have the same number of columns, got {} and {}",
            lhs.cols, rhs.cols
        )));
    }
    let mut data = Vec::with_capacity(lhs.rows * rhs.rows);
    for col in 0..rhs.rows {
        for row in 0..lhs.rows {
            data.push(complex_cosine_value(lhs, row, rhs, col));
        }
    }
    ComplexTensor::new(data, vec![lhs.rows, rhs.rows])
        .map_err(|err| similarity_error(format!("cosineSimilarity: {err}")))
}

fn complex_cosine_value(
    lhs: &ComplexRows,
    lhs_row: usize,
    rhs: &ComplexRows,
    rhs_row: usize,
) -> (f64, f64) {
    let denom = lhs.norms[lhs_row] * rhs.norms[rhs_row];
    if denom == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let (re, im) = complex_dot_conj(&lhs.data[lhs_row], &rhs.data[rhs_row]);
    (re / denom, im / denom)
}

fn complex_dot_conj(lhs: &[(usize, (f64, f64))], rhs: &[(usize, (f64, f64))]) -> (f64, f64) {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut re = 0.0;
    let mut im = 0.0;
    while i < lhs.len() && j < rhs.len() {
        match lhs[i].0.cmp(&rhs[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let (ar, ai) = lhs[i].1;
                let (br, bi) = rhs[j].1;
                re += ar * br + ai * bi;
                im += ar * bi - ai * br;
                i += 1;
                j += 1;
            }
        }
    }
    (re, im)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{
        CellArray, IntegerComplexStorage, IntegerStorage, ObjectInstance, StringArray,
    };

    fn dense_sparse(value: Value) -> Tensor {
        match value {
            Value::SparseTensor(sparse) => sparse.to_dense().expect("dense sparse"),
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).expect("tensor"))
    }

    #[tokio::test]
    async fn integer_gate_all_classes_boundary_and_single_gate() {
        assert_eq!(
            COSINE_SIMILARITY_INTEGER_CAPABILITIES[0].inputs[0]
                .classes
                .len(),
            8
        );
        assert_eq!(
            COSINE_SIMILARITY_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::Double
        );
        let integer = || {
            Value::Tensor(Tensor::new_integer(IntegerStorage::I8(vec![1, 0]), vec![1, 2]).unwrap())
        };
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        assert!(cosine_similarity_builtin(vec![integer()]).await.is_err());
        assert!(cosine_similarity_builtin(vec![Value::Tensor(
            Tensor::from_f32(vec![1.0, 0.0], vec![1, 2]).unwrap(),
        )])
        .await
        .is_err());
        drop(_strict);

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in [
            IntegerStorage::I8(vec![1, 0]),
            IntegerStorage::I16(vec![1, 0]),
            IntegerStorage::I32(vec![1, 0]),
            IntegerStorage::I64(vec![1, 0]),
            IntegerStorage::U8(vec![1, 0]),
            IntegerStorage::U16(vec![1, 0]),
            IntegerStorage::U32(vec![1, 0]),
            IntegerStorage::U64(vec![1, 0]),
        ] {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
            assert!(matches!(
                cosine_similarity_builtin(vec![value]).await.unwrap(),
                Value::SparseTensor(_)
            ));
        }
        let wide = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1, 0]), vec![1, 2])
                .unwrap(),
        );
        assert!(cosine_similarity_builtin(vec![wide]).await.is_err());
        let exact_wide = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54, 0]), vec![1, 2]).unwrap(),
        );
        assert!(cosine_similarity_builtin(vec![exact_wide]).await.is_ok());
        assert!(matches!(
            cosine_similarity_builtin(vec![Value::Tensor(
                Tensor::from_f32(vec![1.0, 0.0], vec![1, 2]).unwrap(),
            )])
            .await
            .unwrap(),
            Value::SparseTensor(_)
        ));
    }

    #[tokio::test]
    async fn sparse_numeric_and_logical_inputs_use_their_matching_extension_gates() {
        let cases = [
            (
                Value::SparseTensor(
                    SparseTensor::new_integer(
                        1,
                        2,
                        vec![0, 1, 1],
                        vec![0],
                        IntegerStorage::I8(vec![1]),
                    )
                    .expect("sparse integer"),
                ),
                COSINE_SIMILARITY_INTEGER_MATRIX_EXTENSION.error_identifier,
            ),
            (
                Value::SparseTensor(
                    SparseTensor::new_f32(1, 2, vec![0, 1, 1], vec![0], vec![1.0])
                        .expect("sparse single"),
                ),
                COSINE_SIMILARITY_SINGLE_MATRIX_EXTENSION.error_identifier,
            ),
            (
                Value::SparseTensor(
                    SparseTensor::new_logical(1, 2, vec![0, 1, 1], vec![0])
                        .expect("sparse logical"),
                ),
                COSINE_SIMILARITY_LOGICAL_MATRIX_EXTENSION.error_identifier,
            ),
        ];
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for (input, expected_identifier) in cases {
            let error = cosine_similarity_builtin(vec![input])
                .await
                .expect_err("MATLAB mode rejects sparse extension input");
            assert_eq!(error.identifier(), expected_identifier);
        }
    }

    #[tokio::test]
    async fn sparse_integer_storage_is_gated_exactly_before_double_computation() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ] {
            let sparse = SparseTensor::new_integer(1, 2, vec![0, 1, 1], vec![0], storage)
                .expect("typed sparse integer");
            assert!(matches!(
                cosine_similarity_builtin(vec![Value::SparseTensor(sparse)])
                    .await
                    .expect("exact sparse integer"),
                Value::SparseTensor(_)
            ));
        }

        let exact = SparseTensor::new_integer(
            1,
            2,
            vec![0, 1, 1],
            vec![0],
            IntegerStorage::U64(vec![1_u64 << 54]),
        )
        .expect("exact wide sparse integer");
        assert!(cosine_similarity_builtin(vec![Value::SparseTensor(exact)])
            .await
            .is_ok());

        let lossy = SparseTensor::new_integer(
            1,
            2,
            vec![0, 1, 1],
            vec![0],
            IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
        )
        .expect("lossy sparse integer");
        let error = cosine_similarity_builtin(vec![Value::SparseTensor(lossy)])
            .await
            .expect_err("lossy sparse integer must not enter the double domain");
        assert!(error.message().contains("exactly representable"));
    }

    #[tokio::test]
    async fn sparse_logical_zero_norm_rows_preserve_nan_cosine_semantics() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let sparse = SparseTensor::new_logical(2, 2, vec![0, 1, 1], vec![0])
            .expect("sparse logical with a zero row");
        let out = dense_sparse(
            cosine_similarity_builtin(vec![Value::SparseTensor(sparse)])
                .await
                .expect("sparse logical cosine"),
        );
        let values = out.materialize_f64();
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan());
        assert!(values[2].is_nan());
        assert!(values[3].is_nan());
    }

    fn typed_complex_integer(real: Vec<i64>, imag: Vec<i64>) -> Value {
        let storage =
            IntegerComplexStorage::new(IntegerStorage::I64(real), IntegerStorage::I64(imag))
                .expect("typed complex integer storage");
        Value::ComplexTensor(
            ComplexTensor::new_integer(storage, vec![1, 2]).expect("typed complex tensor"),
        )
    }

    #[tokio::test]
    async fn typed_complex_integer_roles_are_gated_and_return_complex_double() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for args in [
            vec![typed_complex_integer(vec![1, 0], vec![1, 0])],
            vec![
                tensor(vec![1.0, 0.0], 1, 2),
                typed_complex_integer(vec![1, 0], vec![1, 0]),
            ],
        ] {
            let error = cosine_similarity_builtin(args)
                .await
                .expect_err("MATLAB mode rejects typed complex integer matrices");
            assert_eq!(
                error.identifier(),
                COSINE_SIMILARITY_INTEGER_MATRIX_EXTENSION.error_identifier
            );
        }
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for args in [
            vec![typed_complex_integer(vec![1, 0], vec![1, 0])],
            vec![
                typed_complex_integer(vec![1, 0], vec![1, 0]),
                tensor(vec![1.0, 0.0], 1, 2),
            ],
            vec![
                tensor(vec![1.0, 0.0], 1, 2),
                typed_complex_integer(vec![1, 0], vec![1, 0]),
            ],
        ] {
            assert!(matches!(
                cosine_similarity_builtin(args).await.unwrap(),
                Value::ComplexTensor(_)
            ));
        }

        let lossy = typed_complex_integer(vec![(1_i64 << 53) + 1, 0], vec![0, 0]);
        assert!(cosine_similarity_builtin(vec![lossy]).await.is_err());
    }

    #[tokio::test]
    async fn arity_precedes_extension_gates() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer =
            || Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap());
        let error = cosine_similarity_builtin(vec![integer(), integer(), integer()])
            .await
            .expect_err("arity must be checked first");
        assert_eq!(
            error.identifier(),
            Some("RunMat:cosineSimilarity:InvalidInput")
        );
        assert!(error.message().contains("one or two"));
    }

    fn tokenized(docs: Vec<Vec<&str>>) -> Value {
        let rows = docs.len();
        let values = docs
            .into_iter()
            .map(|doc| {
                let len = doc.len();
                Value::StringArray(
                    StringArray::new(
                        doc.into_iter().map(str::to_string).collect::<Vec<_>>(),
                        vec![1, len],
                    )
                    .expect("strings"),
                )
            })
            .collect::<Vec<_>>();
        let mut object = ObjectInstance::new(TOKENIZED_DOCUMENT_CLASS.to_string());
        object.properties.insert(
            "Documents".to_string(),
            Value::Cell(CellArray::new(values, rows, 1).expect("cell")),
        );
        Value::Object(object)
    }

    #[tokio::test]
    async fn numeric_matrix_rows_return_sparse_cosines() {
        let m = tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let out = dense_sparse(cosine_similarity_builtin(vec![m]).await.expect("cosine"));
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[tokio::test]
    async fn native_single_and_integer_matrix_extensions_use_explicit_double_domain() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let inputs = [
            Value::Tensor(
                Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("single matrix"),
            ),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(vec![1, 0, 0, 1]), vec![2, 2])
                    .expect("integer matrix"),
            ),
        ];
        for input in inputs {
            let out = dense_sparse(
                cosine_similarity_builtin(vec![input])
                    .await
                    .expect("extended numeric cosine"),
            );
            assert_eq!(out.shape, vec![2, 2]);
            assert_eq!(out.materialize_f64(), vec![1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[tokio::test]
    async fn numeric_two_matrix_form_uses_rows() {
        let lhs = tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let rhs = tensor(vec![1.0, 1.0], 1, 2);
        let out = dense_sparse(
            cosine_similarity_builtin(vec![lhs, rhs])
                .await
                .expect("cosine"),
        );
        let inv = 1.0 / 2.0_f64.sqrt();
        assert_eq!(out.shape, vec![2, 1]);
        let values = out.materialize_f64();
        assert!((values[0] - inv).abs() < 1e-12);
        assert!((values[1] - inv).abs() < 1e-12);
    }

    #[tokio::test]
    async fn tokenized_documents_use_tfidf_sparse_output() {
        let docs = tokenized(vec![
            vec!["alpha", "shared"],
            vec!["beta", "shared"],
            vec!["alpha", "unique"],
        ]);
        let out = dense_sparse(cosine_similarity_builtin(vec![docs]).await.expect("cosine"));
        assert_eq!(out.shape, vec![3, 3]);
        let values = out.materialize_f64();
        assert!((values[0] - 1.0).abs() < 1e-12);
        assert!(values[1] > 0.0 && values[1] < 1.0);
        assert!(values[2] > 0.0 && values[2] < 1.0);
    }

    #[tokio::test]
    async fn bag_query_form_uses_bag_vocabulary() {
        let words = StringArray::new(vec!["alpha".to_string(), "beta".to_string()], vec![1, 2])
            .expect("words");
        let counts = Tensor::new(vec![2.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("counts");
        let mut object = ObjectInstance::new(BAG_OF_WORDS_CLASS.to_string());
        object
            .properties
            .insert("Vocabulary".to_string(), Value::StringArray(words));
        object
            .properties
            .insert("Counts".to_string(), Value::Tensor(counts));
        let bag = Value::Object(object);
        let query = Value::StringArray(
            StringArray::new(vec!["alpha".to_string()], vec![1, 1]).expect("query"),
        );
        let out = dense_sparse(
            cosine_similarity_builtin(vec![bag, query])
                .await
                .expect("cosine"),
        );
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.materialize_f64(), vec![1.0, 0.0]);
    }

    #[tokio::test]
    async fn rejects_bag_as_query_argument() {
        let words = StringArray::new(vec!["alpha".to_string()], vec![1, 1]).expect("words");
        let counts = Tensor::new(vec![1.0], vec![1, 1]).expect("counts");
        let mut object = ObjectInstance::new(BAG_OF_WORDS_CLASS.to_string());
        object
            .properties
            .insert("Vocabulary".to_string(), Value::StringArray(words));
        object
            .properties
            .insert("Counts".to_string(), Value::Tensor(counts));
        let bag = Value::Object(object);
        let err = cosine_similarity_builtin(vec![bag.clone(), bag])
            .await
            .expect_err("bag query should be rejected");
        assert!(err.message().contains("query input"));
    }

    #[tokio::test]
    async fn sparse_matrix_input_stays_sparse_output() {
        let sparse =
            SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 4.0]).expect("sparse");
        let out = cosine_similarity_builtin(vec![Value::SparseTensor(sparse)])
            .await
            .expect("cosine");
        match out {
            Value::SparseTensor(sparse) => {
                assert_eq!(sparse.nnz(), 2);
                assert_eq!(
                    sparse.to_dense().unwrap().materialize_f64(),
                    vec![1.0, 0.0, 0.0, 1.0]
                );
            }
            other => panic!("expected sparse result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn complex_matrix_uses_hermitian_dot_and_dense_complex_output() {
        let lhs =
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 1.0)], vec![1, 1]).expect("lhs"));
        let rhs =
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).expect("rhs"));
        let out = cosine_similarity_builtin(vec![lhs, rhs])
            .await
            .expect("cosine");
        let Value::ComplexTensor(out) = out else {
            panic!("expected complex tensor");
        };
        assert_eq!(out.shape, vec![1, 1]);
        let inv = 1.0 / 2.0_f64.sqrt();
        assert!((out.materialize_f64()[0].0 - inv).abs() < 1e-12);
        assert!((out.materialize_f64()[0].1 + inv).abs() < 1e-12);
    }

    #[tokio::test]
    async fn mixed_real_complex_matrix_roles_are_both_supported() {
        let real = tensor(vec![1.0, 0.0], 1, 2);
        let complex = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 1.0), (0.0, 0.0)], vec![1, 2]).unwrap(),
        );
        let Value::ComplexTensor(real_complex) =
            cosine_similarity_builtin(vec![real.clone(), complex.clone()])
                .await
                .expect("real then complex")
        else {
            panic!("expected complex output")
        };
        let Value::ComplexTensor(complex_real) = cosine_similarity_builtin(vec![complex, real])
            .await
            .expect("complex then real")
        else {
            panic!("expected complex output")
        };
        let forward = real_complex.materialize_f64()[0];
        let reverse = complex_real.materialize_f64()[0];
        assert!((forward.0 - reverse.0).abs() < 1e-12);
        assert!((forward.1 + reverse.1).abs() < 1e-12);
    }

    #[tokio::test]
    async fn zero_norm_rows_return_nan_as_mathematically_undefined() {
        let out = dense_sparse(
            cosine_similarity_builtin(vec![tensor(vec![0.0, 1.0], 2, 1)])
                .await
                .expect("zero norm behavior"),
        );
        let values = out.materialize_f64();
        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert!(values[2].is_nan());
        assert_eq!(values[3], 1.0);

        let zero = Value::ComplexTensor(ComplexTensor::zeros(vec![1, 2]));
        let Value::ComplexTensor(out) = cosine_similarity_builtin(vec![zero])
            .await
            .expect("complex zero norm behavior")
        else {
            panic!("expected complex output")
        };
        assert!(out.materialize_f64()[0].0.is_nan());
        assert!(out.materialize_f64()[0].1.is_nan());
    }

    #[tokio::test]
    async fn rejects_mismatched_matrix_columns() {
        let lhs = tensor(vec![1.0, 2.0], 1, 2);
        let rhs = tensor(vec![1.0, 2.0, 3.0], 1, 3);
        let err = cosine_similarity_builtin(vec![lhs, rhs])
            .await
            .expect_err("dimension mismatch");
        assert!(err.message().contains("same number of columns"));
    }

    #[tokio::test]
    async fn rejects_non_row_word_vector() {
        let words = Value::StringArray(
            StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).expect("words"),
        );
        let err = cosine_similarity_builtin(vec![words])
            .await
            .expect_err("non-row");
        assert!(err.message().contains("row word vector"));
    }

    #[test]
    fn char_row_to_string_helper_is_not_dead_code() {
        let array = runmat_value::CharArray::new_row("alpha beta");
        assert_eq!(
            crate::builtins::strings::common::char_row_to_string_slice(&array.data, array.cols, 0),
            "alpha beta"
        );
    }
}
