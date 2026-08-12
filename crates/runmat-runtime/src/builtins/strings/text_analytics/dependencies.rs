//! Dependency-detail helpers for Text Analytics tokenized documents.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, ObjectInstance, StringArray, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::text_analytics::details::add_sentence_details_builtin;
use crate::builtins::strings::text_analytics::documents::{
    document_token_type_with_options, documents_from_object, options_from_document_object,
    text_analytics_error, tokenized_document_language, DocumentTokenType, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::pos::{part_of_speech_for_token, PosLanguage};
use crate::{gather_if_needed_async, BuiltinResult};

pub(in crate::builtins::strings::text_analytics) const HEAD_DETAILS_PROPERTY: &str = "HeadDetails";
pub(in crate::builtins::strings::text_analytics) const DEPENDENCY_DETAILS_PROPERTY: &str =
    "DependencyDetails";

const OUT_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "updatedDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated tokenized document object.",
}];

const IN_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.ADD_DEPENDENCY_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:addDependencyDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object.",
    message: "addDependencyDetails: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const ADD_DEPENDENCY_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "updatedDocuments = addDependencyDetails(documents)",
        inputs: &IN_DOCUMENTS,
        outputs: &OUT_DOCUMENTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const ADD_DEPENDENCY_DETAILS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "addDependencyDetails accepts and returns tokenizedDocument objects only; it has no numeric arguments, and the Head token indices exposed through tokenDetails are double metadata rather than integer-class results.",
    };

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

#[runtime_builtin(
    name = "addDependencyDetails",
    category = "strings/text_analytics",
    summary = "Add dependency parse details to tokenizedDocument objects.",
    keywords = "addDependencyDetails,text analytics,tokenizedDocument,dependency,head,universal dependencies",
    accel = "sink",
    type_resolver(any_type),
    descriptor(
        crate::builtins::strings::text_analytics::dependencies::ADD_DEPENDENCY_DETAILS_DESCRIPTOR
    ),
    integer_audit(
        crate::builtins::strings::text_analytics::dependencies::ADD_DEPENDENCY_DETAILS_INTEGER_AUDIT
    ),
    builtin_path = "crate::builtins::strings::text_analytics::dependencies"
)]
async fn add_dependency_details_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args).await?;
    let documents = parse_args(gathered)?;
    let mut object = tokenized_document_object(documents)?;
    let language = PosLanguage::from_document_for(
        &tokenized_document_language(&object),
        "addDependencyDetails",
    )?;

    if !object.properties.contains_key("SentenceNumbers") {
        object = add_sentence_details_for_dependencies(object).await?;
    }

    let documents = documents_from_object(&object, "addDependencyDetails")?;
    let sentence_numbers = sentence_numbers_from_object(&object, "addDependencyDetails")?;
    let document_options = options_from_document_object(&object);
    let (heads, dependencies) = dependency_details_for_documents(
        &documents,
        sentence_numbers.as_deref(),
        language,
        &document_options,
    )?;
    object
        .properties
        .insert(HEAD_DETAILS_PROPERTY.to_string(), heads);
    object
        .properties
        .insert(DEPENDENCY_DETAILS_PROPERTY.to_string(), dependencies);
    Ok(Value::Object(object))
}

async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            text_analytics_error(
                "addDependencyDetails",
                format!("addDependencyDetails: failed to gather input: {err}"),
            )
        })?);
    }
    Ok(out)
}

fn parse_args(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(text_analytics_error(
            "addDependencyDetails",
            "addDependencyDetails: expected tokenizedDocument input",
        ));
    }
    if args.len() != 1 {
        return Err(text_analytics_error(
            "addDependencyDetails",
            "addDependencyDetails: name-value options are not supported",
        ));
    }
    Ok(args[0].clone())
}

fn tokenized_document_object(value: Value) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => Ok(object),
        Value::Object(object) => Err(text_analytics_error(
            "addDependencyDetails",
            format!(
                "addDependencyDetails: expected tokenizedDocument object, got {}",
                object.class_name
            ),
        )),
        other => Err(text_analytics_error(
            "addDependencyDetails",
            format!("addDependencyDetails: expected tokenizedDocument object, got {other:?}"),
        )),
    }
}

async fn add_sentence_details_for_dependencies(
    object: ObjectInstance,
) -> BuiltinResult<ObjectInstance> {
    let Value::Object(object) = add_sentence_details_builtin(vec![Value::Object(object)]).await?
    else {
        return Err(text_analytics_error(
            "addDependencyDetails",
            "addDependencyDetails: addSentenceDetails did not return tokenizedDocument",
        ));
    };
    Ok(object)
}

fn dependency_details_for_documents(
    documents: &[Vec<String>],
    sentence_numbers: Option<&[Vec<f64>]>,
    language: PosLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> BuiltinResult<(Value, Value)> {
    let mut head_cells = Vec::with_capacity(documents.len());
    let mut dependency_cells = Vec::with_capacity(documents.len());
    if let Some(numbers) = sentence_numbers {
        validate_sentence_numbers(documents, numbers)?;
    }
    for (doc_idx, doc) in documents.iter().enumerate() {
        let default_numbers;
        let numbers = if let Some(numbers) = sentence_numbers.and_then(|all| all.get(doc_idx)) {
            numbers.as_slice()
        } else {
            default_numbers = vec![1.0; doc.len()];
            default_numbers.as_slice()
        };
        let (heads, dependencies) =
            dependency_details_for_document(doc, numbers, language, options);
        head_cells.push(Value::Tensor(
            Tensor::new(heads, vec![1, doc.len()])
                .map_err(|err| text_analytics_error("addDependencyDetails", err))?,
        ));
        dependency_cells.push(Value::StringArray(
            StringArray::new(dependencies, vec![1, doc.len()])
                .map_err(|err| text_analytics_error("addDependencyDetails", err))?,
        ));
    }
    Ok((
        Value::Cell(
            CellArray::new(head_cells, documents.len(), 1)
                .map_err(|err| text_analytics_error("addDependencyDetails", err))?,
        ),
        Value::Cell(
            CellArray::new(dependency_cells, documents.len(), 1)
                .map_err(|err| text_analytics_error("addDependencyDetails", err))?,
        ),
    ))
}

fn validate_sentence_numbers(documents: &[Vec<String>], numbers: &[Vec<f64>]) -> BuiltinResult<()> {
    if numbers.len() != documents.len() {
        return Err(text_analytics_error(
            "addDependencyDetails",
            format!(
                "addDependencyDetails: SentenceNumbers has {} documents but Documents has {}",
                numbers.len(),
                documents.len()
            ),
        ));
    }
    for (idx, (doc_numbers, doc)) in numbers.iter().zip(documents).enumerate() {
        if doc_numbers.len() != doc.len() {
            return Err(text_analytics_error(
                "addDependencyDetails",
                format!(
                    "addDependencyDetails: SentenceNumbers entry {} has {} values but document has {} tokens",
                    idx + 1,
                    doc_numbers.len(),
                    doc.len()
                ),
            ));
        }
        if doc_numbers
            .iter()
            .any(|value| !value.is_finite() || *value < 1.0 || value.fract() != 0.0)
        {
            return Err(text_analytics_error(
                "addDependencyDetails",
                format!(
                    "addDependencyDetails: SentenceNumbers entry {} contains invalid sentence numbers",
                    idx + 1
                ),
            ));
        }
    }
    Ok(())
}

fn dependency_details_for_document(
    doc: &[String],
    sentence_numbers: &[f64],
    language: PosLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> (Vec<f64>, Vec<String>) {
    let mut heads = vec![0.0; doc.len()];
    let mut dependencies = vec!["dep".to_string(); doc.len()];
    let mut start = 0usize;
    while start < doc.len() {
        let sentence = sentence_numbers.get(start).copied().unwrap_or(1.0);
        let mut end = start + 1;
        while end < doc.len() && sentence_numbers.get(end).copied().unwrap_or(1.0) == sentence {
            end += 1;
        }
        fill_sentence_dependencies(
            doc,
            start..end,
            language,
            options,
            &mut heads,
            &mut dependencies,
        );
        start = end;
    }
    (heads, dependencies)
}

fn fill_sentence_dependencies(
    doc: &[String],
    range: std::ops::Range<usize>,
    language: PosLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
    heads: &mut [f64],
    dependencies: &mut [String],
) {
    if range.is_empty() {
        return;
    }
    let root = sentence_root(doc, range.clone(), language, options);
    for idx in range.clone() {
        if idx == root {
            heads[idx] = 0.0;
            dependencies[idx] = "root".to_string();
            continue;
        }
        let tag = part_of_speech_for_token(&doc[idx], language, options);
        let (head, dependency) = dependency_for_token(doc, range.clone(), idx, root, tag, options);
        heads[idx] = (head + 1) as f64;
        dependencies[idx] = dependency.to_string();
    }
}

fn sentence_root(
    doc: &[String],
    range: std::ops::Range<usize>,
    language: PosLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> usize {
    for preferred in ["verb", "auxiliary-verb"] {
        if let Some(idx) = range
            .clone()
            .find(|idx| part_of_speech_for_token(&doc[*idx], language, options) == preferred)
        {
            return idx;
        }
    }
    range
        .clone()
        .find(|idx| {
            document_token_type_with_options(&doc[*idx], options) != DocumentTokenType::Punctuation
        })
        .unwrap_or(range.start)
}

fn dependency_for_token(
    doc: &[String],
    range: std::ops::Range<usize>,
    idx: usize,
    root: usize,
    tag: &str,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> (usize, &'static str) {
    match tag {
        "punctuation" => (root, "punct"),
        "determiner" => (
            nearest_content_to_right(doc, idx, range.clone(), options).unwrap_or(root),
            "det",
        ),
        "adjective" => (
            nearest_content_to_right(doc, idx, range.clone(), options).unwrap_or(root),
            "amod",
        ),
        "numeral" => (
            nearest_content_to_right(doc, idx, range.clone(), options).unwrap_or(root),
            "nummod",
        ),
        "adverb" => (root, "advmod"),
        "coord-conjunction" => (
            nearest_content_to_left(doc, idx, range.clone(), options).unwrap_or(root),
            "cc",
        ),
        "subord-conjunction" => (root, "mark"),
        "adposition" | "particle" => (
            nearest_content_to_right(doc, idx, range.clone(), options).unwrap_or(root),
            "case",
        ),
        "auxiliary-verb" => (root, "aux"),
        "pronoun" | "noun" => {
            if idx < root {
                (root, "nsubj")
            } else if has_near_left_adposition(doc, idx, range, options) {
                (root, "obl")
            } else {
                (root, "obj")
            }
        }
        _ => (root, "dep"),
    }
}

fn nearest_content_to_right(
    doc: &[String],
    idx: usize,
    range: std::ops::Range<usize>,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> Option<usize> {
    ((idx + 1)..range.end).find(|candidate| is_content_token(&doc[*candidate], options))
}

fn nearest_content_to_left(
    doc: &[String],
    idx: usize,
    range: std::ops::Range<usize>,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> Option<usize> {
    (range.start..idx)
        .rev()
        .find(|candidate| is_content_token(&doc[*candidate], options))
}

fn has_near_left_adposition(
    doc: &[String],
    idx: usize,
    range: std::ops::Range<usize>,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> bool {
    let start = idx.saturating_sub(3).max(range.start);
    (start..idx).any(|candidate| {
        let lower = doc[candidate].to_ascii_lowercase();
        matches!(
            lower.as_str(),
            "in" | "on" | "at" | "by" | "from" | "with" | "under" | "over" | "to" | "of"
        ) && is_content_token(&doc[idx], options)
    })
}

fn is_content_token(
    token: &str,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> bool {
    !matches!(
        document_token_type_with_options(token, options),
        DocumentTokenType::Punctuation
            | DocumentTokenType::WebAddress
            | DocumentTokenType::EmailAddress
            | DocumentTokenType::Hashtag
            | DocumentTokenType::AtMention
            | DocumentTokenType::Emoticon
            | DocumentTokenType::Emoji
            | DocumentTokenType::Other
    )
}

pub(in crate::builtins::strings::text_analytics) fn dependency_heads_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Option<Vec<Vec<f64>>>> {
    let Some(value) = object.properties.get(HEAD_DETAILS_PROPERTY) else {
        return Ok(None);
    };
    let Value::Cell(cell) = value else {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid HeadDetails property"),
        ));
    };
    if cell.cols != 1 {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid HeadDetails shape"),
        ));
    }
    let mut out = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        let Value::Tensor(tensor) = item else {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument object has invalid HeadDetails entry"),
            ));
        };
        if tensor.rows != 1 {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument object has invalid HeadDetails entry shape"),
            ));
        }
        out.push(tensor_utils::tensor_values_f64(tensor));
    }
    Ok(Some(out))
}

pub(in crate::builtins::strings::text_analytics) fn dependency_details_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Option<Vec<Vec<String>>>> {
    let Some(value) = object.properties.get(DEPENDENCY_DETAILS_PROPERTY) else {
        return Ok(None);
    };
    let Value::Cell(cell) = value else {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid DependencyDetails property"),
        ));
    };
    if cell.cols != 1 {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid DependencyDetails shape"),
        ));
    }
    let mut out = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        let Value::StringArray(array) = item else {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument object has invalid DependencyDetails entry"),
            ));
        };
        if array.rows != 1 {
            return Err(text_analytics_error(
                fn_name,
                format!(
                    "{fn_name}: tokenizedDocument object has invalid DependencyDetails entry shape"
                ),
            ));
        }
        out.push(array.data.clone());
    }
    Ok(Some(out))
}

fn sentence_numbers_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Option<Vec<Vec<f64>>>> {
    let Some(value) = object.properties.get("SentenceNumbers") else {
        return Ok(None);
    };
    let Value::Cell(cell) = value else {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid SentenceNumbers property"),
        ));
    };
    if cell.cols != 1 {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid SentenceNumbers shape"),
        ));
    }
    let mut out = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        let Value::Tensor(tensor) = item else {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument object has invalid SentenceNumbers entry"),
            ));
        };
        if tensor.rows != 1 {
            return Err(text_analytics_error(
                fn_name,
                format!(
                    "{fn_name}: tokenizedDocument object has invalid SentenceNumbers entry shape"
                ),
            ));
        }
        out.push(tensor_utils::tensor_values_f64(tensor));
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::strings::text_analytics::details::token_details_builtin;
    use crate::builtins::strings::text_analytics::documents::tokenized_document_builtin;
    use crate::builtins::table::{table_variable_names_from_object, table_variables};
    use runmat_value::IntegerStorage;

    fn run_tokenized(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tokenized_document_builtin(args))
    }

    fn run_add_dependency(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(add_dependency_details_builtin(args))
    }

    fn run_token_details(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(token_details_builtin(value))
    }

    fn object(value: Value) -> ObjectInstance {
        let Value::Object(object) = value else {
            panic!("expected object");
        };
        object
    }

    fn table_column(table: &ObjectInstance, name: &str) -> Value {
        table_variables(table)
            .expect("table variables")
            .fields
            .get(name)
            .cloned()
            .unwrap_or_else(|| panic!("missing table column {name}"))
    }

    fn string_column(table: &ObjectInstance, name: &str) -> Vec<String> {
        match table_column(table, name) {
            Value::StringArray(array) => array.data,
            other => panic!("expected string column {name}, got {other:?}"),
        }
    }

    fn numeric_column(table: &ObjectInstance, name: &str) -> Vec<f64> {
        match table_column(table, name) {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected numeric column {name}, got {other:?}"),
        }
    }

    fn poisoned_integer_vector(storage: IntegerStorage, cols: usize) -> Value {
        let tensor = Tensor::new_integer(storage, vec![1, cols]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn dependency_numeric_properties_read_typed_integer_storage_exactly() {
        let mut object = ObjectInstance::new(TOKENIZED_DOCUMENT_CLASS.to_string());
        object.properties.insert(
            HEAD_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![poisoned_integer_vector(IntegerStorage::U16(vec![0, 1]), 2)],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        object.properties.insert(
            "SentenceNumbers".to_string(),
            Value::Cell(
                CellArray::new(
                    vec![poisoned_integer_vector(IntegerStorage::I16(vec![1, 2]), 2)],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );

        assert_eq!(
            dependency_heads_from_object(&object, "tokenDetails")
                .expect("heads")
                .expect("stored"),
            vec![vec![0.0, 1.0]]
        );
        assert_eq!(
            sentence_numbers_from_object(&object, "addDependencyDetails")
                .expect("numbers")
                .expect("stored"),
            vec![vec![1.0, 2.0]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_dependency_details_adds_head_dependency_and_sentence_columns() {
        let docs =
            run_tokenized(vec![Value::String("The dogs chase cats.".into())]).expect("tokenized");
        let updated = run_add_dependency(vec![docs]).expect("dependencies");
        let table = object(run_token_details(updated).expect("details"));

        assert_eq!(
            table_variable_names_from_object(&table).expect("names"),
            vec![
                "Token",
                "DocumentNumber",
                "SentenceNumber",
                "LineNumber",
                "Type",
                "Language",
                "Head",
                "Dependency"
            ]
        );
        assert_eq!(
            string_column(&table, "Token"),
            vec!["The", "dogs", "chase", "cats", "."]
        );
        assert_eq!(
            numeric_column(&table, "Head"),
            vec![2.0, 3.0, 0.0, 3.0, 3.0]
        );
        assert_eq!(
            string_column(&table, "Dependency"),
            vec!["det", "nsubj", "root", "obj", "punct"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_dependency_details_respects_sentence_boundaries() {
        let docs =
            run_tokenized(vec![Value::String("Dogs run. Cats sleep.".into())]).expect("tokenized");
        let updated = run_add_dependency(vec![docs]).expect("dependencies");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            numeric_column(&table, "Head"),
            vec![2.0, 0.0, 2.0, 5.0, 0.0, 5.0]
        );
        assert_eq!(
            string_column(&table, "Dependency"),
            vec!["nsubj", "root", "punct", "nsubj", "root", "punct"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_dependency_details_rejects_options_and_bad_languages() {
        let docs = run_tokenized(vec![Value::String("dogs run".into())]).expect("tokenized");
        let err = run_add_dependency(vec![
            docs.clone(),
            Value::String("DiscardKnownValues".into()),
            Value::Bool(true),
        ])
        .expect_err("expected bad option");
        assert!(err
            .to_string()
            .contains("name-value options are not supported"));

        let mut unsupported = object(docs);
        unsupported
            .properties
            .insert("Language".into(), Value::String("fr".into()));
        let err =
            run_add_dependency(vec![Value::Object(unsupported)]).expect_err("unsupported language");
        assert!(err.to_string().contains("unsupported document language"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_dependency_details_rejects_malformed_sentence_numbers() {
        let docs = object(run_tokenized(vec![Value::String("dogs run".into())]).unwrap());
        let mut malformed = docs;
        malformed.properties.insert(
            "SentenceNumbers".to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap())],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        let err =
            run_add_dependency(vec![Value::Object(malformed)]).expect_err("expected bad numbers");
        assert!(err.to_string().contains("SentenceNumbers entry"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn token_details_rejects_malformed_dependency_details() {
        let docs = object(run_tokenized(vec![Value::String("dogs run".into())]).unwrap());
        let mut malformed = docs.clone();
        malformed.properties.insert(
            HEAD_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap())],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        malformed.properties.insert(
            DEPENDENCY_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["root".into(), "dep".into()], vec![1, 2]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        let err = run_token_details(Value::Object(malformed)).expect_err("expected error");
        assert!(err.to_string().contains("HeadDetails entry"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_dependency_details_accepts_german_language_metadata() {
        let docs = object(run_tokenized(vec![Value::String("Hunde laufen.".into())]).unwrap());
        let mut german = docs;
        german
            .properties
            .insert("Language".into(), Value::String("de".into()));
        let updated = run_add_dependency(vec![Value::Object(german)]).expect("german deps");
        let table = object(run_token_details(updated).expect("details"));
        assert!(string_column(&table, "Dependency").contains(&"root".to_string()));
    }
}
