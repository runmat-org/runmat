//! Entity detail helpers for Text Analytics tokenized documents.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ObjectInstance, ResolveContext, StringArray, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    document_token_type_with_options, documents_from_object, options_from_document_object,
    replace_tokenized_document_documents, text_analytics_error, tokenized_document_language,
    DocumentTokenType, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::pos::{
    add_part_of_speech_details_builtin, part_of_speech_details_from_object, POS_DETAILS_PROPERTY,
};
use crate::{gather_if_needed_async, BuiltinResult};

pub(in crate::builtins::strings::text_analytics) const ENTITY_DETAILS_PROPERTY: &str =
    "EntityDetails";

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

const IN_DOCUMENTS_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documents",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: RetokenizeMethod, DiscardKnownValues, Model.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.ADD_ENTITY_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:addEntityDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "addEntityDetails: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const ADD_ENTITY_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "updatedDocuments = addEntityDetails(documents)",
            inputs: &IN_DOCUMENTS,
            outputs: &OUT_DOCUMENTS,
        },
        BuiltinSignatureDescriptor {
            label: "updatedDocuments = addEntityDetails(documents,Name,Value)",
            inputs: &IN_DOCUMENTS_REST,
            outputs: &OUT_DOCUMENTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

#[runtime_builtin(
    name = "addEntityDetails",
    category = "strings/text_analytics",
    summary = "Add named-entity details to tokenizedDocument objects.",
    keywords = "addEntityDetails,text analytics,tokenizedDocument,entity,named entity,ner",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::entities::ADD_ENTITY_DETAILS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::entities"
)]
async fn add_entity_details_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args).await?;
    let (documents, options) = parse_args(gathered)?;
    let mut object = tokenized_document_object(documents)?;
    let language = EntityLanguage::from_document(&tokenized_document_language(&object))?;

    if options.retokenize_method == RetokenizeMethod::Entity {
        let documents = documents_from_object(&object, "addEntityDetails")?;
        let retokenized = documents
            .iter()
            .map(|doc| retokenize_for_entities(doc, language))
            .collect::<Vec<_>>();
        if retokenized != documents {
            replace_tokenized_document_documents(&mut object, retokenized, "addEntityDetails")?;
            clear_token_aligned_details(&mut object);
        }
    }

    if !object.properties.contains_key(POS_DETAILS_PROPERTY)
        || !object.properties.contains_key("SentenceNumbers")
    {
        object = add_part_of_speech_details_for_entities(object).await?;
    }

    let documents = documents_from_object(&object, "addEntityDetails")?;
    let stored = if options.discard_known_values {
        None
    } else {
        entity_details_from_object(&object, "addEntityDetails")?
    };
    let document_options = options_from_document_object(&object);
    let entity_tags = entity_details_for_documents(
        &documents,
        stored.as_deref(),
        language,
        &document_options,
        options.discard_known_values,
    )?;
    object
        .properties
        .insert(ENTITY_DETAILS_PROPERTY.to_string(), entity_tags);
    mark_entity_pos_as_proper_nouns(&mut object)?;
    Ok(Value::Object(object))
}

async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            text_analytics_error(
                "addEntityDetails",
                format!("addEntityDetails: failed to gather input: {err}"),
            )
        })?);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
struct AddEntityOptions {
    discard_known_values: bool,
    retokenize_method: RetokenizeMethod,
}

impl Default for AddEntityOptions {
    fn default() -> Self {
        Self {
            discard_known_values: false,
            retokenize_method: RetokenizeMethod::Entity,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RetokenizeMethod {
    Entity,
    None,
}

impl RetokenizeMethod {
    fn parse(value: &str) -> BuiltinResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "entity" => Ok(Self::Entity),
            "none" => Ok(Self::None),
            other => Err(text_analytics_error(
                "addEntityDetails",
                format!("addEntityDetails: unsupported RetokenizeMethod '{other}'"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntityLanguage {
    English,
    Japanese,
    German,
    Korean,
}

impl EntityLanguage {
    fn from_document(language: &str) -> BuiltinResult<Self> {
        match language.trim().to_ascii_lowercase().as_str() {
            "en" => Ok(Self::English),
            "ja" => Ok(Self::Japanese),
            "de" => Ok(Self::German),
            "ko" => Ok(Self::Korean),
            other => Err(text_analytics_error(
                "addEntityDetails",
                format!("addEntityDetails: unsupported document language '{other}'"),
            )),
        }
    }
}

fn parse_args(args: Vec<Value>) -> BuiltinResult<(Value, AddEntityOptions)> {
    if args.is_empty() {
        return Err(text_analytics_error(
            "addEntityDetails",
            "addEntityDetails: expected tokenizedDocument input",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(text_analytics_error(
            "addEntityDetails",
            "addEntityDetails: name-value options must appear in pairs",
        ));
    }
    let mut options = AddEntityOptions::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "addEntityDetails")
            .map_err(|err| text_analytics_error("addEntityDetails", err.to_string()))?;
        if name.eq_ignore_ascii_case("DiscardKnownValues") {
            options.discard_known_values = logical_scalar(&args[idx + 1])?;
        } else if name.eq_ignore_ascii_case("RetokenizeMethod") {
            let value = scalar_text(&args[idx + 1], "addEntityDetails")
                .map_err(|err| text_analytics_error("addEntityDetails", err.to_string()))?;
            options.retokenize_method = RetokenizeMethod::parse(&value)?;
        } else if name.eq_ignore_ascii_case("Model") {
            parse_model_option(&args[idx + 1])?;
        } else {
            return Err(text_analytics_error(
                "addEntityDetails",
                format!("addEntityDetails: unsupported option '{name}'"),
            ));
        }
        idx += 2;
    }
    Ok((args[0].clone(), options))
}

fn parse_model_option(value: &Value) -> BuiltinResult<()> {
    if scalar_text(value, "addEntityDetails")
        .map(|model| model.eq_ignore_ascii_case("auto"))
        .unwrap_or(false)
    {
        return Ok(());
    }
    Err(text_analytics_error(
        "addEntityDetails",
        "addEntityDetails: only Model value 'auto' is supported; custom entity models remain tracked",
    ))
}

fn tokenized_document_object(value: Value) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => Ok(object),
        Value::Object(object) => Err(text_analytics_error(
            "addEntityDetails",
            format!(
                "addEntityDetails: expected tokenizedDocument object, got {}",
                object.class_name
            ),
        )),
        other => Err(text_analytics_error(
            "addEntityDetails",
            format!("addEntityDetails: expected tokenizedDocument object, got {other:?}"),
        )),
    }
}

fn clear_token_aligned_details(object: &mut ObjectInstance) {
    for property in [
        "TypeDetails",
        "SentenceNumbers",
        "LemmaDetails",
        POS_DETAILS_PROPERTY,
        ENTITY_DETAILS_PROPERTY,
        "HeadDetails",
        "DependencyDetails",
    ] {
        object.properties.remove(property);
    }
}

async fn add_part_of_speech_details_for_entities(
    object: ObjectInstance,
) -> BuiltinResult<ObjectInstance> {
    let Value::Object(object) = add_part_of_speech_details_builtin(vec![
        Value::Object(object),
        Value::String("RetokenizeMethod".to_string()),
        Value::String("none".to_string()),
    ])
    .await?
    else {
        return Err(text_analytics_error(
            "addEntityDetails",
            "addEntityDetails: addPartOfSpeechDetails did not return tokenizedDocument",
        ));
    };
    Ok(object)
}

fn entity_details_for_documents(
    documents: &[Vec<String>],
    stored: Option<&[Vec<String>]>,
    language: EntityLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
    discard_known_values: bool,
) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .enumerate()
        .map(|(doc_idx, doc)| {
            let tags = doc
                .iter()
                .enumerate()
                .map(|(token_idx, token)| {
                    if !discard_known_values {
                        if let Some(tag) = stored
                            .and_then(|values| values.get(doc_idx))
                            .and_then(|values| values.get(token_idx))
                            .filter(|tag| is_known_entity(tag))
                        {
                            return tag.clone();
                        }
                    }
                    entity_for_token(token, language, options).to_string()
                })
                .collect::<Vec<_>>();
            StringArray::new(tags, vec![1, doc.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("addEntityDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("addEntityDetails", err))?,
    ))
}

fn entity_for_token(
    token: &str,
    language: EntityLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> &'static str {
    if token.split_whitespace().count() > 1 {
        return match language {
            EntityLanguage::English => english_entity(token),
            EntityLanguage::German => german_entity(token),
            EntityLanguage::Japanese | EntityLanguage::Korean => asian_entity(token),
        };
    }
    match document_token_type_with_options(token, options) {
        DocumentTokenType::Punctuation | DocumentTokenType::Digits => "non-entity",
        DocumentTokenType::WebAddress
        | DocumentTokenType::EmailAddress
        | DocumentTokenType::Hashtag
        | DocumentTokenType::AtMention
        | DocumentTokenType::Emoticon
        | DocumentTokenType::Emoji
        | DocumentTokenType::Other => "non-entity",
        DocumentTokenType::Letters => match language {
            EntityLanguage::English => english_entity(token),
            EntityLanguage::German => german_entity(token),
            EntityLanguage::Japanese | EntityLanguage::Korean => asian_entity(token),
        },
    }
}

fn english_entity(token: &str) -> &'static str {
    let compact = compact_entity_token(token);
    let lower = compact.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "mary"
            | "john"
            | "jane"
            | "michael"
            | "sarah"
            | "david"
            | "robert"
            | "linda"
            | "elizabeth"
            | "william"
            | "james"
    ) {
        return "person";
    }
    if matches!(
        lower.as_str(),
        "mathworks"
            | "openai"
            | "microsoft"
            | "google"
            | "apple"
            | "amazon"
            | "runmat"
            | "volkswagen"
    ) {
        return "organization";
    }
    if matches!(
        lower.as_str(),
        "natick"
            | "massachusetts"
            | "boston"
            | "california"
            | "london"
            | "paris"
            | "tokyo"
            | "seoul"
            | "berlin"
            | "munich"
            | "newyork"
            | "sanfrancisco"
            | "unitedstates"
            | "usa"
            | "u.s.a."
    ) {
        return "location";
    }
    if token.chars().next().is_some_and(char::is_uppercase) {
        "other"
    } else {
        "non-entity"
    }
}

fn german_entity(token: &str) -> &'static str {
    let compact = compact_entity_token(token);
    let lower = compact.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "hans" | "anna" | "maria" | "peter" | "johann" | "max" | "sophie"
    ) {
        return "person";
    }
    if matches!(
        lower.as_str(),
        "volkswagen" | "sap" | "siemens" | "bmw" | "mercedes" | "mathworks" | "runmat"
    ) {
        return "organization";
    }
    if matches!(
        lower.as_str(),
        "berlin" | "münchen" | "munich" | "hamburg" | "wolfsburg" | "deutschland" | "germany"
    ) {
        return "location";
    }
    if token.chars().next().is_some_and(char::is_uppercase) {
        "other"
    } else {
        "non-entity"
    }
}

fn asian_entity(token: &str) -> &'static str {
    let lower = token.to_ascii_lowercase();
    if matches!(lower.as_str(), "tokyo" | "seoul" | "japan" | "korea") {
        "location"
    } else {
        "non-entity"
    }
}

fn compact_entity_token(token: &str) -> String {
    token
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '.')
        .collect()
}

fn retokenize_for_entities(tokens: &[String], language: EntityLanguage) -> Vec<String> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < tokens.len() {
        if let Some((merged, consumed)) = multiword_entity_at(tokens, idx, language) {
            out.push(merged);
            idx += consumed;
        } else {
            out.push(tokens[idx].clone());
            idx += 1;
        }
    }
    out
}

fn multiword_entity_at(
    tokens: &[String],
    idx: usize,
    language: EntityLanguage,
) -> Option<(String, usize)> {
    let token = tokens.get(idx)?;
    let mut parts = vec![token.clone()];
    let mut cursor = idx + 1;
    while let Some(next) = tokens.get(cursor) {
        if next == "." || next == "," || next == "-" {
            break;
        }
        if !looks_like_entity_word(next, language) {
            break;
        }
        parts.push(next.clone());
        cursor += 1;
    }
    if parts.len() < 2 || !looks_like_entity_word(token, language) {
        return None;
    }
    let merged = parts.join(" ");
    let tag = match language {
        EntityLanguage::English => english_entity(&merged),
        EntityLanguage::German => german_entity(&merged),
        EntityLanguage::Japanese | EntityLanguage::Korean => asian_entity(&merged),
    };
    if tag == "non-entity" {
        None
    } else {
        Some((merged, cursor - idx))
    }
}

fn looks_like_entity_word(token: &str, language: EntityLanguage) -> bool {
    if !token.chars().next().is_some_and(char::is_uppercase) {
        return false;
    }
    match language {
        EntityLanguage::English => english_entity(token) != "non-entity",
        EntityLanguage::German => german_entity(token) != "non-entity",
        EntityLanguage::Japanese | EntityLanguage::Korean => false,
    }
}

fn mark_entity_pos_as_proper_nouns(object: &mut ObjectInstance) -> BuiltinResult<()> {
    let Some(entities) = entity_details_from_object(object, "addEntityDetails")? else {
        return Ok(());
    };
    let Some(mut pos) = part_of_speech_details_from_object(object, "addEntityDetails")? else {
        return Ok(());
    };
    for (doc_idx, entity_doc) in entities.iter().enumerate() {
        if let Some(pos_doc) = pos.get_mut(doc_idx) {
            for (token_idx, entity) in entity_doc.iter().enumerate() {
                if entity != "non-entity" {
                    if let Some(pos_tag) = pos_doc.get_mut(token_idx) {
                        *pos_tag = "proper-noun".to_string();
                    }
                }
            }
        }
    }
    let values = pos
        .iter()
        .map(|doc| {
            StringArray::new(doc.clone(), vec![1, doc.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("addEntityDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    object.properties.insert(
        POS_DETAILS_PROPERTY.to_string(),
        Value::Cell(
            CellArray::new(values, pos.len(), 1)
                .map_err(|err| text_analytics_error("addEntityDetails", err))?,
        ),
    );
    Ok(())
}

fn is_known_entity(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && trimmed != "unknown"
        && !crate::builtins::strings::common::is_missing_string(trimmed)
}

pub(in crate::builtins::strings::text_analytics) fn entity_details_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Option<Vec<Vec<String>>>> {
    let Some(value) = object.properties.get(ENTITY_DETAILS_PROPERTY) else {
        return Ok(None);
    };
    let Value::Cell(cell) = value else {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid EntityDetails property"),
        ));
    };
    if cell.cols != 1 {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid EntityDetails shape"),
        ));
    }
    let mut out = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        let Value::StringArray(array) = item else {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument object has invalid EntityDetails entry"),
            ));
        };
        if array.rows != 1 {
            return Err(text_analytics_error(
                fn_name,
                format!(
                    "{fn_name}: tokenizedDocument object has invalid EntityDetails entry shape"
                ),
            ));
        }
        out.push(array.data.clone());
    }
    Ok(Some(out))
}

fn logical_scalar(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return match value.try_to_u64() {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    _ => Err(text_analytics_error("addEntityDetails", format!("addEntityDetails: logical scalar option must be true or false, got {value:?}"))),
                };
            }
            match tensor_utils::tensor_value_f64(tensor, 0) {
                0.0 => Ok(false),
                1.0 => Ok(true),
                other => Err(text_analytics_error(
                    "addEntityDetails",
                    format!(
                    "addEntityDetails: logical scalar option must be true or false, got {other}"
                ),
                )),
            }
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(text_analytics_error(
            "addEntityDetails",
            format!("addEntityDetails: logical scalar option must be true or false, got {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::strings::text_analytics::details::token_details_builtin;
    use crate::builtins::strings::text_analytics::documents::tokenized_document_builtin;
    use crate::builtins::table::{table_variable_names_from_object, table_variables};
    use runmat_builtins::{IntegerStorage, LogicalArray, Tensor};

    fn run_tokenized(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tokenized_document_builtin(args))
    }

    fn run_add_entity(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(add_entity_details_builtin(args))
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

    fn poisoned_integer_scalar(storage: IntegerStorage) -> Value {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn logical_scalar_reads_typed_integer_storage_exactly() {
        assert!(
            logical_scalar(&poisoned_integer_scalar(IntegerStorage::U8(vec![1]))).expect("true")
        );
        assert!(
            !logical_scalar(&poisoned_integer_scalar(IntegerStorage::I16(vec![0]))).expect("false")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_entity_details_adds_sentence_pos_and_entity_columns() {
        let docs =
            run_tokenized(vec![Value::String("Mary uses MATLAB at MathWorks.".into())]).unwrap();
        let updated = run_add_entity(vec![docs]).expect("entities");
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
                "PartOfSpeech",
                "Entity"
            ]
        );
        assert_eq!(
            string_column(&table, "Entity"),
            vec![
                "person",
                "non-entity",
                "other",
                "non-entity",
                "organization",
                "non-entity"
            ]
        );
        assert_eq!(
            string_column(&table, "PartOfSpeech"),
            vec![
                "proper-noun",
                "verb",
                "proper-noun",
                "adposition",
                "proper-noun",
                "punctuation"
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_entity_details_retokenizes_multiword_entities() {
        let docs = run_tokenized(vec![Value::String("Mary moved to New York.".into())]).unwrap();
        let updated = run_add_entity(vec![docs]).expect("entities");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Token"),
            vec!["Mary", "moved", "to", "New York", "."]
        );
        assert_eq!(
            string_column(&table, "Entity"),
            vec![
                "person",
                "non-entity",
                "non-entity",
                "location",
                "non-entity"
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_entity_details_preserves_existing_known_values_unless_discarding() {
        let docs = object(
            run_tokenized(vec![
                Value::StringArray(
                    StringArray::new(vec!["dogs".into(), "run".into()], vec![1, 2]).unwrap(),
                ),
                Value::String("TokenizeMethod".into()),
                Value::String("none".into()),
            ])
            .expect("tokenized"),
        );
        let mut stale = docs.clone();
        stale.properties.insert(
            ENTITY_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["custom".into(), "".into()], vec![1, 2]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );

        let preserved = run_add_entity(vec![
            Value::Object(stale.clone()),
            Value::String("RetokenizeMethod".into()),
            Value::String("none".into()),
        ])
        .expect("preserve");
        let table = object(run_token_details(preserved).expect("details"));
        assert_eq!(
            string_column(&table, "Entity"),
            vec!["custom", "non-entity"]
        );

        let recomputed = run_add_entity(vec![
            Value::Object(stale),
            Value::String("RetokenizeMethod".into()),
            Value::String("none".into()),
            Value::String("DiscardKnownValues".into()),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
        ])
        .expect("recompute");
        let table = object(run_token_details(recomputed).expect("details"));
        assert_eq!(
            string_column(&table, "Entity"),
            vec!["non-entity", "non-entity"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_entity_details_adds_sentence_details_when_pos_already_exists() {
        let docs = object(
            run_tokenized(vec![
                Value::StringArray(
                    StringArray::new(vec!["Mary".into(), "runs".into()], vec![1, 2]).unwrap(),
                ),
                Value::String("TokenizeMethod".into()),
                Value::String("none".into()),
            ])
            .expect("tokenized"),
        );
        let mut with_pos = docs;
        with_pos.properties.insert(
            POS_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["custom".into(), "verb".into()], vec![1, 2]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );

        let updated = run_add_entity(vec![
            Value::Object(with_pos),
            Value::String("RetokenizeMethod".into()),
            Value::String("none".into()),
        ])
        .expect("entities");
        let table = object(run_token_details(updated).expect("details"));
        assert!(table_variable_names_from_object(&table)
            .expect("names")
            .contains(&"SentenceNumber".to_string()));
        assert_eq!(
            string_column(&table, "PartOfSpeech"),
            vec!["proper-noun", "verb"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_entity_details_rejects_bad_options_languages_and_models() {
        let docs = run_tokenized(vec![Value::String("Mary runs".into())]).unwrap();
        let err = run_add_entity(vec![
            docs.clone(),
            Value::String("RetokenizeMethod".into()),
            Value::String("bogus".into()),
        ])
        .expect_err("bad retokenizer");
        assert!(err.to_string().contains("unsupported RetokenizeMethod"));

        let err = run_add_entity(vec![
            docs.clone(),
            Value::String("Model".into()),
            Value::String("custom".into()),
        ])
        .expect_err("bad model");
        assert!(err.to_string().contains("only Model value"));

        let mut unsupported = object(docs);
        unsupported
            .properties
            .insert("Language".into(), Value::String("fr".into()));
        let err =
            run_add_entity(vec![Value::Object(unsupported)]).expect_err("unsupported language");
        assert!(err.to_string().contains("unsupported document language"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn token_details_rejects_malformed_entity_details() {
        let docs = object(run_tokenized(vec![Value::String("Mary runs".into())]).unwrap());
        let mut malformed = docs.clone();
        malformed.properties.insert(
            ENTITY_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["person".into()], vec![1, 1]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        let err = run_token_details(Value::Object(malformed)).expect_err("expected error");
        assert!(err.to_string().contains("EntityDetails entry"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_entity_details_discard_known_values_ignores_malformed_stored_tags() {
        let docs = object(run_tokenized(vec![Value::String("Mary runs".into())]).unwrap());
        let mut malformed = docs.clone();
        malformed.properties.insert(
            ENTITY_DETAILS_PROPERTY.to_string(),
            Value::String("bad".into()),
        );
        let updated = run_add_entity(vec![
            Value::Object(malformed),
            Value::String("DiscardKnownValues".into()),
            Value::Bool(true),
        ])
        .expect("discard malformed");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Entity"),
            vec!["person", "non-entity"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_entity_details_accepts_german_language_metadata() {
        let docs = object(
            run_tokenized(vec![Value::String("Volkswagen ist in Wolfsburg.".into())]).unwrap(),
        );
        let mut german = docs;
        german
            .properties
            .insert("Language".into(), Value::String("de".into()));
        let updated = run_add_entity(vec![Value::Object(german)]).expect("german entities");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Entity"),
            vec![
                "organization",
                "non-entity",
                "non-entity",
                "location",
                "non-entity"
            ]
        );
    }

    #[test]
    fn logical_scalar_rejects_non_logical_numeric_values() {
        let err = logical_scalar(&Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()))
            .expect_err("bad logical");
        assert!(err.to_string().contains("logical scalar"));
    }
}
