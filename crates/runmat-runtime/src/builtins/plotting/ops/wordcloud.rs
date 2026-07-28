//! MATLAB-compatible `wordcloud` chart builtin.

use glam::{Vec3, Vec4};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ObjectInstance, StringArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{Figure, TextStyle};
use std::collections::HashMap;

use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    register_wordcloud_handle, update_wordcloud_figure, update_wordcloud_handle_state,
    PlotRenderOptions, WordCloudHandleState,
};
use crate::builtins::plotting::state::{select_axes_for_figure, select_figure, FigureHandle};
use crate::builtins::plotting::style::{
    parse_color_value, value_as_f64, value_as_string, LineStyleParseOptions,
};
use crate::builtins::strings::common::char_row_to_string_slice;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "wordcloud";
const DEFAULT_MAX_DISPLAY_WORDS: usize = 100;
const DEFAULT_COLOR: Vec4 = Vec4::new(0.0, 0.4470, 0.7410, 1.0);
const DEFAULT_HIGHLIGHT_COLOR: Vec4 = Vec4::new(0.8500, 0.3250, 0.0980, 1.0);
const DEFAULT_FONT_NAME: &str = "Helvetica";
const DEFAULT_POSITION: [f64; 4] = [0.13, 0.11, 0.775, 0.815];
const MAX_RENDERED_WORDS: usize = 1000;
const TABLE_VARIABLES_FIELD: &str = "__table_variables";
const TABLE_PROPERTIES_FIELD: &str = "__table_properties";

const OUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "wc",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the WordCloudChart object.",
}];

const IN_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text, tokenizedDocument, bag model, categorical array, or LDA model.",
}];

const IN_WORDS_SIZES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Word vector.",
    },
    BuiltinParamDescriptor {
        name: "sizeData",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Word size vector.",
    },
];

const IN_TABLE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "tbl",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table with word and size variables.",
    },
    BuiltinParamDescriptor {
        name: "wordVar",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table variable containing words.",
    },
    BuiltinParamDescriptor {
        name: "sizeVar",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table variable containing word sizes.",
    },
];

const IN_REST: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "WordCloudChart name-value properties.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(str)",
        inputs: &IN_DATA,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(documents)",
        inputs: &IN_DATA,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(bag)",
        inputs: &IN_DATA,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(words, sizeData)",
        inputs: &IN_WORDS_SIZES,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(tbl, wordVar, sizeVar)",
        inputs: &IN_TABLE,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(C)",
        inputs: &IN_DATA,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(parent, ...)",
        inputs: &IN_REST,
        outputs: &OUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "wc = wordcloud(..., Name, Value)",
        inputs: &IN_REST,
        outputs: &OUT_HANDLE,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORDCLOUD.INVALID_ARGUMENT",
    identifier: Some("RunMat:wordcloud:InvalidArgument"),
    when: "Input data, table selectors, parent axes, or WordCloudChart properties are invalid.",
    message: "wordcloud: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORDCLOUD.INTERNAL",
    identifier: Some("RunMat:wordcloud:Internal"),
    when: "Internal plotting state update fails while creating or mutating a word cloud.",
    message: "wordcloud: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const WORDCLOUD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[derive(Clone)]
struct WordCloudOptions {
    max_display_words: usize,
    color: Vec<Vec4>,
    highlight_color: Vec4,
    shape: String,
    layout_num: usize,
    size_power: f64,
    title: String,
    title_font_name: String,
    visible: bool,
    font_name: String,
    box_visible: bool,
    units: String,
    position: [f64; 4],
    handle_visibility: String,
    display_name: String,
    tag: String,
}

impl Default for WordCloudOptions {
    fn default() -> Self {
        Self {
            max_display_words: DEFAULT_MAX_DISPLAY_WORDS,
            color: vec![DEFAULT_COLOR],
            highlight_color: DEFAULT_HIGHLIGHT_COLOR,
            shape: "oval".to_string(),
            layout_num: 1,
            size_power: 0.5,
            title: String::new(),
            title_font_name: DEFAULT_FONT_NAME.to_string(),
            visible: true,
            font_name: DEFAULT_FONT_NAME.to_string(),
            box_visible: false,
            units: "normalized".to_string(),
            position: DEFAULT_POSITION,
            handle_visibility: "on".to_string(),
            display_name: String::new(),
            tag: String::new(),
        }
    }
}

struct WordCloudData {
    words: Vec<String>,
    sizes: Vec<f64>,
    word_variable: String,
    size_variable: String,
}

#[runtime_builtin(
    name = "wordcloud",
    category = "plotting",
    summary = "Create a word cloud chart from text, tokenized documents, bag models, or word-size data.",
    keywords = "wordcloud,text analytics,bagOfWords,bagOfNgrams,categorical,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::plotting::type_resolvers::handle_scalar_type),
    descriptor(crate::builtins::plotting::wordcloud::WORDCLOUD_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::wordcloud"
)]
pub fn wordcloud_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    let rest = split_leading_parent_handle(args)?;
    let (data_args, options) = split_options(rest)?;
    let data = parse_data(data_args)?;
    validate_options_for_data(&options, &data)?;
    render_wordcloud(data, options)
}

fn split_leading_parent_handle(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    if args.len() < 2 {
        return Ok(args);
    }
    match resolve_plot_handle(&args[0], BUILTIN_NAME) {
        Ok(PlotHandle::Figure(handle)) => {
            select_figure(handle);
            Ok(args[1..].to_vec())
        }
        Ok(PlotHandle::Axes(handle, axes_index)) => {
            select_axes_for_figure(handle, axes_index)
                .map_err(|err| wordcloud_error(format!("wordcloud: {err}")))?;
            Ok(args[1..].to_vec())
        }
        Ok(PlotHandle::Root) => Err(wordcloud_error(
            "Parent must be a figure or chart container",
        )),
        Ok(_) => Err(wordcloud_error(
            "Parent must be a figure or chart container",
        )),
        Err(_) => {
            if let Some(handle) = numeric_figure_parent(&args[0]) {
                select_figure(handle);
                Ok(args[1..].to_vec())
            } else {
                Ok(args)
            }
        }
    }
}

fn numeric_figure_parent(value: &Value) -> Option<FigureHandle> {
    let scalar = value_as_f64(value)?;
    if scalar.is_finite() && scalar > 0.0 && scalar.fract() == 0.0 && scalar <= u32::MAX as f64 {
        Some(FigureHandle::from(scalar as u32))
    } else {
        None
    }
}

fn split_options(args: Vec<Value>) -> BuiltinResult<(Vec<Value>, WordCloudOptions)> {
    let mut start = args.len();
    let mut pairs = Vec::<(String, Value)>::new();
    while start >= 2 {
        let Some(name) = value_as_string(&args[start - 2]) else {
            break;
        };
        let key = canonical_key(&name);
        if !is_option_key(&key) {
            break;
        }
        pairs.push((key, args[start - 1].clone()));
        start -= 2;
    }
    let mut options = WordCloudOptions::default();
    for (key, value) in pairs.into_iter().rev() {
        apply_option(&mut options, &key, &value)?;
    }
    Ok((args[..start].to_vec(), options))
}

fn is_option_key(key: &str) -> bool {
    matches!(
        key,
        "maxdisplaywords"
            | "color"
            | "highlightcolor"
            | "shape"
            | "layoutnum"
            | "sizepower"
            | "title"
            | "titlefontname"
            | "visible"
            | "fontname"
            | "box"
            | "units"
            | "position"
            | "handlevisibility"
            | "displayname"
            | "tag"
    )
}

fn apply_option(options: &mut WordCloudOptions, key: &str, value: &Value) -> BuiltinResult<()> {
    match key {
        "maxdisplaywords" => {
            options.max_display_words = max_display_words_value(value)?;
        }
        "color" => {
            options.color = color_list(value, "Color", None)?;
        }
        "highlightcolor" => {
            options.highlight_color = parse_color(value)?;
        }
        "shape" => {
            let shape = text_scalar(value, "Shape")?.to_ascii_lowercase();
            if shape != "oval" && shape != "rectangle" {
                return Err(wordcloud_error("Shape must be 'oval' or 'rectangle'"));
            }
            options.shape = shape;
        }
        "layoutnum" => {
            options.layout_num = nonnegative_integer(value, "LayoutNum")?;
        }
        "sizepower" => {
            let power = numeric_scalar(value, "SizePower")?;
            if power <= 0.0 {
                return Err(wordcloud_error("SizePower must be positive"));
            }
            options.size_power = power;
        }
        "title" => {
            options.title = text_scalar(value, "Title")?;
        }
        "titlefontname" => {
            options.title_font_name = text_scalar(value, "TitleFontName")?;
        }
        "visible" => {
            options.visible = on_off_bool(value, "Visible")?;
        }
        "fontname" => {
            options.font_name = text_scalar(value, "FontName")?;
        }
        "box" => {
            options.box_visible = on_off_bool(value, "Box")?;
        }
        "units" => {
            options.units = units_value(value)?;
        }
        "position" => {
            options.position = position_value(value, "Position")?;
        }
        "handlevisibility" => {
            options.handle_visibility = handle_visibility_value(value)?;
        }
        "displayname" => {
            options.display_name = text_scalar(value, "DisplayName")?;
        }
        "tag" => {
            options.tag = text_scalar(value, "Tag")?;
        }
        other => {
            return Err(wordcloud_error(format!(
                "unsupported WordCloudChart property `{other}`"
            )))
        }
    }
    Ok(())
}

fn parse_data(args: Vec<Value>) -> BuiltinResult<WordCloudData> {
    match args.as_slice() {
        [] => Err(wordcloud_error("expected input data")),
        [value] => parse_single_data(value),
        [words, sizes] if is_numeric_vector_like(sizes) => words_and_sizes(words, sizes),
        [Value::Object(object), word_var, size_var] if is_table_object(object) => {
            table_words_and_sizes(object, word_var, size_var)
        }
        [Value::Object(object), _topic_idx] if object.is_class("ldaModel") => Err(wordcloud_error(
            "LDA topic word clouds require ldaModel support and remain tracked by the Text Analytics umbrella",
        )),
        _ => Err(wordcloud_error(
            "expected text, tokenizedDocument, bag model, categorical, words and sizes, or table inputs",
        )),
    }
}

fn validate_options_for_data(
    options: &WordCloudOptions,
    data: &WordCloudData,
) -> BuiltinResult<()> {
    if options.color.len() > 1 && options.color.len() != data.words.len() {
        return Err(wordcloud_error(
            "Color matrix rows must match WordData length",
        ));
    }
    Ok(())
}

fn parse_single_data(value: &Value) -> BuiltinResult<WordCloudData> {
    match value {
        Value::Object(object) if object.is_class("tokenizedDocument") => {
            documents_counts(documents_from_object(object)?)
        }
        Value::Object(object) if object.is_class("bagOfWords") => bag_of_words_counts(object),
        Value::Object(object) if object.is_class("bagOfNgrams") => bag_of_ngrams_counts(object),
        Value::Object(object) if object.is_class("categorical") => categorical_counts(object),
        Value::Object(object) if object.is_class("ldaModel") => Err(wordcloud_error(
            "LDA topic word clouds require ldaModel support and remain tracked by the Text Analytics umbrella",
        )),
        Value::Object(object) => Err(wordcloud_error(format!(
            "unsupported object class {}",
            object.class_name
        ))),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
            documents_counts(text_documents(value)?)
        }
        other => Err(wordcloud_error(format!("unsupported input {other:?}"))),
    }
}

fn words_and_sizes(words_value: &Value, sizes_value: &Value) -> BuiltinResult<WordCloudData> {
    let words = words_from_word_vector(words_value)?;
    let sizes = numeric_vector(sizes_value, "sizeData")?;
    if words.len() != sizes.len() {
        return Err(wordcloud_error(
            "words and sizeData must contain the same number of elements",
        ));
    }
    aggregate_pairs(words, sizes)
}

fn table_words_and_sizes(
    object: &ObjectInstance,
    word_var: &Value,
    size_var: &Value,
) -> BuiltinResult<WordCloudData> {
    let variables = match object.properties.get(TABLE_VARIABLES_FIELD) {
        Some(Value::Struct(st)) => st,
        _ => return Err(wordcloud_error("table input is missing variable storage")),
    };
    let names = table_variable_names(object, variables)?;
    let word_name = table_selector(word_var, &names, "wordVar")?;
    let size_name = table_selector(size_var, &names, "sizeVar")?;
    let words_value = variables
        .fields
        .get(&word_name)
        .ok_or_else(|| wordcloud_error(format!("table is missing variable '{word_name}'")))?;
    let sizes_value = variables
        .fields
        .get(&size_name)
        .ok_or_else(|| wordcloud_error(format!("table is missing variable '{size_name}'")))?;
    let mut data = words_and_sizes(words_value, sizes_value)?;
    data.word_variable = word_name;
    data.size_variable = size_name;
    Ok(data)
}

fn table_variable_names(
    object: &ObjectInstance,
    variables: &StructValue,
) -> BuiltinResult<Vec<String>> {
    if let Some(Value::Struct(props)) = object
        .properties
        .get(TABLE_PROPERTIES_FIELD)
        .or_else(|| object.properties.get("Properties"))
    {
        if let Some(value) = props.fields.get("VariableNames") {
            let names = words_from_word_vector(value)?;
            if names.len() == variables.fields.len() {
                return Ok(names);
            }
        }
    }
    Ok(variables.field_names().cloned().collect())
}

fn table_selector(value: &Value, names: &[String], label: &str) -> BuiltinResult<String> {
    if let Some(name) = value_as_string(value) {
        return names
            .iter()
            .find(|candidate| candidate.eq_ignore_ascii_case(name.trim()))
            .cloned()
            .ok_or_else(|| wordcloud_error(format!("{label} '{name}' is not a table variable")));
    }
    let indices = numeric_vector(value, label)?;
    if indices.len() != 1 {
        return Err(wordcloud_error(format!(
            "{label} must select exactly one table variable"
        )));
    }
    let idx = indices[0];
    if idx.fract() != 0.0 || idx <= 0.0 || idx as usize > names.len() {
        return Err(wordcloud_error(format!("{label} index is out of range")));
    }
    Ok(names[idx as usize - 1].clone())
}

fn is_table_object(object: &ObjectInstance) -> bool {
    object.is_class("table") || object.is_class("timetable")
}

fn text_documents(value: &Value) -> BuiltinResult<Vec<Vec<String>>> {
    match value {
        Value::String(text) => Ok(vec![tokenize_text(text)]),
        Value::StringArray(array) => {
            Ok(array.data.iter().map(|text| tokenize_text(text)).collect())
        }
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row(&array.data, array.cols, 0)
            };
            Ok(vec![tokenize_text(&text)])
        }
        Value::CharArray(array) => {
            let mut docs = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                docs.push(tokenize_text(&char_row(&array.data, array.cols, row)));
            }
            Ok(docs)
        }
        Value::Cell(cell) => {
            let mut docs = Vec::with_capacity(cell.data.len());
            for item in &cell.data {
                docs.push(tokenize_text(&text_scalar(item, "cell text")?));
            }
            Ok(docs)
        }
        _ => Err(wordcloud_error("expected text input")),
    }
}

fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            push_token(&mut tokens, &mut current);
        }
    }
    if !current.is_empty() {
        push_token(&mut tokens, &mut current);
    }
    tokens
}

fn push_token(tokens: &mut Vec<String>, current: &mut String) {
    let token = current.trim_matches('\'').to_string();
    current.clear();
    if token.len() < 2 || is_stop_word(&token) {
        return;
    }
    tokens.push(token);
}

fn is_stop_word(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "has"
            | "he"
            | "in"
            | "is"
            | "it"
            | "its"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "this"
            | "to"
            | "was"
            | "were"
            | "will"
            | "with"
    )
}

fn documents_from_object(object: &ObjectInstance) -> BuiltinResult<Vec<Vec<String>>> {
    let Some(Value::Cell(cell)) = object.properties.get("Documents") else {
        return Err(wordcloud_error(
            "tokenizedDocument object missing Documents property",
        ));
    };
    let mut docs = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        docs.push(words_from_word_vector(item)?);
    }
    Ok(docs)
}

fn documents_counts(documents: Vec<Vec<String>>) -> BuiltinResult<WordCloudData> {
    let mut positions = HashMap::<String, usize>::new();
    let mut words = Vec::<String>::new();
    let mut sizes = Vec::<f64>::new();
    for token in documents.into_iter().flatten() {
        if let Some(idx) = positions.get(&token).copied() {
            sizes[idx] += 1.0;
        } else {
            positions.insert(token.clone(), words.len());
            words.push(token);
            sizes.push(1.0);
        }
    }
    Ok(WordCloudData {
        words,
        sizes,
        word_variable: String::new(),
        size_variable: String::new(),
    })
}

fn bag_of_words_counts(object: &ObjectInstance) -> BuiltinResult<WordCloudData> {
    let vocabulary = object
        .properties
        .get("Vocabulary")
        .ok_or_else(|| wordcloud_error("bagOfWords object missing Vocabulary property"))
        .and_then(words_from_word_vector)?;
    let counts = tensor_property(object, "Counts", "bagOfWords")?;
    counts_by_columns(vocabulary, counts)
}

fn bag_of_ngrams_counts(object: &ObjectInstance) -> BuiltinResult<WordCloudData> {
    let counts = tensor_property(object, "Counts", "bagOfNgrams")?;
    let labels = if let Some(Value::StringArray(array)) = object.properties.get("Ngrams") {
        ngram_labels(array)
    } else {
        object
            .properties
            .get("Vocabulary")
            .ok_or_else(|| wordcloud_error("bagOfNgrams object missing Ngrams property"))
            .and_then(words_from_word_vector)?
    };
    counts_by_columns(labels, counts)
}

fn ngram_labels(array: &StringArray) -> Vec<String> {
    let mut labels = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let mut parts = Vec::new();
        for col in 0..array.cols {
            let text = array.data[row + col * array.rows].trim();
            if !text.is_empty() {
                parts.push(text.to_string());
            }
        }
        labels.push(parts.join(" "));
    }
    labels
}

fn counts_by_columns(words: Vec<String>, counts: Tensor) -> BuiltinResult<WordCloudData> {
    if counts.cols != words.len() {
        return Err(wordcloud_error(
            "count matrix column count must match word data length",
        ));
    }
    let mut sizes = Vec::with_capacity(counts.cols);
    for col in 0..counts.cols {
        let mut sum = 0.0;
        for row in 0..counts.rows {
            let value = counts.data[row + col * counts.rows];
            if !value.is_finite() || value < 0.0 {
                return Err(wordcloud_error("counts must be finite nonnegative values"));
            }
            sum += value;
        }
        sizes.push(sum);
    }
    aggregate_pairs(words, sizes)
}

fn categorical_counts(object: &ObjectInstance) -> BuiltinResult<WordCloudData> {
    let categories = object
        .properties
        .get("Categories")
        .ok_or_else(|| wordcloud_error("categorical object missing Categories property"))
        .and_then(words_from_word_vector)?;
    let codes = tensor_property(object, "Codes", "categorical")?;
    let mut sizes = vec![0.0; categories.len()];
    for code in codes.data {
        if code.is_nan() {
            continue;
        }
        if code.fract() != 0.0 || code <= 0.0 || code as usize > categories.len() {
            return Err(wordcloud_error("categorical Codes property is invalid"));
        }
        sizes[code as usize - 1] += 1.0;
    }
    aggregate_pairs(categories, sizes)
}

fn aggregate_pairs(words: Vec<String>, sizes: Vec<f64>) -> BuiltinResult<WordCloudData> {
    let mut positions = HashMap::<String, usize>::new();
    let mut out_words = Vec::new();
    let mut out_sizes = Vec::new();
    for (word, size) in words.into_iter().zip(sizes) {
        if !size.is_finite() || size < 0.0 {
            return Err(wordcloud_error(
                "word sizes must be finite nonnegative values",
            ));
        }
        if word.is_empty() || size == 0.0 {
            continue;
        }
        if let Some(idx) = positions.get(&word).copied() {
            out_sizes[idx] += size;
        } else {
            positions.insert(word.clone(), out_words.len());
            out_words.push(word);
            out_sizes.push(size);
        }
    }
    Ok(WordCloudData {
        words: out_words,
        sizes: out_sizes,
        word_variable: String::new(),
        size_variable: String::new(),
    })
}

fn render_wordcloud(data: WordCloudData, options: WordCloudOptions) -> BuiltinResult<f64> {
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let annotations_slot = std::rc::Rc::new(std::cell::RefCell::new(Vec::<usize>::new()));
    let axes_slot = std::rc::Rc::new(std::cell::RefCell::new(None::<usize>));
    let annotations_out = std::rc::Rc::clone(&annotations_slot);
    let axes_out = std::rc::Rc::clone(&axes_slot);
    let display = display_indices(&data, options.max_display_words);
    let render_words = data.words.clone();
    let render_sizes = data.sizes.clone();
    let render_options = options.clone();

    let render_result = crate::builtins::plotting::state::render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "",
            x_label: "",
            y_label: "",
            grid: false,
            axis_equal: true,
        },
        move |figure, axes| {
            *axes_out.borrow_mut() = Some(axes);
            figure.set_axes_title(axes, render_options.title.clone());
            figure.set_axes_title_style(
                axes,
                TextStyle {
                    visible: render_options.visible,
                    ..Default::default()
                },
            );
            figure.set_axes_box_enabled(axes, render_options.box_visible);
            let mut annotation_indices = Vec::with_capacity(display.len());
            for (rank, idx) in display.iter().copied().enumerate() {
                let position = layout_position(rank, display.len(), &render_options);
                let mut style = TextStyle {
                    color: Some(word_color(rank, idx, &render_options)),
                    font_size: Some(font_size(render_sizes[idx], &render_sizes, &render_options)),
                    visible: render_options.visible,
                    ..Default::default()
                };
                if rank == 0 {
                    style.font_weight = Some("bold".to_string());
                }
                let annotation = figure.add_axes_text_annotation(
                    axes,
                    position,
                    render_words[idx].clone(),
                    style,
                );
                annotation_indices.push(annotation);
            }
            *annotations_out.borrow_mut() = annotation_indices;
            Ok(())
        },
    );

    let Some(axes_index) = *axes_slot.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let state = WordCloudHandleState {
        figure: figure_handle,
        axes_index,
        annotation_indices: annotations_slot.borrow().clone(),
        word_data: data.words,
        size_data: data.sizes,
        word_variable: data.word_variable,
        size_variable: data.size_variable,
        max_display_words: options.max_display_words,
        color: options.color,
        highlight_color: options.highlight_color,
        shape: options.shape,
        layout_num: options.layout_num,
        size_power: options.size_power,
        title: options.title,
        title_font_name: options.title_font_name,
        visible: options.visible,
        font_name: options.font_name,
        box_visible: options.box_visible,
        units: options.units,
        position: options.position,
        handle_visibility: options.handle_visibility,
        display_name: options.display_name,
        tag: options.tag,
    };
    let handle = register_wordcloud_handle(state);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(wordcloud_error_with_descriptor(
            &ERROR_INTERNAL,
            err.message,
        ));
    }
    Ok(handle)
}

pub(crate) fn get_wordcloud_property(
    state: &WordCloudHandleState,
    property: Option<&str>,
    _builtin: &'static str,
) -> BuiltinResult<Value> {
    match property.map(canonical_key).as_deref() {
        None => {
            let mut st = StructValue::new();
            st.insert("Type", Value::String("wordcloud".into()));
            st.insert("Parent", Value::Num(state.figure.as_u32() as f64));
            st.insert("Children", empty_handles());
            st.insert("WordData", string_array_value(state.word_data.clone())?);
            st.insert("SizeData", vector_value(state.size_data.clone())?);
            st.insert("WordVariable", Value::String(state.word_variable.clone()));
            st.insert("SizeVariable", Value::String(state.size_variable.clone()));
            st.insert(
                "MaxDisplayWords",
                Value::Num(state.max_display_words as f64),
            );
            st.insert("Color", color_value(&state.color)?);
            st.insert("HighlightColor", single_color_value(state.highlight_color)?);
            st.insert("Shape", Value::String(state.shape.clone()));
            st.insert("LayoutNum", Value::Num(state.layout_num as f64));
            st.insert("SizePower", Value::Num(state.size_power));
            st.insert("Title", Value::String(state.title.clone()));
            st.insert(
                "TitleFontName",
                Value::String(state.title_font_name.clone()),
            );
            st.insert("FontName", Value::String(state.font_name.clone()));
            st.insert(
                "Visible",
                Value::String(if state.visible { "on" } else { "off" }.into()),
            );
            st.insert(
                "Box",
                Value::String(if state.box_visible { "on" } else { "off" }.into()),
            );
            st.insert("Units", Value::String(state.units.clone()));
            st.insert("Position", vector_value(state.position.to_vec())?);
            st.insert(
                "HandleVisibility",
                Value::String(state.handle_visibility.clone()),
            );
            st.insert("DisplayName", Value::String(state.display_name.clone()));
            st.insert("Tag", Value::String(state.tag.clone()));
            st.insert("SourceTable", Value::Struct(StructValue::new()));
            Ok(Value::Struct(st))
        }
        Some("type") => Ok(Value::String("wordcloud".into())),
        Some("parent") => Ok(Value::Num(state.figure.as_u32() as f64)),
        Some("children") => Ok(empty_handles()),
        Some("worddata") => string_array_value(state.word_data.clone()),
        Some("sizedata") => vector_value(state.size_data.clone()),
        Some("wordvariable") => Ok(Value::String(state.word_variable.clone())),
        Some("sizevariable") => Ok(Value::String(state.size_variable.clone())),
        Some("maxdisplaywords") => Ok(Value::Num(state.max_display_words as f64)),
        Some("color") => color_value(&state.color),
        Some("highlightcolor") => single_color_value(state.highlight_color),
        Some("shape") => Ok(Value::String(state.shape.clone())),
        Some("layoutnum") => Ok(Value::Num(state.layout_num as f64)),
        Some("sizepower") => Ok(Value::Num(state.size_power)),
        Some("title") => Ok(Value::String(state.title.clone())),
        Some("titlefontname") => Ok(Value::String(state.title_font_name.clone())),
        Some("fontname") => Ok(Value::String(state.font_name.clone())),
        Some("visible") => Ok(Value::String(
            if state.visible { "on" } else { "off" }.into(),
        )),
        Some("box") => Ok(Value::String(
            if state.box_visible { "on" } else { "off" }.into(),
        )),
        Some("units") => Ok(Value::String(state.units.clone())),
        Some("position") => vector_value(state.position.to_vec()),
        Some("handlevisibility") => Ok(Value::String(state.handle_visibility.clone())),
        Some("displayname") => Ok(Value::String(state.display_name.clone())),
        Some("tag") => Ok(Value::String(state.tag.clone())),
        Some("sourcetable") => Ok(Value::Struct(StructValue::new())),
        Some(other) => Err(wordcloud_error(format!(
            "unsupported WordCloudChart property `{other}`"
        ))),
    }
}

pub(crate) fn apply_wordcloud_property(
    handle: f64,
    state: &WordCloudHandleState,
    key: &str,
    value: &Value,
    _builtin: &'static str,
) -> BuiltinResult<()> {
    let mut next = match crate::builtins::plotting::state::plot_child_handle_snapshot(handle) {
        Ok(crate::builtins::plotting::state::PlotChildHandleState::WordCloud(current)) => current,
        _ => state.clone(),
    };
    match canonical_key(key).as_str() {
        "worddata" => {
            let words = words_from_word_vector(value)?;
            if words.len() != next.size_data.len() {
                return Err(wordcloud_error(
                    "WordData length must match SizeData length",
                ));
            }
            next.word_data = words;
        }
        "sizedata" => {
            let sizes = numeric_vector(value, "SizeData")?;
            if sizes.len() != next.word_data.len() {
                return Err(wordcloud_error(
                    "SizeData length must match WordData length",
                ));
            }
            validate_sizes(&sizes)?;
            next.size_data = sizes;
        }
        "maxdisplaywords" => {
            next.max_display_words = max_display_words_value(value)?;
        }
        "color" => {
            next.color = color_list(value, "Color", Some(next.word_data.len()))?;
        }
        "highlightcolor" => {
            next.highlight_color = parse_color(value)?;
        }
        "shape" => {
            let shape = text_scalar(value, "Shape")?.to_ascii_lowercase();
            if shape != "oval" && shape != "rectangle" {
                return Err(wordcloud_error("Shape must be 'oval' or 'rectangle'"));
            }
            next.shape = shape;
        }
        "layoutnum" => {
            next.layout_num = nonnegative_integer(value, "LayoutNum")?;
        }
        "sizepower" => {
            let power = numeric_scalar(value, "SizePower")?;
            if power <= 0.0 {
                return Err(wordcloud_error("SizePower must be positive"));
            }
            next.size_power = power;
        }
        "title" => {
            next.title = text_scalar(value, "Title")?;
        }
        "titlefontname" => {
            next.title_font_name = text_scalar(value, "TitleFontName")?;
        }
        "visible" => {
            next.visible = on_off_bool(value, "Visible")?;
        }
        "fontname" => {
            next.font_name = text_scalar(value, "FontName")?;
        }
        "box" => {
            next.box_visible = on_off_bool(value, "Box")?;
        }
        "units" => {
            next.units = units_value(value)?;
        }
        "position" => {
            next.position = position_value(value, "Position")?;
        }
        "handlevisibility" => {
            next.handle_visibility = handle_visibility_value(value)?;
        }
        "displayname" => {
            next.display_name = text_scalar(value, "DisplayName")?;
        }
        "tag" => {
            next.tag = text_scalar(value, "Tag")?;
        }
        "wordvariable" | "sizevariable" | "sourcetable" | "parent" | "children" | "type" => {
            return Err(wordcloud_error(format!("{key} is read-only")));
        }
        other => {
            return Err(wordcloud_error(format!(
                "unsupported WordCloudChart set property `{other}`"
            )));
        }
    }
    refresh_wordcloud_rendering(&mut next)?;
    update_wordcloud_handle_state(handle, next)
        .map_err(|err| wordcloud_error(format!("wordcloud: {err}")))
}

fn refresh_wordcloud_rendering(state: &mut WordCloudHandleState) -> BuiltinResult<()> {
    let display =
        display_indices_from_parts(&state.word_data, &state.size_data, state.max_display_words);
    let location = state.clone();
    update_wordcloud_figure(&location, |figure| {
        figure.set_axes_title(state.axes_index, state.title.clone());
        figure.set_axes_title_style(
            state.axes_index,
            TextStyle {
                visible: state.visible,
                ..Default::default()
            },
        );
        figure.set_axes_box_enabled(state.axes_index, state.box_visible);
        ensure_annotation_slots(figure, state, display.len())?;
        for (slot, annotation_index) in state.annotation_indices.iter().copied().enumerate() {
            let Some(annotation) = figure.axes_text_annotation(state.axes_index, annotation_index)
            else {
                return Err(crate::builtins::plotting::state::FigureError::InvalidPlotObjectHandle);
            };
            let _ = annotation;
            if let Some(idx) = display.get(slot).copied() {
                figure.set_axes_text_annotation_text(
                    state.axes_index,
                    annotation_index,
                    state.word_data[idx].clone(),
                );
                let mut style = TextStyle {
                    color: Some(word_color(slot, idx, &options_from_state(state))),
                    font_size: Some(font_size(
                        state.size_data[idx],
                        &state.size_data,
                        &options_from_state(state),
                    )),
                    visible: state.visible,
                    ..Default::default()
                };
                if slot == 0 {
                    style.font_weight = Some("bold".to_string());
                }
                figure.set_axes_text_annotation_position(
                    state.axes_index,
                    annotation_index,
                    layout_position(slot, display.len(), &options_from_state(state)),
                );
                figure.set_axes_text_annotation_style(state.axes_index, annotation_index, style);
            } else {
                let mut style = TextStyle {
                    visible: false,
                    ..Default::default()
                };
                style.color = Some(DEFAULT_COLOR);
                figure.set_axes_text_annotation_style(state.axes_index, annotation_index, style);
            }
        }
        Ok(())
    })
    .map_err(|err| wordcloud_error(format!("wordcloud: {err}")))?;
    Ok(())
}

fn ensure_annotation_slots(
    figure: &mut Figure,
    state: &mut WordCloudHandleState,
    count: usize,
) -> Result<(), crate::builtins::plotting::state::FigureError> {
    while state.annotation_indices.len() < count {
        let idx = figure.add_axes_text_annotation(
            state.axes_index,
            Vec3::ZERO,
            String::new(),
            TextStyle::default(),
        );
        state.annotation_indices.push(idx);
    }
    Ok(())
}

fn options_from_state(state: &WordCloudHandleState) -> WordCloudOptions {
    WordCloudOptions {
        max_display_words: state.max_display_words,
        color: state.color.clone(),
        highlight_color: state.highlight_color,
        shape: state.shape.clone(),
        layout_num: state.layout_num,
        size_power: state.size_power,
        title: state.title.clone(),
        title_font_name: state.title_font_name.clone(),
        visible: state.visible,
        font_name: state.font_name.clone(),
        box_visible: state.box_visible,
        units: state.units.clone(),
        position: state.position,
        handle_visibility: state.handle_visibility.clone(),
        display_name: state.display_name.clone(),
        tag: state.tag.clone(),
    }
}

fn display_indices(data: &WordCloudData, max_display_words: usize) -> Vec<usize> {
    display_indices_from_parts(&data.words, &data.sizes, max_display_words)
}

fn display_indices_from_parts(
    words: &[String],
    sizes: &[f64],
    max_display_words: usize,
) -> Vec<usize> {
    let mut indices = (0..words.len())
        .filter(|idx| !words[*idx].is_empty() && sizes[*idx] > 0.0)
        .collect::<Vec<_>>();
    indices.sort_by(|a, b| {
        sizes[*b]
            .partial_cmp(&sizes[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    indices.truncate(max_display_words.min(indices.len()));
    indices
}

fn layout_position(rank: usize, total: usize, options: &WordCloudOptions) -> Vec3 {
    if total == 0 {
        return Vec3::ZERO;
    }
    if options.shape == "rectangle" {
        let cols = (total as f64).sqrt().ceil().max(1.0) as usize;
        let rows = total.div_ceil(cols).max(1);
        let row = rank / cols;
        let col = rank % cols;
        let x = if cols == 1 {
            0.0
        } else {
            -1.0 + 2.0 * (col as f32 / (cols - 1) as f32)
        };
        let y = if rows == 1 {
            0.0
        } else {
            1.0 - 2.0 * (row as f32 / (rows - 1) as f32)
        };
        return Vec3::new(x, y, 0.0);
    }
    if rank == 0 {
        return Vec3::ZERO;
    }
    let golden_angle = 2.399_963_1_f32;
    let angle = (rank as f32 + options.layout_num as f32) * golden_angle;
    let radius = ((rank as f32) / (total.max(2) - 1) as f32).sqrt();
    Vec3::new(radius * angle.cos(), 0.72 * radius * angle.sin(), 0.0)
}

fn font_size(size: f64, all_sizes: &[f64], options: &WordCloudOptions) -> f32 {
    let transformed = size.powf(options.size_power);
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for value in all_sizes
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| value.powf(options.size_power))
    {
        min_v = min_v.min(value);
        max_v = max_v.max(value);
    }
    if !min_v.is_finite() || !max_v.is_finite() || (max_v - min_v).abs() < f64::EPSILON {
        return 24.0;
    }
    (12.0 + 34.0 * ((transformed - min_v) / (max_v - min_v)) as f32).clamp(10.0, 52.0)
}

fn word_color(rank: usize, word_index: usize, options: &WordCloudOptions) -> Vec4 {
    if rank == 0 {
        return options.highlight_color;
    }
    if options.color.is_empty() {
        DEFAULT_COLOR
    } else if options.color.len() == 1 {
        options.color[0]
    } else {
        options.color[word_index % options.color.len()]
    }
}

fn tensor_property(object: &ObjectInstance, name: &str, class_name: &str) -> BuiltinResult<Tensor> {
    match object.properties.get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        _ => Err(wordcloud_error(format!(
            "{class_name} object missing {name} property"
        ))),
    }
}

fn is_numeric_vector_like(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        Value::Tensor(tensor) => Ok(tensor.data.clone()),
        other => Err(wordcloud_error(format!(
            "{name} must be numeric, got {other:?}"
        ))),
    }
}

fn validate_sizes(sizes: &[f64]) -> BuiltinResult<()> {
    if sizes.iter().all(|value| value.is_finite() && *value >= 0.0) {
        Ok(())
    } else {
        Err(wordcloud_error(
            "SizeData must contain finite nonnegative values",
        ))
    }
}

fn words_from_word_vector(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(array) if array.rows <= 1 => Ok(vec![if array.rows == 0 {
            String::new()
        } else {
            char_row(&array.data, array.cols, 0)
        }]),
        Value::CharArray(array) => {
            let mut words = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                words.push(char_row(&array.data, array.cols, row));
            }
            Ok(words)
        }
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| text_scalar(item, "word cell"))
            .collect(),
        other => Err(wordcloud_error(format!(
            "expected word vector, got {other:?}"
        ))),
    }
}

fn text_scalar(value: &Value, name: &str) -> BuiltinResult<String> {
    value_as_string(value).ok_or_else(|| wordcloud_error(format!("{name} must be text")))
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    value_as_f64(value)
        .filter(|value| value.is_finite())
        .ok_or_else(|| wordcloud_error(format!("{name} must be a finite numeric scalar")))
}

fn nonnegative_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    let scalar = numeric_scalar(value, name)?;
    if scalar.fract() != 0.0 || scalar < 0.0 || scalar > usize::MAX as f64 {
        return Err(wordcloud_error(format!(
            "{name} must be a nonnegative integer"
        )));
    }
    Ok(scalar as usize)
}

fn max_display_words_value(value: &Value) -> BuiltinResult<usize> {
    let count = nonnegative_integer(value, "MaxDisplayWords")?;
    if count > MAX_RENDERED_WORDS {
        return Err(wordcloud_error(format!(
            "MaxDisplayWords must be <= {MAX_RENDERED_WORDS}"
        )));
    }
    Ok(count)
}

fn units_value(value: &Value) -> BuiltinResult<String> {
    let units = text_scalar(value, "Units")?.to_ascii_lowercase();
    match units.as_str() {
        "normalized" | "pixels" | "inches" | "centimeters" | "points" | "characters" => Ok(units),
        _ => Err(wordcloud_error(
            "Units must be normalized, pixels, inches, centimeters, points, or characters",
        )),
    }
}

fn handle_visibility_value(value: &Value) -> BuiltinResult<String> {
    let visibility = text_scalar(value, "HandleVisibility")?.to_ascii_lowercase();
    match visibility.as_str() {
        "on" | "off" | "callback" => Ok(visibility),
        _ => Err(wordcloud_error(
            "HandleVisibility must be 'on', 'off', or 'callback'",
        )),
    }
}

fn position_value(value: &Value, name: &str) -> BuiltinResult<[f64; 4]> {
    let values = numeric_vector(value, name)?;
    if values.len() != 4 || values.iter().any(|value| !value.is_finite()) {
        return Err(wordcloud_error(format!(
            "{name} must be a four-element finite numeric vector"
        )));
    }
    Ok([values[0], values[1], values[2], values[3]])
}

fn on_off_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Int(value) if value.to_i64() == 0 || value.to_i64() == 1 => Ok(value.to_i64() != 0),
        _ => match value_as_string(value).map(|text| text.to_ascii_lowercase()) {
            Some(text) if text == "on" => Ok(true),
            Some(text) if text == "off" => Ok(false),
            _ => Err(wordcloud_error(format!("{name} must be 'on' or 'off'"))),
        },
    }
}

fn color_list(value: &Value, name: &str, expected_rows: Option<usize>) -> BuiltinResult<Vec<Vec4>> {
    if value_as_string(value).is_some() {
        return Ok(vec![parse_color(value)?]);
    }
    let tensor = Tensor::try_from(value).map_err(wordcloud_error)?;
    if tensor.data.is_empty() {
        return Ok(Vec::new());
    }
    if tensor.data.len() == 3 && (tensor.rows == 1 || tensor.cols == 1) {
        return Ok(vec![rgb_triplet(&tensor.data, name)?]);
    }
    if tensor.cols != 3 {
        return Err(wordcloud_error(format!(
            "{name} matrix must have three columns"
        )));
    }
    if let Some(expected) = expected_rows {
        if tensor.rows != expected {
            return Err(wordcloud_error(format!(
                "{name} matrix rows must match WordData length"
            )));
        }
    }
    let mut colors = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        colors.push(rgb_triplet(
            &[
                tensor.data[row],
                tensor.data[row + tensor.rows],
                tensor.data[row + 2 * tensor.rows],
            ],
            name,
        )?);
    }
    Ok(colors)
}

fn parse_color(value: &Value) -> BuiltinResult<Vec4> {
    parse_color_value(&LineStyleParseOptions::generic(BUILTIN_NAME), value)
}

fn rgb_triplet(values: &[f64], name: &str) -> BuiltinResult<Vec4> {
    if values.len() != 3
        || values
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(wordcloud_error(format!(
            "{name} RGB triplets must contain three finite values in [0,1]"
        )));
    }
    Ok(Vec4::new(
        values[0] as f32,
        values[1] as f32,
        values[2] as f32,
        1.0,
    ))
}

fn color_value(colors: &[Vec4]) -> BuiltinResult<Value> {
    if colors.len() == 1 {
        return single_color_value(colors[0]);
    }
    let mut data = Vec::with_capacity(colors.len() * 3);
    for component in 0..3 {
        for color in colors {
            data.push(color[component] as f64);
        }
    }
    Ok(Value::Tensor(
        Tensor::new(data, vec![colors.len(), 3]).map_err(wordcloud_error)?,
    ))
}

fn single_color_value(color: Vec4) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(
            vec![color.x as f64, color.y as f64, color.z as f64],
            vec![1, 3],
        )
        .map_err(wordcloud_error)?,
    ))
}

fn vector_value(values: Vec<f64>) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(values.clone(), vec![values.len(), 1]).map_err(wordcloud_error)?,
    ))
}

fn string_array_value(values: Vec<String>) -> BuiltinResult<Value> {
    Ok(Value::StringArray(
        StringArray::new(values.clone(), vec![values.len(), 1]).map_err(wordcloud_error)?,
    ))
}

fn empty_handles() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn char_row(data: &[char], cols: usize, row: usize) -> String {
    char_row_to_string_slice(data, cols, row)
        .trim_end()
        .to_string()
}

fn canonical_key(name: impl AsRef<str>) -> String {
    name.as_ref()
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace() && *ch != '_' && *ch != '-')
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn wordcloud_error(message: impl Into<String>) -> RuntimeError {
    wordcloud_error_with_descriptor(&ERROR_INVALID_ARGUMENT, message)
}

fn wordcloud_error_with_descriptor(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message.into()).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, reset_hold_state_for_run};
    use runmat_builtins::CellArray;

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    fn str_array(values: &[&str], shape: Vec<usize>) -> Value {
        Value::StringArray(
            StringArray::new(
                values.iter().map(|value| value.to_string()).collect(),
                shape,
            )
            .unwrap(),
        )
    }

    fn tensor(values: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(values, shape).unwrap())
    }

    #[test]
    fn descriptor_covers_documented_common_forms() {
        let labels = WORDCLOUD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"wc = wordcloud(str)"));
        assert!(labels.contains(&"wc = wordcloud(documents)"));
        assert!(labels.contains(&"wc = wordcloud(bag)"));
        assert!(labels.contains(&"wc = wordcloud(words, sizeData)"));
        assert!(labels.contains(&"wc = wordcloud(tbl, wordVar, sizeVar)"));
        assert!(labels.contains(&"wc = wordcloud(C)"));
    }

    #[test]
    fn wordcloud_from_words_sizes_exposes_chart_properties() {
        let _guard = setup();
        let handle = wordcloud_builtin(vec![
            str_array(&["alpha", "beta", "gamma"], vec![1, 3]),
            tensor(vec![3.0, 7.0, 2.0], vec![1, 3]),
            Value::String("MaxDisplayWords".into()),
            Value::Num(2.0),
            Value::String("Shape".into()),
            Value::String("rectangle".into()),
        ])
        .unwrap();

        let words =
            get_builtin(vec![Value::Num(handle), Value::String("WordData".into())]).unwrap();
        let Value::StringArray(words) = words else {
            panic!("expected WordData string array");
        };
        assert_eq!(words.data, vec!["alpha", "beta", "gamma"]);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("MaxDisplayWords".into())
            ])
            .unwrap(),
            Value::Num(2.0)
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Shape".into())]).unwrap(),
            Value::String("rectangle".into())
        );

        let figure = clone_figure(crate::builtins::plotting::current_figure_handle()).unwrap();
        assert_eq!(figure.axes_text_annotations(0).len(), 2);
    }

    #[test]
    fn wordcloud_accepts_figure_parent_and_chart_properties() {
        let _guard = setup();
        let parent = crate::builtins::plotting::current_figure_handle();
        let handle = wordcloud_builtin(vec![
            Value::Num(parent.as_u32() as f64),
            str_array(&["alpha", "beta"], vec![1, 2]),
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::String("Title".into()),
            Value::String("Vocabulary".into()),
            Value::String("Box".into()),
            Value::String("on".into()),
            Value::String("DisplayName".into()),
            Value::String("Client words".into()),
            Value::String("Tag".into()),
            Value::String("wc1".into()),
        ])
        .unwrap();

        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Parent".into())]).unwrap(),
            Value::Num(parent.as_u32() as f64)
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayName".into())
            ])
            .unwrap(),
            Value::String("Client words".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Tag".into())]).unwrap(),
            Value::String("wc1".into())
        );

        let figure = clone_figure(parent).unwrap();
        let meta = figure.axes_metadata(0).unwrap();
        assert_eq!(meta.title.as_deref(), Some("Vocabulary"));
        assert!(meta.box_enabled);
    }

    #[test]
    fn wordcloud_counts_text_and_tokenized_documents() {
        let _guard = setup();
        let handle =
            wordcloud_builtin(vec![Value::String("alpha beta alpha and gamma".into())]).unwrap();
        let sizes =
            get_builtin(vec![Value::Num(handle), Value::String("SizeData".into())]).unwrap();
        let Value::Tensor(sizes) = sizes else {
            panic!("expected SizeData tensor");
        };
        assert_eq!(sizes.data, vec![2.0, 1.0, 1.0]);

        let docs = tokenized_document_object(vec![
            vec!["delta".into(), "epsilon".into(), "delta".into()],
            vec!["epsilon".into()],
        ]);
        let handle = wordcloud_builtin(vec![docs]).unwrap();
        let words =
            get_builtin(vec![Value::Num(handle), Value::String("WordData".into())]).unwrap();
        let Value::StringArray(words) = words else {
            panic!("expected WordData string array");
        };
        assert_eq!(words.data, vec!["delta", "epsilon"]);
    }

    #[test]
    fn wordcloud_accepts_bag_ngram_categorical_and_table_inputs() {
        let _guard = setup();
        let bag = bag_of_ngrams_object(&["alpha", "beta", "alpha", "gamma"], vec![2.0, 1.0]);
        let handle = wordcloud_builtin(vec![bag]).unwrap();
        let words =
            get_builtin(vec![Value::Num(handle), Value::String("WordData".into())]).unwrap();
        let Value::StringArray(words) = words else {
            panic!("expected ngram labels");
        };
        assert_eq!(words.data, vec!["alpha alpha", "beta gamma"]);

        let categorical = categorical_object(&["red", "blue"], vec![1.0, 2.0, 1.0]);
        let handle = wordcloud_builtin(vec![categorical]).unwrap();
        let sizes =
            get_builtin(vec![Value::Num(handle), Value::String("SizeData".into())]).unwrap();
        let Value::Tensor(sizes) = sizes else {
            panic!("expected categorical counts");
        };
        assert_eq!(sizes.data, vec![2.0, 1.0]);

        let table = table_object(
            &["Word", "Count"],
            vec![
                str_array(&["alpha", "beta"], vec![2, 1]),
                tensor(vec![5.0, 1.0], vec![2, 1]),
            ],
        );
        let handle = wordcloud_builtin(vec![
            table,
            Value::String("Word".into()),
            Value::String("Count".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("MaxDisplayWords".into())
            ])
            .unwrap(),
            Value::Num(100.0)
        );
    }

    #[test]
    fn wordcloud_set_refreshes_properties_and_rendering() {
        let _guard = setup();
        let handle = wordcloud_builtin(vec![
            str_array(&["alpha", "beta", "gamma"], vec![1, 3]),
            tensor(vec![1.0, 2.0, 3.0], vec![1, 3]),
        ])
        .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("MaxDisplayWords".into()),
            Value::Num(1.0),
            Value::String("Visible".into()),
            Value::String("off".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Visible".into())]).unwrap(),
            Value::String("off".into())
        );
        let figure = clone_figure(crate::builtins::plotting::current_figure_handle()).unwrap();
        assert!(!figure.axes_text_annotations(0)[0].style.visible);
    }

    #[test]
    fn wordcloud_rejects_invalid_sizes_and_unsupported_lda() {
        let _guard = setup();
        let err = wordcloud_builtin(vec![
            str_array(&["alpha"], vec![1, 1]),
            tensor(vec![-1.0], vec![1, 1]),
        ])
        .expect_err("negative sizes should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);

        let err = wordcloud_builtin(vec![
            str_array(&["alpha", "beta"], vec![1, 2]),
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::String("Color".into()),
            tensor(
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                vec![3, 3],
            ),
        ])
        .expect_err("Color rows must match words at creation");
        assert!(err.to_string().contains("Color matrix rows"));

        let lda = Value::Object(ObjectInstance::new("ldaModel".into()));
        let err = wordcloud_builtin(vec![lda, Value::Num(1.0)])
            .expect_err("LDA model should be explicit unsupported");
        assert!(err.to_string().contains("ldaModel"));
    }

    fn tokenized_document_object(documents: Vec<Vec<String>>) -> Value {
        let doc_count = documents.len();
        let cell_values = documents
            .into_iter()
            .map(|words| {
                Value::StringArray(StringArray::new(words.clone(), vec![1, words.len()]).unwrap())
            })
            .collect::<Vec<_>>();
        let mut object = ObjectInstance::new("tokenizedDocument".into());
        object.properties.insert(
            "Documents".into(),
            Value::Cell(CellArray::new(cell_values, doc_count, 1).unwrap()),
        );
        Value::Object(object)
    }

    fn bag_of_ngrams_object(ngrams_column_major: &[&str], counts: Vec<f64>) -> Value {
        let mut object = ObjectInstance::new("bagOfNgrams".into());
        object
            .properties
            .insert("Ngrams".into(), str_array(ngrams_column_major, vec![2, 2]));
        object
            .properties
            .insert("Counts".into(), tensor(counts, vec![1, 2]));
        Value::Object(object)
    }

    fn categorical_object(categories: &[&str], codes: Vec<f64>) -> Value {
        let mut object = ObjectInstance::new("categorical".into());
        object.properties.insert(
            "Categories".into(),
            str_array(categories, vec![1, categories.len()]),
        );
        object
            .properties
            .insert("Codes".into(), tensor(codes.clone(), vec![codes.len(), 1]));
        Value::Object(object)
    }

    fn table_object(names: &[&str], columns: Vec<Value>) -> Value {
        let mut variables = StructValue::new();
        for (name, column) in names.iter().zip(columns) {
            variables.insert(*name, column);
        }
        let mut props = StructValue::new();
        props.insert("VariableNames", str_array(names, vec![1, names.len()]));
        let mut object = ObjectInstance::new("table".into());
        object
            .properties
            .insert(TABLE_VARIABLES_FIELD.into(), Value::Struct(variables));
        object
            .properties
            .insert(TABLE_PROPERTIES_FIELD.into(), Value::Struct(props.clone()));
        object
            .properties
            .insert("Properties".into(), Value::Struct(props));
        Value::Object(object)
    }
}
