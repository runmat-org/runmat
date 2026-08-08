use super::*;
use crate::builtins::table::display::trim_float;
use runmat_builtins::NumericScalar;

pub(crate) fn categorical_from_args(args: Vec<Value>) -> BuiltinResult<Value> {
    let source = args
        .first()
        .cloned()
        .unwrap_or_else(|| Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()));
    let parsed = parse_categorical_args(&args[1..])?;
    let labels = categorical_labels(&source)?;
    let category_plan =
        categorical_category_plan(&source, &labels, parsed.valueset.as_ref(), parsed.catnames)?;
    let mut codes = Vec::with_capacity(labels.len());
    for label in labels {
        if let Some(idx) = category_plan
            .lookup
            .iter()
            .position(|value| value == &label)
        {
            codes.push((category_plan.codes[idx] + 1) as f64);
        } else {
            codes.push(f64::NAN);
        }
    }
    let mut object = ObjectInstance::new(CATEGORICAL_CLASS.to_string());
    object.properties.insert(
        "Codes".to_string(),
        Value::Tensor(
            Tensor::new(codes, value_shape_or_column(&source)?).map_err(invalid_variable)?,
        ),
    );
    object.properties.insert(
        "Categories".to_string(),
        Value::StringArray(
            StringArray::new(
                category_plan.categories.clone(),
                vec![1, category_plan.categories.len()],
            )
            .map_err(invalid_variable)?,
        ),
    );
    object
        .properties
        .insert("Ordinal".to_string(), Value::Bool(parsed.ordinal));
    object.properties.insert(
        "Protected".to_string(),
        Value::Bool(parsed.ordinal || parsed.protected.unwrap_or(false)),
    );
    Ok(Value::Object(object))
}

pub(in crate::builtins::table) fn ordinal_from_args(mut args: Vec<Value>) -> BuiltinResult<Value> {
    args.push(Value::from("Ordinal"));
    args.push(Value::Bool(true));
    categorical_from_args(args)
}

#[derive(Clone, Copy)]
pub(crate) enum CategoricalComparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

pub(crate) fn categorical_compare(
    lhs: &Value,
    rhs: &Value,
    comparison: CategoricalComparison,
) -> Option<BuiltinResult<Value>> {
    if let Err(err) = validate_categorical_category_compatibility(lhs, rhs) {
        return Some(Err(err));
    }
    let base = categorical_compare_base(lhs).or_else(|| categorical_compare_base(rhs))?;
    let left = match categorical_compare_operand(lhs, &base.categories) {
        Ok(operand) => operand,
        Err(err) => return Some(Err(err)),
    };
    let right = match categorical_compare_operand(rhs, &base.categories) {
        Ok(operand) => operand,
        Err(err) => return Some(Err(err)),
    };
    if !matches!(
        comparison,
        CategoricalComparison::Eq | CategoricalComparison::Ne
    ) && (!left.ordinal || !right.ordinal)
    {
        return Some(Err(invalid_argument(
            "categorical: relational comparisons require ordinal categorical arrays",
        )));
    }
    let shape = match crate::builtins::common::broadcast::broadcast_shapes(
        "categorical",
        &left.shape,
        &right.shape,
    ) {
        Ok(shape) => shape,
        Err(_) => {
            return Some(Err(invalid_argument(
                "categorical: array sizes are not compatible for broadcasting",
            )));
        }
    };
    let total = crate::builtins::common::tensor::element_count(&shape);
    let left_strides = crate::builtins::common::broadcast::compute_strides(&left.shape);
    let right_strides = crate::builtins::common::broadcast::compute_strides(&right.shape);
    let mut data = Vec::with_capacity(total);
    for idx in 0..total {
        let left_idx = crate::builtins::common::broadcast::broadcast_index(
            idx,
            &shape,
            &left.shape,
            &left_strides,
        );
        let right_idx = crate::builtins::common::broadcast::broadcast_index(
            idx,
            &shape,
            &right.shape,
            &right_strides,
        );
        let flag = compare_category_codes(left.codes[left_idx], right.codes[right_idx], comparison);
        data.push(if flag { 1 } else { 0 });
    }
    Some(categorical_logical_result(data, shape))
}

pub(crate) struct CategoricalExtremaEvaluation {
    values: Value,
    indices: Value,
}

impl CategoricalExtremaEvaluation {
    fn new(values: Value, indices: Value) -> Self {
        Self { values, indices }
    }

    fn into_value(self) -> Value {
        self.values
    }

    fn into_pair(self) -> (Value, Value) {
        (self.values, self.indices)
    }
}

pub(crate) fn categorical_extrema_to_value(
    eval: CategoricalExtremaEvaluation,
) -> BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.into_value()]));
        }
        let (values, indices) = eval.into_pair();
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![values, indices],
        ));
    }
    Ok(eval.into_value())
}

pub(crate) async fn categorical_max_evaluate(
    value: &Value,
    rest: &[Value],
) -> Option<BuiltinResult<CategoricalExtremaEvaluation>> {
    let Value::Object(object) = value else {
        return None;
    };
    if !object.is_class(CATEGORICAL_CLASS) {
        return None;
    }
    Some(categorical_extrema_evaluate(object, rest, true).await)
}

pub(crate) async fn categorical_min_evaluate(
    value: &Value,
    rest: &[Value],
) -> Option<BuiltinResult<CategoricalExtremaEvaluation>> {
    let Value::Object(object) = value else {
        return None;
    };
    if !object.is_class(CATEGORICAL_CLASS) {
        return None;
    }
    Some(categorical_extrema_evaluate(object, rest, false).await)
}

async fn categorical_extrema_evaluate(
    object: &ObjectInstance,
    rest: &[Value],
    maximum: bool,
) -> BuiltinResult<CategoricalExtremaEvaluation> {
    if !categorical_is_ordinal(object) {
        return Err(invalid_argument(
            "categorical: min and max require ordinal categorical arrays",
        ));
    }
    let codes = categorical_codes(object)?;
    let categories = categorical_categories_value(object)?;
    let protected = object
        .properties
        .get("Protected")
        .cloned()
        .unwrap_or_else(|| Value::Bool(true));
    let (value, indices) = if maximum {
        crate::builtins::math::reduction::evaluate_max(Value::Tensor(codes), rest)
            .await?
            .into_pair()
    } else {
        crate::builtins::math::reduction::evaluate_min(Value::Tensor(codes), rest)
            .await?
            .into_pair()
    };
    let reduced = categorical_from_code_value(value, categories, protected)?;
    Ok(CategoricalExtremaEvaluation::new(reduced, indices))
}

fn categorical_from_code_value(
    value: Value,
    categories: Value,
    protected: Value,
) -> BuiltinResult<Value> {
    let codes = match value {
        Value::Tensor(tensor) => tensor,
        Value::Num(number) => Tensor::new(vec![number], vec![1, 1]).map_err(invalid_variable)?,
        other => {
            return Err(invalid_variable(format!(
                "categorical: extrema returned unsupported code value {other:?}"
            )))
        }
    };
    let mut object = ObjectInstance::new(CATEGORICAL_CLASS.to_string());
    object
        .properties
        .insert("Codes".to_string(), Value::Tensor(codes));
    object
        .properties
        .insert("Categories".to_string(), categories);
    object
        .properties
        .insert("Ordinal".to_string(), Value::Bool(true));
    object.properties.insert("Protected".to_string(), protected);
    Ok(Value::Object(object))
}

pub(crate) fn categorical_labels(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![normalize_category_label(text)]),
        Value::StringArray(array) => Ok(array
            .data
            .iter()
            .map(|value| normalize_category_label(value))
            .collect()),
        Value::CharArray(array) => Ok(char_rows(array)
            .into_iter()
            .map(|value| normalize_category_label(&value))
            .collect()),
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid_variable("categorical: invalid numeric tensor storage"))
                    .map(categorical_numeric_label)
            })
            .collect(),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|flag| if *flag != 0 { "true" } else { "false" }.to_string())
            .collect()),
        Value::Cell(cell) => cell.data.iter().map(cell_scalar_label).collect(),
        Value::Object(object) if object.is_class(CATEGORICAL_CLASS) => {
            categorical_object_labels(object)
        }
        other => Ok(vec![other.to_string()]),
    }
}

pub(in crate::builtins::table) fn cell_scalar_label(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(normalize_category_label(text)),
        Value::CharArray(array) if array.rows == 1 => {
            let text: String = array.data.iter().collect();
            Ok(normalize_category_label(&text))
        }
        Value::Num(value) if value.is_nan() => Ok(String::new()),
        Value::Num(value) => Ok(format_key_number(*value)),
        Value::Bool(value) => Ok(if *value { "true" } else { "false" }.to_string()),
        other => Ok(other.to_string()),
    }
}

pub(in crate::builtins::table) fn value_shape_or_column(
    value: &Value,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.shape.clone()),
        Value::StringArray(array) => Ok(array.shape.clone()),
        Value::LogicalArray(array) => Ok(array.shape.clone()),
        Value::Cell(cell) => Ok(cell.shape.clone()),
        Value::CharArray(array) => Ok(vec![array.rows, 1]),
        _ => Ok(vec![1, 1]),
    }
}

#[derive(Default)]
struct CategoricalArgs {
    valueset: Option<Value>,
    catnames: Option<Vec<String>>,
    ordinal: bool,
    protected: Option<bool>,
}

fn parse_categorical_args(args: &[Value]) -> BuiltinResult<CategoricalArgs> {
    let mut parsed = CategoricalArgs::default();
    let mut idx = 0usize;
    if idx < args.len() && !is_categorical_option_pair(args, idx) {
        parsed.valueset = Some(args[idx].clone());
        idx += 1;
    }
    if idx < args.len() && !is_categorical_option_pair(args, idx) {
        parsed.catnames = Some(categorical_category_names(&args[idx])?);
        idx += 1;
    }
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "categorical: name-value options must be provided in pairs",
            ));
        }
        let option_name = scalar_text(&args[idx], "categorical option")?;
        if option_name.eq_ignore_ascii_case("Ordinal") {
            parsed.ordinal = categorical_bool_scalar(&args[idx + 1], "Ordinal")?;
        } else if option_name.eq_ignore_ascii_case("Protected") {
            parsed.protected = Some(categorical_bool_scalar(&args[idx + 1], "Protected")?);
        } else {
            return Err(invalid_argument(format!(
                "categorical: unsupported option '{option_name}'"
            )));
        }
        idx += 2;
    }
    Ok(parsed)
}

fn categorical_category_names(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => categorical_labels(value),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| match value {
                Value::String(_) | Value::CharArray(_) => cell_scalar_label(value),
                _ => Err(invalid_argument(
                    "categorical: category names must be a string array or cell array of character vectors",
                )),
            })
            .collect(),
        _ => Err(invalid_argument(
            "categorical: category names must be a string array or cell array of character vectors",
        )),
    }
}

fn is_categorical_option_pair(args: &[Value], idx: usize) -> bool {
    let Some(value) = args.get(idx) else {
        return false;
    };
    let Ok(name) = scalar_text(value, "categorical option") else {
        return false;
    };
    if !(name.eq_ignore_ascii_case("Ordinal") || name.eq_ignore_ascii_case("Protected")) {
        return false;
    }
    args.get(idx + 1)
        .map(|value| categorical_bool_scalar(value, &name).is_ok())
        .unwrap_or(false)
}

fn categorical_bool_scalar(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Int(value) => match value.try_to_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid_argument(format!(
                "categorical: {context} must be logical or numeric 0 or 1"
            ))),
        },
        Value::Num(value) if *value == 0.0 => Ok(false),
        Value::Num(value) if *value == 1.0 => Ok(true),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .numeric_value_at(0)
            .and_then(categorical_numeric_flag)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "categorical: {context} must be logical or numeric 0 or 1"
                ))
            }),
        _ => Err(invalid_argument(format!(
            "categorical: {context} must be logical or numeric 0 or 1"
        ))),
    }
}

fn categorical_numeric_flag(value: NumericScalar) -> Option<bool> {
    match value {
        NumericScalar::F64(0.0) | NumericScalar::F32(0.0) => Some(false),
        NumericScalar::F64(1.0) | NumericScalar::F32(1.0) => Some(true),
        value if value.into_int_value().and_then(|value| value.try_to_u64()) == Some(0) => {
            Some(false)
        }
        value if value.into_int_value().and_then(|value| value.try_to_u64()) == Some(1) => {
            Some(true)
        }
        _ => None,
    }
}

struct CategoryPlan {
    lookup: Vec<String>,
    codes: Vec<usize>,
    categories: Vec<String>,
}

fn categorical_category_plan(
    source: &Value,
    labels: &[String],
    valueset: Option<&Value>,
    catnames: Option<Vec<String>>,
) -> BuiltinResult<CategoryPlan> {
    let source_dtype = categorical_numeric_dtype(source);
    if let Some(valueset) = valueset {
        let valueset_dtype = categorical_numeric_dtype(valueset);
        match (source_dtype, valueset_dtype) {
            (Some(source_dtype), Some(valueset_dtype)) if source_dtype != valueset_dtype => {
                return Err(invalid_argument(
                    "categorical: numeric valueset must have the same class as A",
                ));
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(invalid_argument(
                    "categorical: valueset must have the same class as A",
                ));
            }
            _ => {}
        }
    }
    let valueset_labels = valueset.map(categorical_labels).transpose()?;
    match (valueset, catnames) {
        (None, None) => {
            let mut lookup = labels
                .iter()
                .filter(|label| !label.is_empty())
                .cloned()
                .collect::<Vec<_>>();
            sort_categorical_values(&mut lookup, source_dtype)?;
            lookup.dedup();
            let categories = default_category_names(&lookup, source_dtype)?;
            let codes = (0..categories.len()).collect();
            Ok(CategoryPlan {
                lookup,
                codes,
                categories,
            })
        }
        (Some(_), None) => {
            let valueset = valueset_labels.expect("valueset labels");
            ensure_unique_valueset(&valueset, false)?;
            let categories = default_category_names(&valueset, source_dtype)?;
            let codes = valueset
                .iter()
                .enumerate()
                .map(|(index, _)| index)
                .collect();
            Ok(CategoryPlan {
                lookup: valueset,
                codes,
                categories,
            })
        }
        (Some(_), Some(catnames)) => {
            let valueset = valueset_labels.expect("valueset labels");
            if valueset.len() != catnames.len() {
                return Err(invalid_argument(
                    "categorical: valueset and category names must have the same number of elements",
                ));
            }
            ensure_unique_valueset(&valueset, true)?;
            if catnames.iter().any(|label| label.is_empty()) {
                return Err(invalid_argument(
                    "categorical: category names cannot be empty or missing",
                ));
            }
            let categories = unique_nonempty(catnames.clone())?;
            let codes = catnames
                .iter()
                .map(|label| {
                    categories
                        .iter()
                        .position(|category| category == label)
                        .unwrap_or(0)
                })
                .collect();
            Ok(CategoryPlan {
                lookup: valueset,
                codes,
                categories,
            })
        }
        (None, Some(_)) => Err(invalid_argument(
            "categorical: category names require a valueset argument",
        )),
    }
}

fn ensure_unique_valueset(valueset: &[String], allow_missing: bool) -> BuiltinResult<()> {
    let mut seen = HashSet::new();
    for label in valueset {
        if label.is_empty() && !allow_missing {
            return Err(invalid_argument(
                "categorical: categories cannot be empty or missing",
            ));
        }
        if !seen.insert(label) {
            return Err(invalid_argument(
                "categorical: valueset entries must be unique",
            ));
        }
    }
    Ok(())
}

fn unique_nonempty(labels: Vec<String>) -> BuiltinResult<Vec<String>> {
    let mut out = Vec::new();
    for label in labels {
        if label.is_empty() {
            return Err(invalid_argument(
                "categorical: categories cannot be empty or missing",
            ));
        }
        if !out.contains(&label) {
            out.push(label);
        }
    }
    Ok(out)
}

fn normalize_category_label(value: &str) -> String {
    value.trim().to_string()
}

fn categorical_numeric_dtype(value: &Value) -> Option<runmat_builtins::NumericDType> {
    match value {
        Value::Num(_) => Some(runmat_builtins::NumericDType::F64),
        Value::Int(value) => Some(NumericScalar::from(value.clone()).numeric_dtype()),
        Value::Tensor(tensor) => Some(tensor.numeric_dtype()),
        _ => None,
    }
}

fn sort_categorical_values(
    labels: &mut [String],
    dtype: Option<runmat_builtins::NumericDType>,
) -> BuiltinResult<()> {
    use runmat_builtins::NumericDType;
    match dtype {
        Some(NumericDType::I8 | NumericDType::I16 | NumericDType::I32 | NumericDType::I64) => {
            labels.sort_by_key(|label| label.parse::<i128>().unwrap_or_default());
        }
        Some(NumericDType::U8 | NumericDType::U16 | NumericDType::U32 | NumericDType::U64) => {
            labels.sort_by_key(|label| label.parse::<u128>().unwrap_or_default());
        }
        Some(NumericDType::F32 | NumericDType::F64) => {
            labels.sort_by(|left, right| {
                let left = left.parse::<f64>().unwrap_or(f64::NAN);
                let right = right.parse::<f64>().unwrap_or(f64::NAN);
                left.total_cmp(&right)
            });
        }
        None => labels.sort(),
    }
    Ok(())
}

fn default_category_names(
    lookup: &[String],
    dtype: Option<runmat_builtins::NumericDType>,
) -> BuiltinResult<Vec<String>> {
    let categories = if dtype.is_some() {
        lookup
            .iter()
            .map(|label| {
                label
                    .parse::<f64>()
                    .map(format_five_significant)
                    .unwrap_or_else(|_| label.clone())
            })
            .collect::<Vec<_>>()
    } else {
        lookup.to_vec()
    };
    let mut seen = HashSet::new();
    if categories.iter().any(|category| !seen.insert(category)) {
        return Err(invalid_argument(
            "categorical: numeric values are too closely spaced to create unique default category names; specify category names",
        ));
    }
    Ok(categories)
}

fn format_five_significant(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    if !value.is_finite() {
        return value.to_string();
    }
    let exponent = value.abs().log10().floor() as i32;
    if !(-4..=4).contains(&exponent) {
        let formatted = format!("{value:.4e}");
        let (mantissa, exponent) = formatted.split_once('e').unwrap_or((&formatted, "0"));
        let exponent = exponent.parse::<i32>().unwrap_or_default();
        return format!("{}e{:+}", trim_float(mantissa.to_string()), exponent);
    }
    let decimals = usize::try_from(4 - exponent).unwrap_or_default();
    trim_float(format!("{value:.decimals$}"))
}

fn categorical_numeric_label(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => {
            if value.is_nan() {
                String::new()
            } else {
                format_key_number(value)
            }
        }
        NumericScalar::F32(value) => {
            if value.is_nan() {
                String::new()
            } else {
                format_key_number(f64::from(value))
            }
        }
        value => value
            .into_int_value()
            .map(|value| value.decimal_string())
            .unwrap_or_default(),
    }
}

fn categorical_object_labels(object: &ObjectInstance) -> BuiltinResult<Vec<String>> {
    let categories = categorical_categories(object)?;
    let codes = categorical_codes(object)?;
    (0..codes.len())
        .map(|index| {
            codes
                .numeric_value_at(index)
                .ok_or_else(|| invalid_variable("categorical: invalid numeric code storage"))
                .map(|code| category_label_for_numeric_code(code, &categories).unwrap_or_default())
        })
        .collect()
}

pub(crate) fn categorical_categories(object: &ObjectInstance) -> BuiltinResult<Vec<String>> {
    match object.properties.get("Categories") {
        Some(Value::StringArray(array)) => Ok(array.data.clone()),
        _ => Err(invalid_variable("categorical: missing Categories property")),
    }
}

fn categorical_categories_value(object: &ObjectInstance) -> BuiltinResult<Value> {
    match object.properties.get("Categories") {
        Some(Value::StringArray(array)) => Ok(Value::StringArray(array.clone())),
        _ => Err(invalid_variable("categorical: missing Categories property")),
    }
}

fn categorical_codes(object: &ObjectInstance) -> BuiltinResult<Tensor> {
    match object.properties.get("Codes") {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        _ => Err(invalid_variable("categorical: missing Codes property")),
    }
}

fn categorical_is_ordinal(object: &ObjectInstance) -> bool {
    matches!(object.properties.get("Ordinal"), Some(Value::Bool(true)))
}

fn category_label_for_code(code: f64, categories: &[String]) -> Option<String> {
    if !code.is_finite() || code < 1.0 {
        return None;
    }
    let idx = code as usize;
    if (idx as f64 - code).abs() > f64::EPSILON {
        return None;
    }
    categories.get(idx - 1).cloned()
}

fn category_label_for_integer_code(
    code: &runmat_builtins::IntValue,
    categories: &[String],
) -> Option<String> {
    code.try_to_usize()
        .and_then(|index| index.checked_sub(1))
        .and_then(|index| categories.get(index))
        .cloned()
}

fn category_label_for_numeric_code(code: NumericScalar, categories: &[String]) -> Option<String> {
    match code {
        NumericScalar::F64(code) => category_label_for_code(code, categories),
        NumericScalar::F32(code) => category_label_for_code(f64::from(code), categories),
        code => code
            .into_int_value()
            .and_then(|code| category_label_for_integer_code(&code, categories)),
    }
}

struct CategoricalCompareBase {
    categories: Vec<String>,
}

struct CategoricalCompareOperand {
    codes: Vec<Option<usize>>,
    shape: Vec<usize>,
    ordinal: bool,
}

fn categorical_compare_base(value: &Value) -> Option<CategoricalCompareBase> {
    match value {
        Value::Object(object) if object.is_class(CATEGORICAL_CLASS) => {
            Some(CategoricalCompareBase {
                categories: categorical_categories(object).ok()?,
            })
        }
        _ => None,
    }
}

fn validate_categorical_category_compatibility(lhs: &Value, rhs: &Value) -> BuiltinResult<()> {
    let left = match lhs {
        Value::Object(object) if object.is_class(CATEGORICAL_CLASS) => Some(object),
        _ => None,
    };
    let right = match rhs {
        Value::Object(object) if object.is_class(CATEGORICAL_CLASS) => Some(object),
        _ => None,
    };
    let (Some(left), Some(right)) = (left, right) else {
        return Ok(());
    };
    if categorical_categories(left)? != categorical_categories(right)? {
        return Err(invalid_argument(
            "categorical: categorical arrays must have matching categories",
        ));
    }
    Ok(())
}

fn categorical_compare_operand(
    value: &Value,
    categories: &[String],
) -> BuiltinResult<CategoricalCompareOperand> {
    match value {
        Value::Object(object) if object.is_class(CATEGORICAL_CLASS) => {
            let object_categories = categorical_categories(object)?;
            let codes = categorical_codes(object)?;
            let mapped = (0..codes.len())
                .map(|index| {
                    codes
                        .numeric_value_at(index)
                        .and_then(|code| category_label_for_numeric_code(code, &object_categories))
                        .and_then(|label| categories.iter().position(|category| category == &label))
                })
                .collect();
            Ok(CategoricalCompareOperand {
                codes: mapped,
                shape: codes.shape,
                ordinal: categorical_is_ordinal(object),
            })
        }
        other => {
            let labels = categorical_labels(other)?;
            let shape = value_shape_or_column(other)?;
            Ok(CategoricalCompareOperand {
                codes: labels
                    .iter()
                    .map(|label| categories.iter().position(|category| category == label))
                    .collect(),
                shape,
                ordinal: true,
            })
        }
    }
}

fn compare_category_codes(
    lhs: Option<usize>,
    rhs: Option<usize>,
    comparison: CategoricalComparison,
) -> bool {
    match comparison {
        CategoricalComparison::Eq => lhs.is_some() && rhs.is_some() && lhs == rhs,
        CategoricalComparison::Ne => lhs.is_none() || rhs.is_none() || lhs != rhs,
        CategoricalComparison::Lt => matches!((lhs, rhs), (Some(a), Some(b)) if a < b),
        CategoricalComparison::Le => matches!((lhs, rhs), (Some(a), Some(b)) if a <= b),
        CategoricalComparison::Gt => matches!((lhs, rhs), (Some(a), Some(b)) if a > b),
        CategoricalComparison::Ge => matches!((lhs, rhs), (Some(a), Some(b)) if a >= b),
    }
}

fn categorical_logical_result(data: Vec<u8>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if crate::builtins::common::tensor::element_count(&shape) <= 1 && data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(invalid_variable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    fn integer_storages(values: &[u64]) -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(values.iter().map(|&value| value as i8).collect()),
            IntegerStorage::I16(values.iter().map(|&value| value as i16).collect()),
            IntegerStorage::I32(values.iter().map(|&value| value as i32).collect()),
            IntegerStorage::I64(values.iter().map(|&value| value as i64).collect()),
            IntegerStorage::U8(values.iter().map(|&value| value as u8).collect()),
            IntegerStorage::U16(values.iter().map(|&value| value as u16).collect()),
            IntegerStorage::U32(values.iter().map(|&value| value as u32).collect()),
            IntegerStorage::U64(values.to_vec()),
        ]
    }

    #[test]
    fn categorical_metadata_preserves_every_integer_class() {
        for storage in integer_storages(&[2]) {
            let values = Tensor::new_integer(storage.clone(), vec![1, 1]).unwrap();
            assert_eq!(
                categorical_labels(&Value::Tensor(values)).unwrap(),
                vec!["2"]
            );

            let codes = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let mut object = ObjectInstance::new(CATEGORICAL_CLASS.to_string());
            object
                .properties
                .insert("Codes".to_string(), Value::Tensor(codes));
            object.properties.insert(
                "Categories".to_string(),
                Value::StringArray(
                    StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
                ),
            );
            assert_eq!(categorical_object_labels(&object).unwrap(), vec!["two"]);
            assert!(matches!(
                categorical_compare(
                    &Value::Object(object),
                    &Value::String("two".to_string()),
                    CategoricalComparison::Eq,
                )
                .unwrap()
                .unwrap(),
                Value::Bool(true)
            ));
        }
    }

    #[test]
    fn categorical_metadata_reads_native_single_values_and_codes() {
        let values = Tensor::from_f32(vec![1.25, f32::NAN], vec![1, 2]).unwrap();
        assert_eq!(
            categorical_labels(&Value::Tensor(values)).unwrap(),
            vec!["1.25", ""]
        );

        let codes = Tensor::from_f32(vec![2.0, f32::NAN, 1.5], vec![1, 3]).unwrap();
        let mut object = ObjectInstance::new(CATEGORICAL_CLASS.to_string());
        object
            .properties
            .insert("Codes".to_string(), Value::Tensor(codes));
        object.properties.insert(
            "Categories".to_string(),
            Value::StringArray(
                StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
            ),
        );
        assert_eq!(
            categorical_object_labels(&object).unwrap(),
            vec!["two", "", ""]
        );
    }
}
