use super::*;
use crate::builtins::table::display::trim_float;
use runmat_value::{NumericScalar, StringElement};

mod labels;
pub(crate) use self::labels::categorical_label_view;

pub(crate) fn categorical_from_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    if args.len() == 1
        && matches!(args.first(),Some(Value::Object(object)) if object.is_class(CATEGORICAL_CLASS))
    {
        return Ok(args.first().expect("one categorical source").clone());
    }
    let empty;
    let source = match args.first() {
        Some(value) => value,
        None => {
            empty = Value::StringArray(
                StringArray::new(Vec::new(), vec![0, 0]).map_err(invalid_variable)?,
            );
            &empty
        }
    };
    let rest = args.range(args.len().min(1)..)?;
    let parsed = parse_categorical_args(rest)?;
    let labels = categorical_labels(source)?;
    let category_plan =
        categorical_category_plan(source, &labels, parsed.valueset, parsed.catnames)?;
    categorical_object_from_plan(
        source,
        labels,
        category_plan,
        parsed.ordinal,
        parsed.protected.unwrap_or(false),
    )
}

fn categorical_object_from_plan(
    source: &Value,
    labels: Vec<StringElement>,
    category_plan: CategoryPlan,
    ordinal: bool,
    protected: bool,
) -> BuiltinResult<Value> {
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
    let mut object =
        runmat_value::UnboundedObjectBuilder::with_property_capacity(CATEGORICAL_CLASS, 4);
    object.insert(
        "Codes".to_string(),
        Value::Tensor(
            Tensor::new(codes, value_shape_or_column(source)?).map_err(invalid_variable)?,
        ),
    );
    object.insert(
        "Categories".to_string(),
        Value::StringArray(
            StringArray::new(
                category_plan.categories.clone(),
                vec![1, category_plan.categories.len()],
            )
            .map_err(invalid_variable)?,
        ),
    );
    object.insert("Ordinal".to_string(), Value::Bool(ordinal));
    object.insert("Protected".to_string(), Value::Bool(ordinal || protected));
    Ok(Value::Object(object.finish()))
}

pub(in crate::builtins::table) fn ordinal_from_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(invalid_argument("ordinal: X is required"));
    }
    if args.len() == 1 {
        let source = args.first().expect("one source");
        let labels = categorical_labels(source)?;
        let plan = categorical_category_plan(source, &labels, None, None)?;
        return categorical_object_from_plan(source, labels, plan, true, true);
    }
    if args.len() > 4 {
        return Err(invalid_argument(
            "ordinal: expected X, optional labels, optional levels, or empty levels plus edges",
        ));
    }
    let source = &args[0];
    let category_names = categorical_category_names(&args[1])?;
    match args.len() {
        2 => ordinal_from_inferred_levels(source, category_names),
        3 => ordinal_from_explicit_levels(source, &args[2], category_names),
        4 => ordinal_from_edges(source, &args[2], &args[3], category_names),
        _ => unreachable!("ordinal arity checked above"),
    }
}

fn ordinal_from_inferred_levels(
    source: &Value,
    category_names: Vec<String>,
) -> BuiltinResult<Value> {
    let labels = categorical_labels(source)?;
    let mut lookup = labels
        .iter()
        .filter_map(|label| {
            label
                .as_str()
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    sort_categorical_values(&mut lookup, categorical_numeric_dtype(source))?;
    lookup.dedup();
    let plan = ordinal_category_plan(lookup.into_iter().map(Into::into).collect(), category_names)?;
    categorical_object_from_plan(source, labels, plan, true, true)
}

fn ordinal_from_explicit_levels(
    source: &Value,
    levels: &Value,
    category_names: Vec<String>,
) -> BuiltinResult<Value> {
    ensure_ordinal_vector(levels, "levels")?;
    let labels = categorical_labels(source)?;
    let lookup = categorical_labels(levels)?;
    ensure_unique_valueset(&lookup, true)?;
    let plan = ordinal_category_plan(lookup, category_names)?;
    categorical_object_from_plan(source, labels, plan, true, true)
}

fn ordinal_category_plan(
    lookup: Vec<StringElement>,
    category_names: Vec<String>,
) -> BuiltinResult<CategoryPlan> {
    if lookup.len() != category_names.len() {
        return Err(invalid_argument(
            "ordinal: labels must contain one entry for every level",
        ));
    }
    let categories = unique_nonempty(category_names.clone())?;
    let codes = category_names
        .iter()
        .map(|label| {
            categories
                .iter()
                .position(|category| category == label)
                .expect("validated nonempty category name")
        })
        .collect();
    Ok(CategoryPlan {
        lookup,
        codes,
        categories,
    })
}

fn ordinal_from_edges(
    source: &Value,
    empty_levels: &Value,
    edges: &Value,
    category_names: Vec<String>,
) -> BuiltinResult<Value> {
    if value_element_count(empty_levels) != Some(0) {
        return Err(invalid_argument(
            "ordinal: the third argument must be an empty levels array when edges are supplied",
        ));
    }
    ensure_ordinal_vector(edges, "edges")?;
    let source_values = ordinal_numeric_values(source, "X")?;
    let edge_values = ordinal_numeric_values(edges, "edges")?;
    if edge_values.len() != category_names.len().saturating_add(1) {
        return Err(invalid_argument(
            "ordinal: edges must contain one more element than labels",
        ));
    }
    use crate::builtins::logical::rel::integer_comparison::compare_numeric_scalars_exact;
    if edge_values.windows(2).any(|pair| {
        compare_numeric_scalars_exact(pair[0], pair[1]) != Some(std::cmp::Ordering::Less)
    }) {
        return Err(invalid_argument(
            "ordinal: edges must be finite and strictly increasing",
        ));
    }
    let categories = unique_nonempty(category_names.clone())?;
    let bin_codes = category_names
        .iter()
        .map(|label| {
            categories
                .iter()
                .position(|category| category == label)
                .expect("validated nonempty ordinal label")
        })
        .collect::<Vec<_>>();
    let mut codes = Vec::with_capacity(source_values.len());
    for value in source_values {
        let mut selected = None;
        for bin in 0..category_names.len() {
            let lower = compare_numeric_scalars_exact(value, edge_values[bin]);
            let upper = compare_numeric_scalars_exact(value, edge_values[bin + 1]);
            let lower_inclusive = matches!(
                lower,
                Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater)
            );
            let upper_inclusive =
                bin + 1 == category_names.len() && upper == Some(std::cmp::Ordering::Equal);
            if lower_inclusive && (upper == Some(std::cmp::Ordering::Less) || upper_inclusive) {
                selected = Some(bin_codes[bin]);
                break;
            }
        }
        codes.push(selected.map_or(f64::NAN, |code| (code + 1) as f64));
    }
    categorical_object_from_codes(source, codes, categories, true, true)
}

fn categorical_object_from_codes(
    source: &Value,
    codes: Vec<f64>,
    categories: Vec<String>,
    ordinal: bool,
    protected: bool,
) -> BuiltinResult<Value> {
    let mut object =
        runmat_value::UnboundedObjectBuilder::with_property_capacity(CATEGORICAL_CLASS, 4);
    object.insert(
        "Codes".to_string(),
        Value::Tensor(
            Tensor::new(codes, value_shape_or_column(source)?).map_err(invalid_variable)?,
        ),
    );
    object.insert(
        "Categories".to_string(),
        Value::StringArray(
            StringArray::new(categories.clone(), vec![1, categories.len()])
                .map_err(invalid_variable)?,
        ),
    );
    object.insert("Ordinal".to_string(), Value::Bool(ordinal));
    object.insert("Protected".to_string(), Value::Bool(ordinal || protected));
    Ok(Value::Object(object.finish()))
}

fn ordinal_numeric_values(value: &Value, context: &str) -> BuiltinResult<Vec<NumericScalar>> {
    match value {
        Value::Num(value) => Ok(vec![NumericScalar::F64(*value)]),
        Value::Int(value) => Ok(vec![NumericScalar::from(value.clone())]),
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| {
                tensor.numeric_value_at(index).ok_or_else(|| {
                    invalid_argument(format!("ordinal: {context} has invalid numeric storage"))
                })
            })
            .collect(),
        _ => Err(invalid_argument(format!(
            "ordinal: {context} must be a real numeric array"
        ))),
    }
}

fn ensure_ordinal_vector(value: &Value, context: &str) -> BuiltinResult<()> {
    let shape = value_shape_or_column(value)?;
    let non_singleton = shape.iter().filter(|&&dimension| dimension > 1).count();
    if non_singleton > 1 {
        return Err(invalid_argument(format!(
            "ordinal: {context} must be a vector"
        )));
    }
    Ok(())
}

fn value_element_count(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(tensor) => Some(tensor.len()),
        Value::StringArray(array) => Some(array.data().len()),
        Value::LogicalArray(array) => Some(array.data.len()),
        Value::Cell(cell) => Some(cell.data().len()),
        _ => None,
    }
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
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.into_value()]));
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
    rest: crate::call::FunctionArgumentView<'_>,
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
    rest: crate::call::FunctionArgumentView<'_>,
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
    rest: crate::call::FunctionArgumentView<'_>,
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
        .properties()
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
    let mut object =
        runmat_value::UnboundedObjectBuilder::with_property_capacity(CATEGORICAL_CLASS, 4);
    object.insert("Codes".to_string(), Value::Tensor(codes));
    object.insert("Categories".to_string(), categories);
    object.insert("Ordinal".to_string(), Value::Bool(true));
    object.insert("Protected".to_string(), protected);
    Ok(Value::Object(object.finish()))
}

pub(crate) fn categorical_labels(value: &Value) -> BuiltinResult<Vec<StringElement>> {
    match value {
        Value::Num(value) => Ok(vec![categorical_numeric_label(NumericScalar::F64(*value))]),
        Value::String(text) => Ok(vec![normalize_category_element(text)]),
        Value::StringArray(array) => Ok(array
            .data()
            .iter()
            .map(normalize_category_element)
            .collect()),
        Value::CharArray(array) => Ok(char_rows(array)?
            .into_iter()
            .map(|value| StringElement::text(normalize_category_label(&value)))
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
            .map(|flag| StringElement::text(if *flag != 0 { "true" } else { "false" }))
            .collect()),
        Value::Cell(cell) => cell.iter_column_major().map(cell_scalar_label).collect(),
        Value::Object(object) if object.is_class(CATEGORICAL_CLASS) => {
            categorical_observation_labels(object).map(|labels| {
                labels
                    .into_iter()
                    .map(|label| {
                        label
                            .map(StringElement::text)
                            .unwrap_or(StringElement::Missing)
                    })
                    .collect()
            })
        }
        other => Ok(vec![StringElement::text(other.to_string())]),
    }
}

pub(in crate::builtins::table) fn cell_scalar_label(value: &Value) -> BuiltinResult<StringElement> {
    match value {
        Value::String(text) => Ok(normalize_category_element(text)),
        Value::StringArray(array) if array.data().len() == 1 => {
            Ok(normalize_category_element(&array.data()[0]))
        }
        Value::CharArray(array) if array.rows == 1 => {
            let text = runmat_value::decode_utf16_code_units(array.data()).map_err(|error| {
                table_error_with_source(
                    &TABLE_ERROR_INVALID_VARIABLE,
                    format!("categorical: label is not valid UTF-16 text: {error}"),
                    error,
                )
            })?;
            Ok(StringElement::text(normalize_category_label(&text)))
        }
        Value::Num(value) if value.is_nan() => Ok(StringElement::Missing),
        Value::Num(value) => Ok(StringElement::text(format_key_number(*value))),
        Value::Bool(value) => Ok(StringElement::text(if *value { "true" } else { "false" })),
        other => Ok(StringElement::text(other.to_string())),
    }
}

pub(in crate::builtins::table) fn value_shape_or_column(
    value: &Value,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.shape().to_vec()),
        Value::StringArray(array) => Ok(array.shape().to_vec()),
        Value::LogicalArray(array) => Ok(array.shape.to_vec()),
        Value::Cell(cell) => Ok(cell.shape().to_vec()),
        Value::CharArray(array) => Ok(vec![array.rows, 1]),
        Value::Object(object) if object.is_class(CATEGORICAL_CLASS) => {
            categorical_codes(object).map(|codes| codes.shape().to_vec())
        }
        _ => Ok(vec![1, 1]),
    }
}

#[derive(Default)]
struct CategoricalArgs<'a> {
    valueset: Option<&'a Value>,
    catnames: Option<Vec<String>>,
    ordinal: bool,
    protected: Option<bool>,
}

fn parse_categorical_args<'a>(
    args: crate::call::FunctionArgumentView<'a>,
) -> BuiltinResult<CategoricalArgs<'a>> {
    let mut parsed = CategoricalArgs::default();
    let mut idx = 0usize;
    if idx < args.len() && !is_categorical_option_pair(args, idx) {
        parsed.valueset = args.get(idx);
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
        match runmat_builtins::CategoricalOption::parse(&option_name) {
            Some(runmat_builtins::CategoricalOption::Ordinal) => {
                parsed.ordinal = categorical_bool_scalar(&args[idx + 1], "Ordinal")?
            }
            Some(runmat_builtins::CategoricalOption::Protected) => {
                parsed.protected = Some(categorical_bool_scalar(&args[idx + 1], "Protected")?)
            }
            None => {
                return Err(invalid_argument(format!(
                    "categorical: unsupported option '{option_name}'"
                )))
            }
        }
        idx += 2;
    }
    Ok(parsed)
}

fn categorical_category_names(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => categorical_labels(value)?.into_iter().map(category_name_from_element).collect(),
        Value::Cell(cell) => cell
            .iter_column_major()
            .map(|value| match value {
                Value::String(_) | Value::CharArray(_) => cell_scalar_label(value).and_then(category_name_from_element),
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

fn is_categorical_option_pair(args: crate::call::FunctionArgumentView<'_>, idx: usize) -> bool {
    let Some(value) = args.get(idx) else {
        return false;
    };
    let Ok(name) = scalar_text(value, "categorical option") else {
        return false;
    };
    if runmat_builtins::CategoricalOption::parse(&name).is_none() {
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
    lookup: Vec<StringElement>,
    codes: Vec<usize>,
    categories: Vec<String>,
}

fn categorical_category_plan(
    source: &Value,
    labels: &[StringElement],
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
                .filter_map(|label| {
                    label
                        .as_str()
                        .filter(|text| !text.is_empty())
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>();
            sort_categorical_values(&mut lookup, source_dtype)?;
            lookup.dedup();
            let categories = default_category_names(&lookup, source_dtype)?;
            let codes = (0..categories.len()).collect();
            Ok(CategoryPlan {
                lookup: lookup.into_iter().map(StringElement::text).collect(),
                codes,
                categories,
            })
        }
        (Some(_), None) => {
            let valueset = valueset_labels.expect("valueset labels");
            ensure_unique_valueset(&valueset, false)?;
            let present_valueset = valueset
                .iter()
                .map(|label| {
                    label
                        .as_str()
                        .expect("validated nonmissing valueset")
                        .to_owned()
                })
                .collect::<Vec<_>>();
            let categories = default_category_names(&present_valueset, source_dtype)?;
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

fn ensure_unique_valueset(valueset: &[StringElement], allow_missing: bool) -> BuiltinResult<()> {
    let mut seen = HashSet::new();
    for label in valueset {
        if label.as_str().is_none_or(str::is_empty) && !allow_missing {
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

fn categorical_numeric_dtype(value: &Value) -> Option<runmat_value::NumericDType> {
    match value {
        Value::Num(_) => Some(runmat_value::NumericDType::F64),
        Value::Int(value) => Some(NumericScalar::from(value.clone()).numeric_dtype()),
        Value::Tensor(tensor) => Some(tensor.numeric_dtype()),
        _ => None,
    }
}

fn sort_categorical_values(
    labels: &mut [String],
    dtype: Option<runmat_value::NumericDType>,
) -> BuiltinResult<()> {
    use runmat_value::NumericDType;
    match dtype {
        Some(NumericDType::I8 | NumericDType::I16 | NumericDType::I32 | NumericDType::I64) => {
            labels.sort_by_key(|label| label.parse::<i128>().unwrap_or_default());
        }
        Some(NumericDType::U8 | NumericDType::U16 | NumericDType::U32 | NumericDType::U64) => {
            labels.sort_by_key(|label| label.parse::<u128>().unwrap_or_default());
        }
        Some(NumericDType::F32) => {
            labels.sort_by(|left, right| {
                let left = left.parse::<f32>().unwrap_or(f32::NAN);
                let right = right.parse::<f32>().unwrap_or(f32::NAN);
                left.total_cmp(&right)
            });
        }
        Some(NumericDType::F64) => {
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
    dtype: Option<runmat_value::NumericDType>,
) -> BuiltinResult<Vec<String>> {
    let categories = if dtype.is_some() {
        lookup
            .iter()
            .map(|label| {
                if dtype == Some(runmat_value::NumericDType::F32) {
                    // Lookup text round-trips the original single. Restore
                    // that class before the unchanged five-digit name policy.
                    return label
                        .parse::<f32>()
                        .map(|value| format_five_significant(f64::from(value)))
                        .unwrap_or_else(|_| label.clone());
                }
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

fn categorical_numeric_label(value: NumericScalar) -> StringElement {
    if matches!(value, NumericScalar::F64(value) if value.is_nan())
        || matches!(value, NumericScalar::F32(value) if value.is_nan())
    {
        return StringElement::Missing;
    }
    StringElement::text(match value {
        // Private identity must round-trip independently of rounded category names.
        NumericScalar::F64(value) => value.to_string(),
        NumericScalar::F32(value) => value.to_string(),
        value => value
            .into_int_value()
            .expect("nonfloating numeric category is integer")
            .decimal_string(),
    })
}

fn normalize_category_element(value: &StringElement) -> StringElement {
    match value.as_str() {
        Some(text) if text.trim() == text => value.clone(),
        Some(value) => StringElement::text(normalize_category_label(value)),
        None => StringElement::Missing,
    }
}

fn category_name_from_element(value: StringElement) -> BuiltinResult<String> {
    value
        .into_text()
        .map(|text| text.as_str().to_owned())
        .filter(|text| !text.is_empty())
        .ok_or_else(|| invalid_argument("categorical: category names cannot be empty or missing"))
}

#[cfg(test)]
fn categorical_object_labels(object: &ObjectInstance) -> BuiltinResult<Vec<String>> {
    let labels = categorical_label_view(object)?;
    (0..labels.len())
        .map(|index| labels.label_at(index).map(str::to_owned))
        .collect()
}

pub(crate) fn categorical_observation_labels(
    object: &ObjectInstance,
) -> BuiltinResult<Vec<Option<String>>> {
    let view = categorical_label_view(object)?;
    (0..view.len())
        .map(|index| {
            view.observation_label_at(index)
                .map(|label| label.map(str::to_owned))
        })
        .collect()
}

pub(crate) fn categorical_declared_levels(object: &ObjectInstance) -> BuiltinResult<Value> {
    let count = categorical_categories(object)?.len();
    let codes = Tensor::new(
        (1..=count).map(|code| code as f64).collect(),
        vec![count, 1],
    )
    .map_err(invalid_variable)?;
    let mut levels = object.clone();
    crate::execution::object::replacement::replace_property(
        None,
        &mut levels,
        "Codes",
        Value::Tensor(codes),
        |_, source| {
            invalid_variable(format!(
                "categorical: replacement admission failed: {source}"
            ))
        },
        invalid_variable,
    )?;
    Ok(Value::Object(levels))
}

pub(crate) fn categorical_levels_for_observations(
    object: &ObjectInstance,
    labels: &[Option<String>],
) -> BuiltinResult<Value> {
    let categories = categorical_categories(object)?;
    let codes = labels
        .iter()
        .map(|label| {
            let Some(label) = label else {
                return Ok(f64::NAN);
            };
            categories
                .iter()
                .position(|category| category == label)
                .map(|index| index as f64 + 1.0)
                .ok_or_else(|| {
                    invalid_variable("categorical: group label is not a declared category")
                })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let codes = Tensor::new(codes, vec![labels.len(), 1]).map_err(invalid_variable)?;
    let mut levels = object.clone();
    crate::execution::object::replacement::replace_property(
        None,
        &mut levels,
        "Codes",
        Value::Tensor(codes),
        |_, source| {
            invalid_variable(format!(
                "categorical: replacement admission failed: {source}"
            ))
        },
        invalid_variable,
    )?;
    Ok(Value::Object(levels))
}

pub(crate) fn categorical_categories(object: &ObjectInstance) -> BuiltinResult<Vec<String>> {
    match object.properties().get("Categories") {
        Some(Value::StringArray(array)) => array
            .data()
            .iter()
            .map(|text| {
                text.as_str().map(str::to_owned).ok_or_else(|| {
                    invalid_variable("categorical: Categories must contain nonmissing strings")
                })
            })
            .collect(),
        _ => Err(invalid_variable("categorical: missing Categories property")),
    }
}

fn categorical_categories_value(object: &ObjectInstance) -> BuiltinResult<Value> {
    match object.properties().get("Categories") {
        Some(Value::StringArray(array)) => Ok(Value::StringArray(array.clone())),
        _ => Err(invalid_variable("categorical: missing Categories property")),
    }
}

fn categorical_codes(object: &ObjectInstance) -> BuiltinResult<Tensor> {
    match object.properties().get("Codes") {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        _ => Err(invalid_variable("categorical: missing Codes property")),
    }
}

fn categorical_is_ordinal(object: &ObjectInstance) -> bool {
    matches!(object.properties().get("Ordinal"), Some(Value::Bool(true)))
}

trait CategoryText {
    fn category_text(&self) -> Option<&str>;
}
impl CategoryText for String {
    fn category_text(&self) -> Option<&str> {
        Some(self.as_str())
    }
}
impl CategoryText for runmat_value::StringElement {
    fn category_text(&self) -> Option<&str> {
        self.as_str()
    }
}

fn category_label_for_code<S: CategoryText>(code: f64, categories: &[S]) -> Option<&str> {
    if !code.is_finite() || code < 1.0 {
        return None;
    }
    let idx = code as usize;
    if (idx as f64 - code).abs() > f64::EPSILON {
        return None;
    }
    categories
        .get(idx - 1)
        .and_then(CategoryText::category_text)
}

fn category_label_for_integer_code<'a, S: CategoryText>(
    code: &runmat_value::IntValue,
    categories: &'a [S],
) -> Option<&'a str> {
    code.try_to_usize()
        .and_then(|index| index.checked_sub(1))
        .and_then(|index| categories.get(index))
        .and_then(CategoryText::category_text)
}

fn category_label_ref_for_numeric_code<S: CategoryText>(
    code: NumericScalar,
    categories: &[S],
) -> Option<&str> {
    match code {
        NumericScalar::F64(code) => category_label_for_code(code, categories),
        NumericScalar::F32(code) => category_label_for_code(f64::from(code), categories),
        code => code
            .into_int_value()
            .and_then(|code| category_label_for_integer_code(&code, categories)),
    }
}

fn category_label_for_numeric_code(code: NumericScalar, categories: &[String]) -> Option<String> {
    category_label_ref_for_numeric_code(code, categories).map(str::to_owned)
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
                shape: codes.shape().to_vec(),
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
#[path = "categorical/numeric_identity_tests.rs"]
mod numeric_identity_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

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
            let mut object =
                runmat_value::UnboundedObjectBuilder::with_property_capacity(CATEGORICAL_CLASS, 2);
            object.insert("Codes".to_string(), Value::Tensor(codes));
            object.insert(
                "Categories".to_string(),
                Value::StringArray(
                    StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
                ),
            );
            let object = object.finish();
            assert_eq!(categorical_object_labels(&object).unwrap(), vec!["two"]);
            assert!(matches!(
                categorical_compare(
                    &Value::Object(object),
                    &Value::String("two".to_string().into()),
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
            vec!["1.25".into(), runmat_value::StringElement::Missing]
        );

        let codes = Tensor::from_f32(vec![2.0, f32::NAN, 1.5], vec![1, 3]).unwrap();
        let mut object =
            runmat_value::UnboundedObjectBuilder::with_property_capacity(CATEGORICAL_CLASS, 2);
        object.insert("Codes".to_string(), Value::Tensor(codes));
        object.insert(
            "Categories".to_string(),
            Value::StringArray(
                StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
            ),
        );
        let object = object.finish();
        assert_eq!(
            categorical_object_labels(&object).unwrap(),
            vec!["two", "", ""]
        );
    }
    #[test]
    fn empty_cell_geometry_and_categorical_copy_preserve_codes_and_metadata() {
        use crate::call::FunctionArgumentView;
        let Value::Object(empty) =
            categorical_from_args(FunctionArgumentView::from_slice(&[])).unwrap()
        else {
            panic!("categorical")
        };
        let Value::Tensor(codes) = &empty.properties()["Codes"] else {
            panic!("codes")
        };
        assert_eq!(codes.shape(), &[0, 0]);
        let source = Value::Cell(
            CellArray::new_with_shape(
                ["a", "b", "c", "d", "e", "f", "g", "h"]
                    .into_iter()
                    .map(Value::from)
                    .collect(),
                vec![2, 2, 2],
            )
            .unwrap(),
        );
        let names = Value::StringArray(
            StringArray::new(
                ["a", "b", "c", "d", "e", "f", "g", "h", "unused"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                vec![1, 9],
            )
            .unwrap(),
        );
        let args = [source, names, Value::from("Ordinal"), Value::Bool(true)];
        let original = categorical_from_args(FunctionArgumentView::from_slice(&args)).unwrap();
        let Value::Object(object) = &original else {
            panic!("categorical")
        };
        let Value::Tensor(codes) = &object.properties()["Codes"] else {
            panic!("codes")
        };
        assert_eq!(codes.shape(), &[2, 2, 2]);
        assert_eq!(
            codes.materialize_f64(),
            vec![1., 3., 2., 4., 5., 7., 6., 8.]
        );
        let copy = categorical_from_args(FunctionArgumentView::from_slice(std::slice::from_ref(
            &original,
        )))
        .unwrap();
        assert_eq!(copy, original);
    }
}
