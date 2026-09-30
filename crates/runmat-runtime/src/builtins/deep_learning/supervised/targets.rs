use super::*;

#[derive(Clone)]
pub(super) enum Task {
    Regression(Matrix),
    Classification(ClassificationTarget),
}

impl Task {
    pub(super) fn select_rows(
        &self,
        rows: &[usize],
        function: &'static str,
    ) -> BuiltinResult<Self> {
        match self {
            Self::Regression(targets) => targets.select_rows(rows, function).map(Self::Regression),
            Self::Classification(targets) => {
                Ok(Self::Classification(targets.select_rows(rows, function)?))
            }
        }
    }

    pub(super) fn loss_name(&self) -> &'static str {
        match self {
            Self::Regression(_) => "mse",
            Self::Classification(_) => "crossentropy",
        }
    }
}

#[derive(Clone)]
pub(super) struct ClassificationTarget {
    pub(super) classes: Vec<String>,
    pub(super) one_hot: Matrix,
}

impl ClassificationTarget {
    pub(super) fn select_rows(
        &self,
        rows: &[usize],
        function: &'static str,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            classes: self.classes.clone(),
            one_hot: self.one_hot.select_rows(rows, function)?,
        })
    }

    pub(super) fn classes_value(&self, function: &'static str) -> BuiltinResult<Value> {
        string_array(self.classes.clone(), vec![1, self.classes.len()], function)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LossKind {
    Regression,
    Classification,
}

pub(super) fn infer_train_network_task(
    layers: &[Value],
    y: &Value,
    observations: usize,
) -> BuiltinResult<Task> {
    if layers.iter().any(|layer| {
        matches!(
            layer,
            Value::Object(object) if object.class_name.is(super::super::CLASSIFICATION_OUTPUT_LAYER_CLASS)
        )
    }) {
        classification_target(y, observations, "trainNetwork").map(Task::Classification)
    } else if layers.iter().any(|layer| {
        matches!(
            layer,
            Value::Object(object) if object.class_name.is(super::super::REGRESSION_OUTPUT_LAYER_CLASS)
        )
    }) {
        numeric_matrix_ref(y, "trainNetwork", "Y").and_then(|target| {
            validate_target_rows(&target, observations, "trainNetwork")?;
            Ok(Task::Regression(target))
        })
    } else if is_label_target(y) {
        classification_target(y, observations, "trainNetwork").map(Task::Classification)
    } else {
        numeric_matrix_ref(y, "trainNetwork", "Y").and_then(|target| {
            validate_target_rows(&target, observations, "trainNetwork")?;
            Ok(Task::Regression(target))
        })
    }
}

pub(super) fn task_from_loss(
    loss: LossKind,
    y: &Value,
    observations: usize,
) -> BuiltinResult<Task> {
    match loss {
        LossKind::Regression => numeric_matrix_ref(y, "trainnet", "Y").and_then(|target| {
            validate_target_rows(&target, observations, "trainnet")?;
            Ok(Task::Regression(target))
        }),
        LossKind::Classification => {
            classification_target(y, observations, "trainnet").map(Task::Classification)
        }
    }
}

pub(super) fn parse_trainnet_loss(value: &Value) -> BuiltinResult<LossKind> {
    if is_function_handle(value) {
        return Err(deep_learning_error(
            "trainnet",
            "trainnet: custom loss function handles require autodiff infrastructure and are not supported in this training slice",
        ));
    }
    match scalar_text(value, "trainnet")?
        .to_ascii_lowercase()
        .as_str()
    {
        "mse" | "mean-squared-error" | "mean_squared_error" => Ok(LossKind::Regression),
        "crossentropy" | "cross-entropy" => Ok(LossKind::Classification),
        other => Err(deep_learning_error(
            "trainnet",
            format!("trainnet: unsupported loss function '{other}'"),
        )),
    }
}

fn is_function_handle(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}

fn validate_target_rows(
    target: &Matrix,
    observations: usize,
    function: &'static str,
) -> BuiltinResult<()> {
    if target.rows != observations {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: Y must have one row per observation; expected {observations}, got {}",
                target.rows
            ),
        ));
    }
    Ok(())
}

fn is_label_target(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => true,
        Value::Cell(cell) => cell.data.iter().all(is_label_target),
        Value::Object(object) => object.class_name.is(runmat_types::standard::CATEGORICAL),
        _ => false,
    }
}

fn classification_target(
    value: &Value,
    observations: usize,
    function: &'static str,
) -> BuiltinResult<ClassificationTarget> {
    if let Some(labels) = label_strings(value, function)? {
        if labels.len() != observations {
            return Err(deep_learning_error(
                function,
                format!(
                    "{function}: classification labels must have one label per observation; expected {observations}, got {}",
                    labels.len()
                ),
            ));
        }
        return one_hot_from_labels(labels, function);
    }
    let target = numeric_matrix_ref(value, function, "Y")?;
    validate_target_rows(&target, observations, function)?;
    if target.cols > 1 {
        let classes = (1..=target.cols)
            .map(|idx| idx.to_string())
            .collect::<Vec<_>>();
        return Ok(ClassificationTarget {
            classes,
            one_hot: target,
        });
    }
    let labels = target
        .data
        .iter()
        .map(|value| {
            if !value.is_finite() || *value < 1.0 || value.fract().abs() > f64::EPSILON {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: numeric classification labels must be positive integers"),
                ));
            }
            Ok(format!("{}", *value as usize))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    one_hot_from_labels(labels, function)
}

fn label_strings(value: &Value, function: &'static str) -> BuiltinResult<Option<Vec<String>>> {
    match value {
        Value::Int(code) => {
            if code.to_i128() < 1 {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: numeric classification labels must be positive integers"),
                ));
            }
            Ok(Some(vec![code.decimal_string()]))
        }
        Value::Tensor(codes)
            if codes.numeric_dtype().integer_class().is_some()
                && codes.shape.len() <= 2
                && codes.cols == 1 =>
        {
            let mut labels = Vec::with_capacity(codes.len());
            for index in 0..codes.len() {
                let code = codes
                    .numeric_value_at(index)
                    .and_then(NumericScalar::into_int_value)
                    .ok_or_else(|| {
                        deep_learning_error(
                            function,
                            format!("{function}: numeric classification label storage is invalid"),
                        )
                    })?;
                if code.to_i128() < 1 {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: numeric classification labels must be positive integers"
                        ),
                    ));
                }
                labels.push(code.decimal_string());
            }
            Ok(Some(labels))
        }
        Value::String(text) => Ok(Some(vec![text.clone()])),
        Value::CharArray(chars) if chars.rows == 1 => Ok(Some(vec![chars.data.iter().collect()])),
        Value::StringArray(array) => Ok(Some(array.data.clone())),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for item in &cell.data {
                labels.push(scalar_text(item, function)?);
            }
            Ok(Some(labels))
        }
        Value::Object(object) if object.class_name.is(runmat_types::standard::CATEGORICAL) => {
            let categories = match object.properties.get("Categories") {
                Some(Value::StringArray(categories)) => categories.data.clone(),
                _ => {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: categorical labels are missing Categories"),
                    ));
                }
            };
            let codes = match object.properties.get("Codes") {
                Some(Value::Tensor(codes)) => codes,
                _ => {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: categorical labels are missing numeric Codes"),
                    ));
                }
            };
            let mut labels = Vec::with_capacity(codes.len());
            for index in 0..codes.len() {
                let code = codes.numeric_value_at(index).ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!("{function}: categorical Codes storage is invalid"),
                    )
                })?;
                let idx = categorical_code_index(code, function)?;
                let Some(label) = categories.get(idx) else {
                    return Err(deep_learning_error(
                        function,
                        format!("{function}: categorical Codes reference an unknown category"),
                    ));
                };
                labels.push(label.clone());
            }
            Ok(Some(labels))
        }
        _ => Ok(None),
    }
}

fn categorical_code_index(code: NumericScalar, function: &'static str) -> BuiltinResult<usize> {
    match code {
        NumericScalar::F64(code) => {
            if code.is_nan() {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: classification labels must not be missing"),
                ));
            }
            if code < 1.0 || code.fract().abs() > f64::EPSILON {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: categorical Codes must be positive integers"),
                ));
            }
            Ok(code as usize - 1)
        }
        NumericScalar::F32(code) => {
            categorical_code_index(NumericScalar::F64(f64::from(code)), function)
        }
        code => code
            .into_int_value()
            .and_then(|code| code.try_to_usize())
            .filter(|code| *code >= 1)
            .map(|code| code - 1)
            .ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: categorical Codes must be positive integers"),
                )
            }),
    }
}

fn one_hot_from_labels(
    labels: Vec<String>,
    function: &'static str,
) -> BuiltinResult<ClassificationTarget> {
    let mut classes = labels.clone();
    classes.sort();
    classes.dedup();
    if classes.is_empty() {
        return Err(deep_learning_error(
            function,
            format!("{function}: classification labels must not be empty"),
        ));
    }
    let mut matrix = Matrix::zeros(labels.len(), classes.len());
    for (row, label) in labels.iter().enumerate() {
        let class = classes.binary_search(label).map_err(|_| {
            deep_learning_error(function, "internal classification class lookup failed")
        })?;
        matrix.set(row, class, 1.0);
    }
    Ok(ClassificationTarget {
        classes,
        one_hot: matrix,
    })
}
