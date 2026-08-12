use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderAdamUpdateRequest,
    ProviderAdamUpdateResult,
};
use runmat_builtins::{CellArray, NumericDType, ObjectInstance, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::BuiltinResult;

use super::{
    any_type, autodiff, deep_learning_error, ensure_dlarray_class_registered, gather_args, model,
    object, parse_name_values, positive_usize, scalar_text, text_or_missing, unsupported_error,
};
use crate::builtins::common::{gpu_helpers, tensor};

const DLUPDATE_MAX_DEPTH: usize = 256;

#[runtime_builtin(
    name = "trainingOptions",
    category = "deep_learning",
    summary = "Create training options compatibility objects.",
    keywords = "trainingOptions,deep learning,training,options,solver",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn training_options_builtin(
    solver: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let solver = scalar_text(&solver, "trainingOptions")?.to_ascii_lowercase();
    match solver.as_str() {
        "sgdm" | "adam" | "rmsprop" => {}
        other => {
            return Err(deep_learning_error(
                "trainingOptions",
                format!("trainingOptions: unsupported solver '{other}'"),
            ));
        }
    }
    let mut properties = vec![
        ("SolverName".to_string(), Value::String(solver.clone())),
        ("MaxEpochs".to_string(), Value::Num(30.0)),
        ("MiniBatchSize".to_string(), Value::Num(128.0)),
        (
            "InitialLearnRate".to_string(),
            Value::Num(if solver == "adam" { 0.001 } else { 0.01 }),
        ),
        ("Shuffle".to_string(), Value::String("once".into())),
        ("Verbose".to_string(), Value::Bool(true)),
        ("Plots".to_string(), Value::String("none".into())),
        (
            "ExecutionEnvironment".to_string(),
            Value::String("auto".into()),
        ),
    ];
    for (key, value) in parse_name_values(gather_args(rest).await?, "trainingOptions")? {
        properties.push((canonical_training_option(&key), value));
    }
    Ok(object("nnet.cnn.TrainingOptions", properties))
}

fn canonical_training_option(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "maxepochs" => "MaxEpochs",
        "minibatchsize" => "MiniBatchSize",
        "initiallearnrate" => "InitialLearnRate",
        "shuffle" => "Shuffle",
        "verbose" => "Verbose",
        "plots" => "Plots",
        "executionenvironment" => "ExecutionEnvironment",
        "validationdata" => "ValidationData",
        "validationfrequency" => "ValidationFrequency",
        "learnrateschedule" => "LearnRateSchedule",
        "learnratedropperiod" => "LearnRateDropPeriod",
        "learnratedropfactor" => "LearnRateDropFactor",
        "gradientthreshold" => "GradientThreshold",
        "l2regularization" => "L2Regularization",
        "momentum" => "Momentum",
        "squaredgradientdecayfactor" => "SquaredGradientDecayFactor",
        "gradientdecayfactor" => "GradientDecayFactor",
        "epsilon" => "Epsilon",
        other => other,
    }
    .to_string()
}

#[runtime_builtin(
    name = "dlarray",
    category = "deep_learning",
    summary = "Create a dlarray compatibility object around numeric or gpuArray data.",
    keywords = "dlarray,deep learning,array,labels",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlarray_builtin(data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_dlarray_class_registered();
    validate_dlarray_data(&data)?;
    let gathered = gather_args(rest).await?;
    if gathered.len() > 1 {
        return Err(deep_learning_error(
            "dlarray",
            "dlarray: only the optional dimension-label format argument is supported",
        ));
    }
    let format = text_or_missing(gathered.first(), "", "dlarray")?;
    autodiff::annotate_dlarray_value(object(
        "dlarray",
        vec![
            ("Data", data),
            ("Format", Value::String(format.clone())),
            ("Labels", Value::String(format)),
        ],
    ))
}

fn validate_dlarray_data(data: &Value) -> BuiltinResult<()> {
    if matches!(data, Value::Int(_))
        || matches!(data, Value::Tensor(tensor) if tensor.integer_storage().is_some())
    {
        return Err(deep_learning_error(
            "dlarray",
            "dlarray: integer data is not supported; use double, single, logical, or gpuArray data",
        ));
    }
    Ok(())
}

#[runtime_builtin(
    name = "dlfeval",
    category = "deep_learning",
    summary = "Evaluate a function handle in Deep Learning compatibility context.",
    keywords = "dlfeval,deep learning,autodiff,function evaluation",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::DLFEVAL_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlfeval_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let Some((function, rest)) = args.split_first() else {
        return Err(deep_learning_error(
            "dlfeval",
            "dlfeval: expected a function handle followed by input arguments",
        ));
    };
    if !matches!(
        function,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    ) {
        return Err(deep_learning_error(
            "dlfeval",
            format!("dlfeval: expected a function handle, got {function:?}"),
        ));
    }
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);
    autodiff::with_tape_context(function.clone(), rest, requested_outputs).await
}

#[runtime_builtin(
    name = "adamupdate",
    category = "deep_learning",
    summary = "Update numeric parameters using the Adam optimizer rule.",
    keywords = "adamupdate,deep learning,adam,optimizer",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ADAMUPDATE_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn adamupdate_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let has_gpu_input = args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)));
    let options = if has_gpu_input {
        match try_adamupdate_gpu(&args)? {
            Some(eval) => return eval.output(),
            None => AdamUpdateArgs::parse(gather_args(args).await?)?,
        }
    } else {
        AdamUpdateArgs::parse(args)?
    };
    let eval = evaluate_adamupdate(options)?;
    match crate::output_count::current_output_count() {
        None => Ok(eval.parameters),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![eval.parameters])),
        Some(2) => Ok(Value::OutputList(vec![eval.parameters, eval.average_grad])),
        Some(3) => Ok(Value::OutputList(vec![
            eval.parameters,
            eval.average_grad,
            eval.average_sq_grad,
        ])),
        Some(requested) => Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: requested {requested} outputs, but at most 3 are supported"),
        )),
    }
}

#[runtime_builtin(
    name = "dlupdate",
    category = "deep_learning",
    summary = "Apply a function handle across matching parameter trees.",
    keywords = "dlupdate,deep learning,model update",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::DLUPDATE_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlupdate_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let Some((function, rest)) = args.split_first() else {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: expected a function handle followed by one or more parameter trees",
        ));
    };
    if !is_function_handle(function) {
        return Err(deep_learning_error(
            "dlupdate",
            format!("dlupdate: expected a function handle, got {function:?}"),
        ));
    }
    if rest.is_empty() {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: expected at least one parameter tree",
        ));
    }

    let trees = rest.to_vec();
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);
    if requested_outputs == 0 {
        return Ok(Value::OutputList(Vec::new()));
    }

    let outputs = dlupdate_node(function, &trees, requested_outputs, 0).await?;
    if requested_outputs == 1 {
        Ok(outputs.into_iter().next().unwrap())
    } else {
        Ok(Value::OutputList(outputs))
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

#[async_recursion::async_recursion(?Send)]
async fn dlupdate_node(
    function: &Value,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    if depth > DLUPDATE_MAX_DEPTH {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: parameter tree nesting exceeds supported recursion depth",
        ));
    }
    let Some(first) = values.first() else {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: internal empty parameter-tree node",
        ));
    };

    match first {
        Value::Struct(reference) => {
            if is_learnables_struct(reference) {
                dlupdate_learnables_struct(function, reference, values, requested_outputs, depth + 1)
                    .await
            } else {
                dlupdate_struct(function, reference, values, requested_outputs, depth + 1).await
            }
        }
        Value::Cell(reference) => {
            dlupdate_cell(function, reference, values, requested_outputs, depth + 1).await
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
            dlupdate_table(function, object, values, requested_outputs, depth + 1).await
        }
        Value::Object(object) if model::is_deep_learning_network_object(object) => {
            dlupdate_network(function, object, values, requested_outputs, depth + 1).await
        }
        Value::Object(object) if is_leaf_object(object) => {
            if values.iter().any(is_tree_container) {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: all parameter trees must have matching container structure",
                ));
            }
            dlupdate_leaf(function, values, requested_outputs).await
        }
        Value::Object(object) => Err(unsupported_error(
            "dlupdate",
            format!(
                "dlupdate: traversal for '{}' objects requires dlnetwork/model metadata infrastructure",
                object.class_name
            ),
        )),
        _ => {
            if values.iter().any(is_tree_container) {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: all parameter trees must have matching container structure",
                ));
            }
            dlupdate_leaf(function, values, requested_outputs).await
        }
    }
}

async fn dlupdate_struct(
    function: &Value,
    reference: &StructValue,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let field_names = reference.field_names().cloned().collect::<Vec<_>>();
    let structs = values
        .iter()
        .map(|value| match value {
            Value::Struct(st) => {
                let names = st.field_names().cloned().collect::<Vec<_>>();
                if names == field_names {
                    Ok(st)
                } else {
                    Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: struct parameter trees must have the same fields in the same order",
                    ))
                }
            }
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching struct tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    let mut outputs = (0..requested_outputs)
        .map(|_| StructValue::new())
        .collect::<Vec<_>>();
    for name in field_names {
        let field_values = structs
            .iter()
            .map(|st| {
                st.fields
                    .get(&name)
                    .cloned()
                    .unwrap_or(Value::Num(f64::NAN))
            })
            .collect::<Vec<_>>();
        let updated = dlupdate_node(function, &field_values, requested_outputs, depth).await?;
        for (out, value) in outputs.iter_mut().zip(updated.into_iter()) {
            out.insert(name.clone(), value);
        }
    }

    Ok(outputs.into_iter().map(Value::Struct).collect())
}

async fn dlupdate_learnables_struct(
    function: &Value,
    reference: &StructValue,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let structs = values
        .iter()
        .map(|value| match value {
            Value::Struct(st) if is_learnables_struct(st) => Ok(st),
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching learnables struct tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let tables = structs.iter().cloned().cloned().collect::<Vec<_>>();
    require_matching_learnable_metadata(&tables)?;
    let value_columns = structs
        .iter()
        .map(|st| {
            st.fields.get("Value").cloned().ok_or_else(|| {
                deep_learning_error("dlupdate", "dlupdate: learnables struct is missing Value")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let updated_values = dlupdate_node(function, &value_columns, requested_outputs, depth).await?;
    updated_values
        .into_iter()
        .map(|value_column| {
            let mut out = reference.clone();
            out.insert("Value", value_column);
            Ok(Value::Struct(out))
        })
        .collect()
}

async fn dlupdate_cell(
    function: &Value,
    reference: &CellArray,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let cells = values
        .iter()
        .map(|value| match value {
            Value::Cell(cell) if cell.shape == reference.shape => Ok(cell),
            Value::Cell(_) => Err(deep_learning_error(
                "dlupdate",
                "dlupdate: cell parameter trees must have the same shape",
            )),
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching cell tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    let mut outputs = (0..requested_outputs)
        .map(|_| Vec::with_capacity(reference.data.len()))
        .collect::<Vec<_>>();
    for idx in 0..reference.data.len() {
        let elements = cells
            .iter()
            .map(|cell| cell.data[idx].clone())
            .collect::<Vec<_>>();
        let updated = dlupdate_node(function, &elements, requested_outputs, depth).await?;
        for (out, value) in outputs.iter_mut().zip(updated.into_iter()) {
            out.push(value);
        }
    }

    outputs
        .into_iter()
        .map(|data| {
            CellArray::new_with_shape(data, reference.shape.clone())
                .map(Value::Cell)
                .map_err(|err| deep_learning_error("dlupdate", err))
        })
        .collect()
}

async fn dlupdate_table(
    function: &Value,
    reference: &ObjectInstance,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let variable_names = crate::builtins::table::table_variable_names_from_object(reference)
        .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
    require_learnables_table(&variable_names)?;
    let reference_height = crate::builtins::table::table_height(reference)
        .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
    let tables = values
        .iter()
        .map(|value| match value {
            Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
                if object.class_name != reference.class_name {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: table parameter trees must have the same tabular class",
                    ));
                }
                let names = crate::builtins::table::table_variable_names_from_object(object)
                    .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
                if names != variable_names {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: table parameter trees must have the same variable names",
                    ));
                }
                let height = crate::builtins::table::table_height(object)
                    .map_err(|err| deep_learning_error("dlupdate", err.to_string()))?;
                if height != reference_height {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: table parameter trees must have the same height",
                    ));
                }
                Ok(object)
            }
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching table tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let table_variables = tables
        .iter()
        .map(|object| {
            crate::builtins::table::table_variables(object)
                .map_err(|err| deep_learning_error("dlupdate", err.to_string()))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;

    require_matching_learnable_metadata(&table_variables)?;
    let value_columns = table_variables
        .iter()
        .map(|vars| {
            vars.fields.get("Value").cloned().ok_or_else(|| {
                deep_learning_error("dlupdate", "dlupdate: learnables table is missing Value")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    if !value_columns
        .iter()
        .all(|value| matches!(value, Value::Cell(_)))
    {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: learnables table Value variable must be a cell array",
        ));
    }
    let updated_values = dlupdate_node(function, &value_columns, requested_outputs, depth).await?;

    updated_values
        .into_iter()
        .map(|value_column| {
            let mut variables = table_variables[0].clone();
            variables.insert("Value", value_column);
            crate::builtins::table::table_replace_variables_like(reference, variables)
                .map_err(|err| deep_learning_error("dlupdate", err.to_string()))
        })
        .collect()
}

async fn dlupdate_network(
    function: &Value,
    reference: &ObjectInstance,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<Vec<Value>> {
    let networks = values
        .iter()
        .map(|value| match value {
            Value::Object(object) if model::is_deep_learning_network_object(object) => {
                if object.class_name != reference.class_name {
                    return Err(deep_learning_error(
                        "dlupdate",
                        "dlupdate: dlnetwork parameter trees must have the same network class",
                    ));
                }
                Ok(object)
            }
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected matching dlnetwork tree, got {other:?}"),
            )),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let learnables = networks
        .iter()
        .map(|object| {
            object.properties.get("Learnables").cloned().ok_or_else(|| {
                deep_learning_error("dlupdate", "dlupdate: dlnetwork is missing Learnables")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let updated = dlupdate_node(function, &learnables, requested_outputs, depth).await?;
    updated
        .into_iter()
        .map(|learnables| apply_learnables_to_network(reference.clone(), learnables))
        .collect()
}

fn apply_learnables_to_network(
    mut network: ObjectInstance,
    learnables: Value,
) -> BuiltinResult<Value> {
    let mut parameter_values = Vec::new();
    match &learnables {
        Value::Struct(st) => {
            let layer_names = string_cells(st.fields.get("Layer"), "Layer")?;
            let parameter_names = string_cells(st.fields.get("Parameter"), "Parameter")?;
            let Value::Cell(values) = st.fields.get("Value").ok_or_else(|| {
                deep_learning_error("dlupdate", "dlupdate: learnables tree is missing Value")
            })?
            else {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: learnables Value variable must be a cell array",
                ));
            };
            if layer_names.len() != parameter_names.len() || values.data.len() != layer_names.len()
            {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: learnables metadata and Value rows must match",
                ));
            }
            for idx in 0..layer_names.len() {
                parameter_values.push((
                    layer_names[idx].clone(),
                    parameter_names[idx].clone(),
                    values.data[idx].clone(),
                ));
            }
        }
        other => {
            return Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: expected Learnables struct, got {other:?}"),
            ))
        }
    }

    if let Some(Value::Cell(mut layers)) = network.properties.get("Layers").cloned() {
        for layer in &mut layers.data {
            let Value::Object(layer_object) = layer else {
                continue;
            };
            let layer_name = model::layer_name(layer_object);
            for (target_layer, parameter, value) in &parameter_values {
                if *target_layer == layer_name {
                    layer_object
                        .properties
                        .insert(parameter.clone(), value.clone());
                }
            }
        }
        network
            .properties
            .insert("Layers".to_string(), Value::Cell(layers));
    }
    network
        .properties
        .insert("Learnables".to_string(), learnables);
    Ok(Value::Object(network))
}

fn string_cells(value: Option<&Value>, label: &'static str) -> BuiltinResult<Vec<String>> {
    let Some(Value::Cell(cell)) = value else {
        return Err(deep_learning_error(
            "dlupdate",
            format!("dlupdate: learnables tree is missing {label} cell metadata"),
        ));
    };
    cell.data
        .iter()
        .map(|value| match value {
            Value::String(text) => Ok(text.clone()),
            other => Err(deep_learning_error(
                "dlupdate",
                format!("dlupdate: learnables {label} entries must be strings, got {other:?}"),
            )),
        })
        .collect()
}

fn require_learnables_table(variable_names: &[String]) -> BuiltinResult<()> {
    for required in ["Layer", "Parameter", "Value"] {
        if !variable_names.iter().any(|name| name == required) {
            return Err(deep_learning_error(
                "dlupdate",
                "dlupdate: learnables tables must contain Layer, Parameter, and Value variables",
            ));
        }
    }
    Ok(())
}

fn is_learnables_struct(value: &StructValue) -> bool {
    ["Layer", "Parameter", "Value"]
        .iter()
        .all(|field| value.fields.contains_key(*field))
}

fn require_matching_learnable_metadata(tables: &[StructValue]) -> BuiltinResult<()> {
    let Some(reference) = tables.first() else {
        return Ok(());
    };
    for variable in ["Layer", "Parameter"] {
        let expected = reference.fields.get(variable).ok_or_else(|| {
            deep_learning_error(
                "dlupdate",
                format!("dlupdate: learnables table is missing {variable}"),
            )
        })?;
        for table in tables.iter().skip(1) {
            let actual = table.fields.get(variable).ok_or_else(|| {
                deep_learning_error(
                    "dlupdate",
                    format!("dlupdate: learnables table is missing {variable}"),
                )
            })?;
            if actual != expected {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: learnables table Layer and Parameter metadata must match",
                ));
            }
        }
    }
    Ok(())
}

async fn dlupdate_leaf(
    function: &Value,
    values: &[Value],
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let result = crate::call_feval_async_with_outputs(function.clone(), values, requested_outputs)
        .await
        .map_err(|err| {
            deep_learning_error(
                "dlupdate",
                format!("dlupdate: function-handle evaluation failed: {err}"),
            )
        })?;
    normalize_dlupdate_outputs(result, requested_outputs)
}

fn normalize_dlupdate_outputs(value: Value, requested_outputs: usize) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::OutputList(values) if values.len() == requested_outputs => Ok(values),
        Value::OutputList(values) => Err(deep_learning_error(
            "dlupdate",
            format!(
                "dlupdate: function returned {} outputs but {} were requested",
                values.len(),
                requested_outputs
            ),
        )),
        other if requested_outputs == 1 => Ok(vec![other]),
        _ => Err(deep_learning_error(
            "dlupdate",
            "dlupdate: function did not return the requested number of outputs",
        )),
    }
}

fn is_leaf_object(object: &ObjectInstance) -> bool {
    object.class_name == "dlarray"
}

fn is_tree_container(value: &Value) -> bool {
    match value {
        Value::Struct(_) | Value::Cell(_) => true,
        Value::Object(object) => crate::builtins::table::is_tabular_object(object),
        _ => false,
    }
}

struct AdamUpdateArgs {
    parameters: NumericPayload,
    gradient: NumericPayload,
    average_grad: NumericPayload,
    average_sq_grad: NumericPayload,
    iteration: usize,
    learn_rate: f64,
    gradient_decay_factor: f64,
    squared_gradient_decay_factor: f64,
    epsilon: f64,
}

impl AdamUpdateArgs {
    fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        if !(5..=9).contains(&args.len()) {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: expected 5 to 9 positional arguments",
            ));
        }
        let parameters = NumericPayload::parse(&args[0], "parameters", true)?;
        if parameters.data.is_empty() {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: parameters must not be empty",
            ));
        }
        let gradient = NumericPayload::parse(&args[1], "grad", true)?;
        let average_grad = NumericPayload::parse(&args[2], "averageGrad", false)?
            .with_default_shape(&parameters)?;
        let average_sq_grad = NumericPayload::parse(&args[3], "averageSqGrad", false)?
            .with_default_shape(&parameters)?;
        require_same_shape(&parameters, &gradient, "grad")?;
        require_same_shape(&parameters, &average_grad, "averageGrad")?;
        require_same_shape(&parameters, &average_sq_grad, "averageSqGrad")?;
        let iteration = positive_iteration(&args[4])?;
        Ok(Self {
            parameters,
            gradient,
            average_grad,
            average_sq_grad,
            iteration,
            learn_rate: optional_positive_scalar(&args, 5, 0.001, "learnRate")?,
            gradient_decay_factor: optional_unit_scalar(&args, 6, 0.9, "gradDecay")?,
            squared_gradient_decay_factor: optional_unit_scalar(&args, 7, 0.999, "sqGradDecay")?,
            epsilon: optional_positive_scalar(&args, 8, 1.0e-8, "epsilon")?,
        })
    }
}

struct AdamUpdateEval {
    parameters: Value,
    average_grad: Value,
    average_sq_grad: Value,
}

struct AdamUpdateGpuEval {
    provider: &'static dyn AccelProvider,
    parameters: GpuTensorHandle,
    average_grad: GpuTensorHandle,
    average_sq_grad: GpuTensorHandle,
}

impl AdamUpdateGpuEval {
    fn from_provider(
        provider: &'static dyn AccelProvider,
        result: ProviderAdamUpdateResult,
    ) -> Self {
        Self {
            provider,
            parameters: result.parameters,
            average_grad: result.average_grad,
            average_sq_grad: result.average_sq_grad,
        }
    }

    fn output(self) -> BuiltinResult<Value> {
        match crate::output_count::current_output_count() {
            None => {
                let _ = self.provider.free(&self.average_grad);
                let _ = self.provider.free(&self.average_sq_grad);
                Ok(gpu_helpers::resident_gpu_value(self.parameters))
            }
            Some(0) => {
                let _ = self.provider.free(&self.parameters);
                let _ = self.provider.free(&self.average_grad);
                let _ = self.provider.free(&self.average_sq_grad);
                Ok(Value::OutputList(Vec::new()))
            }
            Some(1) => {
                let _ = self.provider.free(&self.average_grad);
                let _ = self.provider.free(&self.average_sq_grad);
                Ok(Value::OutputList(vec![gpu_helpers::resident_gpu_value(
                    self.parameters,
                )]))
            }
            Some(2) => {
                let _ = self.provider.free(&self.average_sq_grad);
                Ok(Value::OutputList(vec![
                    gpu_helpers::resident_gpu_value(self.parameters),
                    gpu_helpers::resident_gpu_value(self.average_grad),
                ]))
            }
            Some(3) => Ok(Value::OutputList(vec![
                gpu_helpers::resident_gpu_value(self.parameters),
                gpu_helpers::resident_gpu_value(self.average_grad),
                gpu_helpers::resident_gpu_value(self.average_sq_grad),
            ])),
            Some(requested) => {
                let _ = self.provider.free(&self.parameters);
                let _ = self.provider.free(&self.average_grad);
                let _ = self.provider.free(&self.average_sq_grad);
                Err(deep_learning_error(
                    "adamupdate",
                    format!(
                        "adamupdate: requested {requested} outputs, but at most 3 are supported"
                    ),
                ))
            }
        }
    }
}

fn try_adamupdate_gpu(args: &[Value]) -> BuiltinResult<Option<AdamUpdateGpuEval>> {
    if !(5..=9).contains(&args.len()) {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: expected 5 to 9 positional arguments",
        ));
    }

    let Value::GpuTensor(parameters) = &args[0] else {
        return Ok(None);
    };
    let Value::GpuTensor(gradient) = &args[1] else {
        return Ok(None);
    };
    if args[4..]
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }
    let average_grad = match gpu_optimizer_state(&args[2], "averageGrad")? {
        GpuOptimizerState::Resident(handle) => Some(handle),
        GpuOptimizerState::Empty => None,
        GpuOptimizerState::HostFallback => return Ok(None),
    };
    let average_sq_grad = match gpu_optimizer_state(&args[3], "averageSqGrad")? {
        GpuOptimizerState::Resident(handle) => Some(handle),
        GpuOptimizerState::Empty => None,
        GpuOptimizerState::HostFallback => return Ok(None),
    };

    if runmat_accelerate_api::handle_storage(parameters) == GpuTensorStorage::ComplexInterleaved
        || runmat_accelerate_api::handle_storage(gradient) == GpuTensorStorage::ComplexInterleaved
        || average_grad.is_some_and(|handle| {
            runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
        })
        || average_sq_grad.is_some_and(|handle| {
            runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
        })
    {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: complex gpuArray optimizer tensors are not supported",
        ));
    }

    if parameters.shape != gradient.shape
        || average_grad.is_some_and(|handle| handle.shape != parameters.shape)
        || average_sq_grad.is_some_and(|handle| handle.shape != parameters.shape)
    {
        return Err(deep_learning_error(
            "adamupdate",
            format!(
                "adamupdate: optimizer tensors must match parameter shape {:?}",
                parameters.shape
            ),
        ));
    }

    let Some(provider) = runmat_accelerate_api::provider_for_handle(parameters) else {
        return Ok(None);
    };
    let Some(gradient_provider) = runmat_accelerate_api::provider_for_handle(gradient) else {
        return Ok(None);
    };
    if !std::ptr::eq(gradient_provider, provider) {
        return Ok(None);
    }
    if average_grad.is_some_and(|handle| {
        runmat_accelerate_api::provider_for_handle(handle)
            .is_none_or(|state_provider| !std::ptr::eq(state_provider, provider))
    }) || average_sq_grad.is_some_and(|handle| {
        runmat_accelerate_api::provider_for_handle(handle)
            .is_none_or(|state_provider| !std::ptr::eq(state_provider, provider))
    }) {
        return Ok(None);
    }

    let iteration = positive_iteration(&args[4])?;
    let request = ProviderAdamUpdateRequest {
        parameters,
        gradient,
        average_grad,
        average_sq_grad,
        iteration,
        learn_rate: optional_positive_scalar(args, 5, 0.001, "learnRate")?,
        gradient_decay_factor: optional_unit_scalar(args, 6, 0.9, "gradDecay")?,
        squared_gradient_decay_factor: optional_unit_scalar(args, 7, 0.999, "sqGradDecay")?,
        epsilon: optional_positive_scalar(args, 8, 1.0e-8, "epsilon")?,
    };

    match provider.adam_update(&request) {
        Ok(result) => Ok(Some(AdamUpdateGpuEval::from_provider(provider, result))),
        Err(err) if provider_is_unsupported(&err) => Ok(None),
        Err(err) => Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {err}"),
        )),
    }
}

enum GpuOptimizerState<'a> {
    Resident(&'a GpuTensorHandle),
    Empty,
    HostFallback,
}

fn gpu_optimizer_state<'a>(
    value: &'a Value,
    label: &'static str,
) -> BuiltinResult<GpuOptimizerState<'a>> {
    match value {
        Value::GpuTensor(handle) => Ok(GpuOptimizerState::Resident(handle)),
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 0 => {
            Ok(GpuOptimizerState::Empty)
        }
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => Ok(GpuOptimizerState::HostFallback),
        other => Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be a gpuArray or empty numeric array for resident gpuArray updates, got {other:?}"),
        )),
    }
}

fn provider_is_unsupported(err: &anyhow::Error) -> bool {
    let message = err.to_string();
    message.contains("not supported") || message.contains("unsupported")
}

fn evaluate_adamupdate(args: AdamUpdateArgs) -> BuiltinResult<AdamUpdateEval> {
    let count = args.parameters.data.len();
    let mut updated_parameters = Vec::with_capacity(count);
    let mut updated_average_grad = Vec::with_capacity(count);
    let mut updated_average_sq_grad = Vec::with_capacity(count);
    let iteration = args.iteration as f64;
    let grad_correction = 1.0 - args.gradient_decay_factor.powf(iteration);
    let sq_grad_correction = 1.0 - args.squared_gradient_decay_factor.powf(iteration);
    if grad_correction <= 0.0 || sq_grad_correction <= 0.0 {
        return Err(deep_learning_error(
            "adamupdate",
            "adamupdate: decay factors and iteration produced invalid bias correction",
        ));
    }
    for idx in 0..count {
        let grad = args.gradient.data[idx];
        let avg_grad = args.gradient_decay_factor * args.average_grad.data[idx]
            + (1.0 - args.gradient_decay_factor) * grad;
        let avg_sq_grad = args.squared_gradient_decay_factor * args.average_sq_grad.data[idx]
            + (1.0 - args.squared_gradient_decay_factor) * grad * grad;
        let corrected_grad = avg_grad / grad_correction;
        let corrected_sq_grad = avg_sq_grad / sq_grad_correction;
        let step = args.learn_rate * corrected_grad / (corrected_sq_grad.sqrt() + args.epsilon);
        let updated = args.parameters.data[idx] - step;
        if !updated.is_finite() || !avg_grad.is_finite() || !avg_sq_grad.is_finite() {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: update produced a non-finite value",
            ));
        }
        updated_parameters.push(updated);
        updated_average_grad.push(avg_grad);
        updated_average_sq_grad.push(avg_sq_grad);
    }
    Ok(AdamUpdateEval {
        parameters: args.parameters.materialize(updated_parameters)?,
        average_grad: args.average_grad.materialize(updated_average_grad)?,
        average_sq_grad: args.average_sq_grad.materialize(updated_average_sq_grad)?,
    })
}

#[derive(Clone)]
struct NumericPayload {
    data: Vec<f64>,
    shape: Vec<usize>,
    repr: NumericRepr,
}

impl NumericPayload {
    fn parse(value: &Value, label: &'static str, reject_empty: bool) -> BuiltinResult<Self> {
        let payload = match value {
            Value::Num(n) if n.is_finite() => Self {
                data: vec![*n],
                shape: vec![1, 1],
                repr: NumericRepr::Scalar,
            },
            Value::Int(i) => Self {
                data: vec![i.to_f64()],
                shape: vec![1, 1],
                repr: NumericRepr::Scalar,
            },
            Value::Tensor(tensor) => {
                let dtype = tensor.numeric_dtype();
                if !matches!(dtype, NumericDType::F64 | NumericDType::F32) {
                    return Err(deep_learning_error(
                        "adamupdate",
                        format!(
                            "adamupdate: {label} tensor must be double or single, got {}",
                            dtype.class_name()
                        ),
                    ));
                }
                let data = tensor
                    .clone()
                    .into_numeric_storage()
                    .map_err(|err| deep_learning_error("adamupdate", err))?
                    .materialize_f64();
                ensure_finite(&data, label)?;
                Self {
                    data,
                    shape: tensor.shape.clone(),
                    repr: NumericRepr::Dense {
                        shape: tensor.shape.clone(),
                        dtype,
                    },
                }
            }
            Value::Object(object) if object.class_name == "dlarray" => {
                let data = object.properties.get("Data").ok_or_else(|| {
                    deep_learning_error("adamupdate", "adamupdate: dlarray is missing Data")
                })?;
                let inner = Self::parse(data, label, reject_empty)?;
                Self {
                    data: inner.data,
                    shape: inner.shape,
                    repr: NumericRepr::Dlarray {
                        inner: Box::new(inner.repr),
                        format: Box::new(
                            object
                                .properties
                                .get("Format")
                                .cloned()
                                .unwrap_or_else(|| Value::String(String::new())),
                        ),
                        labels: Box::new(
                            object
                                .properties
                                .get("Labels")
                                .cloned()
                                .unwrap_or_else(|| Value::String(String::new())),
                        ),
                    },
                }
            }
            Value::GpuTensor(_) => {
                return Err(deep_learning_error(
                    "adamupdate",
                    "adamupdate: gpuArray optimizer updates require provider kernels and are not implemented in this runtime slice",
                ));
            }
            other => {
                return Err(deep_learning_error(
                    "adamupdate",
                    format!("adamupdate: {label} must be a finite numeric array or dlarray, got {other:?}"),
                ));
            }
        };
        if reject_empty && payload.data.is_empty() {
            return Err(deep_learning_error(
                "adamupdate",
                format!("adamupdate: {label} must not be empty"),
            ));
        }
        Ok(payload)
    }

    fn with_default_shape(mut self, fallback: &NumericPayload) -> BuiltinResult<Self> {
        if !self.data.is_empty() {
            return Ok(self);
        }
        self.data = vec![0.0; fallback.data.len()];
        self.shape = fallback.shape.clone();
        self.repr = fallback.repr.clone();
        Ok(self)
    }

    fn materialize(&self, data: Vec<f64>) -> BuiltinResult<Value> {
        self.repr.materialize(data)
    }
}

#[derive(Clone)]
enum NumericRepr {
    Scalar,
    Dense {
        shape: Vec<usize>,
        dtype: NumericDType,
    },
    Dlarray {
        inner: Box<NumericRepr>,
        format: Box<Value>,
        labels: Box<Value>,
    },
}

impl NumericRepr {
    fn materialize(&self, data: Vec<f64>) -> BuiltinResult<Value> {
        match self {
            Self::Scalar => match data.as_slice() {
                [value] => Ok(Value::Num(*value)),
                _ => {
                    let len = data.len();
                    Tensor::new(data, vec![1, len])
                        .map(Value::Tensor)
                        .map_err(|err| deep_learning_error("adamupdate", err))
                }
            },
            Self::Dense { shape, dtype } => Tensor::new_with_dtype(data, shape.clone(), *dtype)
                .map(Value::Tensor)
                .map_err(|err| deep_learning_error("adamupdate", err)),
            Self::Dlarray {
                inner,
                format,
                labels,
            } => Ok(object(
                "dlarray",
                vec![
                    ("Data", inner.materialize(data)?),
                    ("Format", format.as_ref().clone()),
                    ("Labels", labels.as_ref().clone()),
                ],
            )),
        }
    }
}

fn ensure_finite(values: &[f64], label: &'static str) -> BuiltinResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must contain finite values"),
        ));
    }
    Ok(())
}

fn require_same_shape(
    expected: &NumericPayload,
    actual: &NumericPayload,
    label: &'static str,
) -> BuiltinResult<()> {
    if expected.shape != actual.shape || expected.data.len() != actual.data.len() {
        return Err(deep_learning_error(
            "adamupdate",
            format!(
                "adamupdate: {label} must match parameter shape {:?}, got {:?}",
                expected.shape, actual.shape
            ),
        ));
    }
    Ok(())
}

fn positive_iteration(value: &Value) -> BuiltinResult<usize> {
    positive_usize(value, "adamupdate", "iteration").map_err(|_| {
        deep_learning_error(
            "adamupdate",
            "adamupdate: iteration must be a positive integer",
        )
    })
}

fn optional_finite_scalar(
    args: &[Value],
    index: usize,
    default: f64,
    label: &'static str,
) -> BuiltinResult<f64> {
    let Some(value) = args.get(index) else {
        return Ok(default);
    };
    match value {
        Value::Num(n) if n.is_finite() => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            let n = crate::builtins::common::tensor::tensor_value_f64(tensor, 0);
            if n.is_finite() {
                Ok(n)
            } else {
                Err(deep_learning_error(
                    "adamupdate",
                    format!("adamupdate: {label} must be a finite numeric scalar, got {value:?}"),
                ))
            }
        }
        other => Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be a finite numeric scalar, got {other:?}"),
        )),
    }
}

fn optional_positive_scalar(
    args: &[Value],
    index: usize,
    default: f64,
    label: &'static str,
) -> BuiltinResult<f64> {
    let value = optional_finite_scalar(args, index, default, label)?;
    if value <= 0.0 {
        return Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be positive"),
        ));
    }
    Ok(value)
}

fn optional_unit_scalar(
    args: &[Value],
    index: usize,
    default: f64,
    label: &'static str,
) -> BuiltinResult<f64> {
    let value = optional_finite_scalar(args, index, default, label)?;
    if !(0.0..1.0).contains(&value) {
        return Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be greater than or equal to 0 and less than 1"),
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, IntegerStorage};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "got {actual}, expected {expected}"
        );
    }

    #[test]
    fn adamupdate_iteration_parser_preserves_typed_integer_bounds() {
        assert_eq!(
            positive_iteration(&Value::Int(IntValue::U16(7))).unwrap(),
            7
        );
        assert!(positive_iteration(&Value::Int(IntValue::I8(-1))).is_err());

        let exact = (1_u64 << 53) + 1;
        let typed =
            Tensor::new_integer(IntegerStorage::U64(vec![exact]), vec![1, 1]).expect("iteration");
        let parsed = positive_iteration(&Value::Tensor(typed));
        if usize::BITS == 64 {
            assert_eq!(parsed.unwrap(), exact as usize);
        } else {
            assert!(parsed.is_err());
        }

        let boundary = positive_iteration(&Value::Num(usize::MAX as f64));
        if usize::BITS == 64 {
            assert!(boundary.is_err());
        } else {
            assert_eq!(boundary.unwrap(), usize::MAX);
        }
        assert!(positive_iteration(&Value::Num((usize::MAX as f64) + 1.0)).is_err());
    }

    #[test]
    fn adamupdate_payload_preserves_native_single_representation() {
        let input = Tensor::from_f32(vec![0.1, f32::MAX], vec![1, 2]).expect("single input");
        let payload = NumericPayload::parse(&Value::Tensor(input), "parameters", true)
            .expect("single payload");
        assert_eq!(payload.data, vec![f64::from(0.1_f32), f64::from(f32::MAX)]);
        let output = payload.materialize(vec![0.25, 0.5]).expect("single output");
        let Value::Tensor(output) = output else {
            panic!("expected tensor output");
        };
        assert_eq!(
            output.into_numeric_storage().expect("single storage"),
            runmat_builtins::NumericStorage::F32(vec![0.25, 0.5])
        );
    }

    #[test]
    fn adamupdate_gpu_inputs_return_resident_outputs() {
        test_support::with_test_provider(|provider| {
            let shape = [1usize, 3usize];
            let parameters = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &shape,
                })
                .expect("upload parameters");
            let gradient = provider
                .upload(&HostTensorView {
                    data: &[0.1, -0.2, 0.3],
                    shape: &shape,
                })
                .expect("upload gradient");
            let empty = Tensor::new(Vec::new(), vec![0, 0]).expect("empty");

            provider.reset_telemetry();
            let _guard = crate::output_count::push_output_count(Some(3));
            let out = block_on(adamupdate_builtin(vec![
                Value::GpuTensor(parameters),
                Value::GpuTensor(gradient),
                Value::Tensor(empty.clone()),
                Value::Tensor(empty),
                Value::Num(1.0),
                Value::Num(0.01),
            ]))
            .expect("adamupdate gpu");

            let Value::OutputList(values) = out else {
                panic!("expected output list");
            };
            assert_eq!(values.len(), 3);
            assert!(matches!(values[0], Value::GpuTensor(_)));
            assert!(matches!(values[1], Value::GpuTensor(_)));
            assert!(matches!(values[2], Value::GpuTensor(_)));

            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.download_bytes, 0);

            let updated = test_support::gather(values[0].clone()).expect("updated");
            let avg = test_support::gather(values[1].clone()).expect("avg");
            let avg_sq = test_support::gather(values[2].clone()).expect("avg sq");

            assert_eq!(updated.shape, shape);
            assert_eq!(avg.shape, shape);
            assert_eq!(avg_sq.shape, shape);
            assert_close(updated.materialize_f64()[0], 0.990000001);
            assert_close(updated.materialize_f64()[1], 2.0099999995);
            assert_close(updated.materialize_f64()[2], 2.9900000003333334);
            assert_close(avg.materialize_f64()[0], 0.01);
            assert_close(avg.materialize_f64()[1], -0.02);
            assert_close(avg.materialize_f64()[2], 0.03);
            assert_close(avg_sq.materialize_f64()[0], 0.00001);
            assert_close(avg_sq.materialize_f64()[1], 0.00004);
            assert_close(avg_sq.materialize_f64()[2], 0.00009);
        });
    }

    #[test]
    fn adamupdate_gpu_single_output_frees_state_outputs() {
        test_support::with_test_provider(|provider| {
            let shape = [1usize, 1usize];
            let parameters = provider
                .upload(&HostTensorView {
                    data: &[1.0],
                    shape: &shape,
                })
                .expect("upload parameters");
            let gradient = provider
                .upload(&HostTensorView {
                    data: &[0.1],
                    shape: &shape,
                })
                .expect("upload gradient");
            let average_grad = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &shape,
                })
                .expect("upload averageGrad");
            let average_sq_grad = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &shape,
                })
                .expect("upload averageSqGrad");

            let _guard = crate::output_count::push_output_count(Some(1));
            let out = block_on(adamupdate_builtin(vec![
                Value::GpuTensor(parameters),
                Value::GpuTensor(gradient),
                Value::GpuTensor(average_grad),
                Value::GpuTensor(average_sq_grad),
                Value::Num(1.0),
            ]))
            .expect("adamupdate gpu");

            let Value::OutputList(values) = out else {
                panic!("expected output list");
            };
            assert_eq!(values.len(), 1);
            assert!(matches!(values[0], Value::GpuTensor(_)));
        });
    }
}
