//! Typed per-role initializer selection and effect-free admission.
use super::*;
use runmat_builtins::DeepLearningLayerKind;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BuiltinInitializer {
    Glorot,
    He,
    NarrowNormal,
    Orthogonal,
    Zeros,
    Ones,
    UnitForget,
}

pub(super) enum Initializer {
    Builtin(BuiltinInitializer),
    Callback(Value),
}

pub(super) struct Task {
    pub(super) layer_index: usize,
    pub(super) kind: DeepLearningLayerKind,
    pub(super) role: ParameterRole,
    pub(super) shape: Vec<usize>,
    pub(super) expected_shape: ParameterShape,
    /// Precision already established by a supplied sibling, if any.
    pub(super) precision: Option<NumericDType>,
    pub(super) initializer: Initializer,
}

fn initializer_property(role: ParameterRole) -> &'static str {
    match role {
        ParameterRole::Weights => "WeightsInitializer",
        ParameterRole::InputWeights => "InputWeightsInitializer",
        ParameterRole::RecurrentWeights => "RecurrentWeightsInitializer",
        ParameterRole::Bias => "BiasInitializer",
        ParameterRole::Scale => "ScaleInitializer",
        ParameterRole::Offset => "OffsetInitializer",
    }
}

fn default_initializer(kind: DeepLearningLayerKind, role: ParameterRole) -> BuiltinInitializer {
    use BuiltinInitializer as I;
    match (kind, role) {
        (DeepLearningLayerKind::Lstm | DeepLearningLayerKind::Bilstm, ParameterRole::Bias) => {
            I::UnitForget
        }
        (_, ParameterRole::Bias | ParameterRole::Offset) => I::Zeros,
        (_, ParameterRole::Scale) => I::Ones,
        (_, ParameterRole::RecurrentWeights) => I::Orthogonal,
        _ => I::Glorot,
    }
}

fn parse_initializer(
    layer: &ObjectInstance,
    kind: DeepLearningLayerKind,
    role: ParameterRole,
    function: &'static str,
) -> BuiltinResult<Initializer> {
    let property = initializer_property(role);
    let Some(value) = layer.properties().get(property) else {
        return Ok(Initializer::Builtin(default_initializer(kind, role)));
    };
    if crate::builtins::deep_learning::layer_helpers::options::is_callable(value) {
        return Ok(Initializer::Callback(value.clone()));
    }
    let name = scalar_text(value, function)?;
    let builtin = match name.to_ascii_lowercase().as_str() {
        "glorot" => BuiltinInitializer::Glorot,
        "he" => BuiltinInitializer::He,
        "narrow-normal" => BuiltinInitializer::NarrowNormal,
        "orthogonal" => BuiltinInitializer::Orthogonal,
        "zeros" => BuiltinInitializer::Zeros,
        "ones" => BuiltinInitializer::Ones,
        "unit-forget-gate"
            if matches!(
                kind,
                DeepLearningLayerKind::Lstm | DeepLearningLayerKind::Bilstm
            ) && role == ParameterRole::Bias =>
        {
            BuiltinInitializer::UnitForget
        }
        _ => {
            return Err(deep_learning_error(
                function,
                format!("{function}: unsupported {property} '{name}'"),
            ));
        }
    };
    let supported = match role {
        ParameterRole::Weights if kind == DeepLearningLayerKind::Convolution1D => !matches!(
            builtin,
            BuiltinInitializer::Orthogonal | BuiltinInitializer::UnitForget
        ),
        ParameterRole::Weights | ParameterRole::InputWeights | ParameterRole::RecurrentWeights => {
            !matches!(builtin, BuiltinInitializer::UnitForget)
        }
        ParameterRole::Bias
            if matches!(
                kind,
                DeepLearningLayerKind::Lstm | DeepLearningLayerKind::Bilstm
            ) =>
        {
            matches!(
                builtin,
                BuiltinInitializer::UnitForget
                    | BuiltinInitializer::NarrowNormal
                    | BuiltinInitializer::Ones
            )
        }
        ParameterRole::Bias | ParameterRole::Scale | ParameterRole::Offset => {
            matches!(
                builtin,
                BuiltinInitializer::Zeros
                    | BuiltinInitializer::Ones
                    | BuiltinInitializer::NarrowNormal
            )
        }
    };
    if !supported {
        return Err(deep_learning_error(
            function,
            format!("{function}: {property} '{name}' is not supported for this layer"),
        ));
    }
    Ok(Initializer::Builtin(builtin))
}

fn exact_shape(
    shape: &ParameterShape,
    role: ParameterRole,
    function: &'static str,
) -> BuiltinResult<Vec<usize>> {
    let dimensions = match shape {
        ParameterShape::Matrix { rows, columns } => vec![
            *rows,
            columns.ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!(
                        "{function}: cannot initialize {} before its input width is known",
                        role.name()
                    ),
                )
            })?,
        ],
        ParameterShape::Dimensions(dimensions) => dimensions
            .iter()
            .map(|dimension| {
                dimension.ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!(
                            "{function}: cannot initialize {} before its channel count is known",
                            role.name()
                        ),
                    )
                })
            })
            .collect::<BuiltinResult<Vec<_>>>()?,
        ParameterShape::Elements(count) => vec![*count, 1],
    };
    let count = dimensions
        .iter()
        .try_fold(1usize, |product, value| product.checked_mul(*value))
        .ok_or_else(|| {
            deep_learning_error(
                function,
                format!("{function}: initialized {} shape overflows", role.name()),
            )
        })?;
    if count > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: initialized {} exceeds addressable storage",
                role.name()
            ),
        ));
    }
    Ok(dimensions)
}

fn admit_builtin(
    initializer: BuiltinInitializer,
    kind: DeepLearningLayerKind,
    shape: &[usize],
    function: &'static str,
) -> BuiltinResult<()> {
    match initializer {
        BuiltinInitializer::Glorot | BuiltinInitializer::He => {
            let (input, output) = generate::fans(shape, function)?;
            input.checked_add(output).ok_or_else(|| {
                deep_learning_error(
                    function,
                    format!("{function}: initializer fan geometry overflows"),
                )
            })?;
        }
        BuiltinInitializer::Orthogonal if shape.len() != 2 => {
            return Err(deep_learning_error(
                function,
                format!("{function}: orthogonal initializer requires a matrix"),
            ));
        }
        BuiltinInitializer::UnitForget => {
            let directions = if kind == DeepLearningLayerKind::Bilstm {
                2
            } else {
                1
            };
            if !matches!(shape, [rows, 1] if *rows % (4 * directions) == 0) {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: unit-forget initializer requires recurrent column bias"),
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

fn missing_shapes(
    rows: &[parameters::ParameterView<'_>],
    function: &'static str,
) -> BuiltinResult<HashMap<(usize, ParameterRole), Vec<usize>>> {
    let mut shapes = HashMap::new();
    for row in rows {
        let needs_value = match row.value {
            None => true,
            Some(value) => parameters::numeric_payload_shape(value, row.key.role.name(), function)?
                .contains(&0),
        };
        if needs_value {
            shapes.insert(
                (row.key.layer_index, row.key.role),
                exact_shape(&row.expected_shape, row.key.role, function)?,
            );
        }
    }
    Ok(shapes)
}

/// Only the geometry needed by training's target admission. Initializer
/// policy and callbacks are checked after the target has been admitted, but
/// before any initializer effect.
pub(super) fn planned_shapes(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<Vec<(usize, ParameterRole, Vec<usize>)>> {
    super::super::state::admit_initial(layers, function)?;
    let rows = parameters::view_layers(layers, function)?;
    let shapes = missing_shapes(&rows, function)?;
    Ok(rows
        .iter()
        .filter_map(|row| {
            shapes
                .get(&(row.key.layer_index, row.key.role))
                .map(|shape| (row.key.layer_index, row.key.role, shape.clone()))
        })
        .collect())
}

pub(super) fn tasks(layers: &[Value], function: &'static str) -> BuiltinResult<Vec<Task>> {
    // The canonical State schema and every existing parameter are admitted
    // before either a random draw or callback can occur.
    super::super::state::admit_initial(layers, function)?;
    let rows = parameters::view_layers(layers, function)?;
    let planned = missing_shapes(&rows, function)?;
    let mut layer_precision = vec![None; layers.len()];
    for row in &rows {
        if let Some(value) = row.value {
            if let Some(dtype @ (NumericDType::F32 | NumericDType::F64)) =
                parameters::floating_precision(value)
            {
                let prior = &mut layer_precision[row.key.layer_index];
                if prior.is_some_and(|known| known != dtype) {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: layer '{}' has mixed learnable precision",
                            row.key.layer_name
                        ),
                    ));
                }
                *prior = Some(dtype);
            }
        }
    }
    let mut tasks = Vec::new();
    tasks.try_reserve_exact(rows.len()).map_err(|_| {
        deep_learning_error(
            function,
            format!("{function}: cannot reserve initializer plan"),
        )
    })?;
    for row in rows {
        let Value::Object(layer) = &layers[row.key.layer_index] else {
            unreachable!("canonical parameter view admitted layer objects")
        };
        let kind = DeepLearningLayerKind::from_identity(layer.class_name())
            .expect("canonical parameter view admitted known classes");
        let initializer = parse_initializer(layer, kind, row.key.role, function)?;
        if let Some(shape) = planned.get(&(row.key.layer_index, row.key.role)) {
            if let Initializer::Builtin(builtin) = &initializer {
                admit_builtin(*builtin, kind, shape, function)?;
            }
            tasks.push(Task {
                layer_index: row.key.layer_index,
                kind,
                role: row.key.role,
                shape: shape.clone(),
                expected_shape: row.expected_shape,
                precision: layer_precision[row.key.layer_index],
                initializer,
            });
        }
    }
    Ok(tasks)
}
