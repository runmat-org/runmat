use super::*;
use runmat_builtins::fea_contract::study_types::descriptors::observation_target::*;

async fn region_target(
    name: &'static str,
    descriptor: &'static BuiltinDescriptor,
    region: Value,
    volume: bool,
) -> BuiltinResult<Value> {
    study_outputs(descriptor, name)?;
    let region = region_ref_from_value(name, &region)?;
    let target = if volume {
        RegionTarget::volume(region.region_id)
    } else {
        RegionTarget::boundary(region.region_id)
    };
    to_typed_value(
        name,
        TARGET_CLASS,
        ObservationTarget::Spatial(ExcitationTarget::Region(target)),
    )
}
#[runtime_builtin(
    name = "fea.ObservationTarget.boundary",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::targets"
)]
pub async fn target_boundary(region: Value) -> BuiltinResult<Value> {
    region_target(
        "fea.ObservationTarget.boundary",
        &FEA_OBSERVATION_TARGET_BOUNDARY_DESCRIPTOR,
        region,
        false,
    )
    .await
}
#[runtime_builtin(
    name = "fea.ObservationTarget.volume",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::targets"
)]
pub async fn target_volume(region: Value) -> BuiltinResult<Value> {
    region_target(
        "fea.ObservationTarget.volume",
        &FEA_OBSERVATION_TARGET_VOLUME_DESCRIPTOR,
        region,
        true,
    )
    .await
}
#[runtime_builtin(
    name = "fea.ObservationTarget.point",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::targets"
)]
pub async fn target_point(position: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ObservationTarget.point";
    study_outputs(&FEA_OBSERVATION_TARGET_POINT_DESCRIPTOR, NAME)?;
    let position = Position::new(values::position_meters(NAME, &position)?);
    to_typed_value(NAME, TARGET_CLASS, ObservationTarget::Point { position })
}
#[runtime_builtin(
    name = "fea.ObservationTarget.path",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::targets"
)]
pub async fn target_path(positions: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ObservationTarget.path";
    study_outputs(&FEA_OBSERVATION_TARGET_PATH_DESCRIPTOR, NAME)?;
    let Value::Tensor(tensor) = positions else {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "positions must be an N-by-3 host numeric matrix",
        ));
    };
    let shape = &tensor.shape();
    if shape.len() != 2 || shape[1] != 3 || shape[0] < 2 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "positions must be an N-by-3 matrix with N at least two",
        ));
    }
    let data = tensor.materialize_f64();
    if data.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "positions must contain finite coordinates",
        ));
    }
    let mut path = Vec::with_capacity(shape[0]);
    for row in 0..shape[0] {
        path.push(Position::new([
            data[row],
            data[row + shape[0]],
            data[row + 2 * shape[0]],
        ]));
    }
    to_typed_value(
        NAME,
        TARGET_CLASS,
        ObservationTarget::Path { positions: path },
    )
}
macro_rules! id_target {
    ($function:ident, $name:literal, $descriptor:ident, $value:expr) => {
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::observations::targets")]
        pub async fn $function(id: String) -> BuiltinResult<Value> {
            study_outputs(&$descriptor, $name)?;
            to_typed_value(
                $name,
                TARGET_CLASS,
                ($value)(nonempty_id($name, "target id", id)?),
            )
        }
    };
}
id_target!(
    target_interface,
    "fea.ObservationTarget.interface",
    FEA_OBSERVATION_TARGET_INTERFACE_DESCRIPTOR,
    |id| ObservationTarget::Spatial(ExcitationTarget::Interface { interface_id: id })
);
id_target!(
    target_port,
    "fea.ObservationTarget.port",
    FEA_OBSERVATION_TARGET_PORT_DESCRIPTOR,
    |id| ObservationTarget::Spatial(ExcitationTarget::Port { port_id: id })
);
id_target!(
    target_connector,
    "fea.ObservationTarget.connector",
    FEA_OBSERVATION_TARGET_CONNECTOR_DESCRIPTOR,
    |id| ObservationTarget::Connector { connector_id: id }
);
id_target!(
    target_constraint,
    "fea.ObservationTarget.constraint",
    FEA_OBSERVATION_TARGET_CONSTRAINT_DESCRIPTOR,
    |id| ObservationTarget::Constraint { constraint_id: id }
);
#[runtime_builtin(
    name = "fea.ObservationTarget.wholeModel",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::observations::targets"
)]
pub async fn target_whole() -> BuiltinResult<Value> {
    study_outputs(
        &FEA_OBSERVATION_TARGET_WHOLE_MODEL_DESCRIPTOR,
        "fea.ObservationTarget.wholeModel",
    )?;
    to_typed_value(
        "fea.ObservationTarget.wholeModel",
        TARGET_CLASS,
        ObservationTarget::WholeModel,
    )
}
