use super::*;

const REGION_TARGET: [BuiltinParamDescriptor; 1] = [required(
    "region",
    BuiltinParamType::Any,
    "Typed geometry.Region.",
)];
const ID_TARGET: [BuiltinParamDescriptor; 1] = [required(
    "id",
    BuiltinParamType::StringScalar,
    "Stable target identifier.",
)];
const POINT_TARGET: [BuiltinParamDescriptor; 1] = [required(
    "position",
    BuiltinParamType::Any,
    "Typed fea.Position.",
)];
const PATH_TARGET: [BuiltinParamDescriptor; 1] = [required(
    "positions",
    BuiltinParamType::NumericArray,
    "N-by-3 path positions in meters.",
)];
descriptor!(
    TARGET_BOUNDARY_DESCRIPTOR,
    TARGET_BOUNDARY_SIGNATURES,
    "target = fea.ObservationTarget.boundary(region)",
    &REGION_TARGET
);
descriptor!(
    TARGET_VOLUME_DESCRIPTOR,
    TARGET_VOLUME_SIGNATURES,
    "target = fea.ObservationTarget.volume(region)",
    &REGION_TARGET
);
descriptor!(
    TARGET_POINT_DESCRIPTOR,
    TARGET_POINT_SIGNATURES,
    "target = fea.ObservationTarget.point(position)",
    &POINT_TARGET
);
descriptor!(
    TARGET_PATH_DESCRIPTOR,
    TARGET_PATH_SIGNATURES,
    "target = fea.ObservationTarget.path(positions)",
    &PATH_TARGET
);
descriptor!(
    TARGET_INTERFACE_DESCRIPTOR,
    TARGET_INTERFACE_SIGNATURES,
    "target = fea.ObservationTarget.interface(id)",
    &ID_TARGET
);
descriptor!(
    TARGET_PORT_DESCRIPTOR,
    TARGET_PORT_SIGNATURES,
    "target = fea.ObservationTarget.port(id)",
    &ID_TARGET
);
descriptor!(
    TARGET_CONNECTOR_DESCRIPTOR,
    TARGET_CONNECTOR_SIGNATURES,
    "target = fea.ObservationTarget.connector(id)",
    &ID_TARGET
);
descriptor!(
    TARGET_CONSTRAINT_DESCRIPTOR,
    TARGET_CONSTRAINT_SIGNATURES,
    "target = fea.ObservationTarget.constraint(id)",
    &ID_TARGET
);
descriptor!(
    TARGET_WHOLE_DESCRIPTOR,
    TARGET_WHOLE_SIGNATURES,
    "target = fea.ObservationTarget.wholeModel()",
    &NO_INPUTS
);

async fn region_target(name: &'static str, region: Value, volume: bool) -> BuiltinResult<Value> {
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
    category = "fea",
    summary = "Target a geometry boundary region for observation.",
    keywords = "fea,observation,target,boundary,typed",
    descriptor(TARGET_BOUNDARY_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_boundary(region: Value) -> BuiltinResult<Value> {
    region_target("fea.ObservationTarget.boundary", region, false).await
}
#[runtime_builtin(
    name = "fea.ObservationTarget.volume",
    category = "fea",
    summary = "Target a geometry volume region for observation.",
    keywords = "fea,observation,target,volume,typed",
    descriptor(TARGET_VOLUME_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_volume(region: Value) -> BuiltinResult<Value> {
    region_target("fea.ObservationTarget.volume", region, true).await
}
#[runtime_builtin(
    name = "fea.ObservationTarget.point",
    category = "fea",
    summary = "Target a point for observation.",
    keywords = "fea,observation,target,point,typed",
    descriptor(TARGET_POINT_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_point(position: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ObservationTarget.point";
    let position = Position::new(values::position_meters(NAME, &position)?);
    to_typed_value(NAME, TARGET_CLASS, ObservationTarget::Point { position })
}
#[runtime_builtin(
    name = "fea.ObservationTarget.path",
    category = "fea",
    summary = "Target an ordered path for observation.",
    keywords = "fea,observation,target,path,typed",
    descriptor(TARGET_PATH_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_path(positions: Value) -> BuiltinResult<Value> {
    const NAME: &str = "fea.ObservationTarget.path";
    let Value::Tensor(tensor) = positions else {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "positions must be an N-by-3 host numeric matrix",
        ));
    };
    let shape = &tensor.shape;
    if shape.len() != 2 || shape[1] != 3 || shape[0] < 2 {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "positions must be an N-by-3 matrix with N at least two",
        ));
    }
    let data = tensor.materialize_f64();
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
        #[runtime_builtin(name = $name, category = "fea", summary = "Create a typed named observation target.", keywords = "fea,observation,target,typed", descriptor($descriptor), integer_audit(STRUCTURAL_AUDIT), builtin_path = "crate::builtins::fea::observations")]
        pub async fn $function(id: String) -> BuiltinResult<Value> {
            to_typed_value($name, TARGET_CLASS, ($value)(nonempty_id($name, "target id", id)?))
        }
    };
}
id_target!(
    target_interface,
    "fea.ObservationTarget.interface",
    TARGET_INTERFACE_DESCRIPTOR,
    |id| ObservationTarget::Spatial(ExcitationTarget::Interface { interface_id: id })
);
id_target!(
    target_port,
    "fea.ObservationTarget.port",
    TARGET_PORT_DESCRIPTOR,
    |id| ObservationTarget::Spatial(ExcitationTarget::Port { port_id: id })
);
id_target!(
    target_connector,
    "fea.ObservationTarget.connector",
    TARGET_CONNECTOR_DESCRIPTOR,
    |id| ObservationTarget::Connector { connector_id: id }
);
id_target!(
    target_constraint,
    "fea.ObservationTarget.constraint",
    TARGET_CONSTRAINT_DESCRIPTOR,
    |id| ObservationTarget::Constraint { constraint_id: id }
);
#[runtime_builtin(
    name = "fea.ObservationTarget.wholeModel",
    category = "fea",
    summary = "Target the whole analysis model for observation.",
    keywords = "fea,observation,target,whole,model,typed",
    descriptor(TARGET_WHOLE_DESCRIPTOR),
    integer_audit(STRUCTURAL_AUDIT),
    builtin_path = "crate::builtins::fea::observations"
)]
pub async fn target_whole() -> BuiltinResult<Value> {
    to_typed_value(
        "fea.ObservationTarget.wholeModel",
        TARGET_CLASS,
        ObservationTarget::WholeModel,
    )
}
