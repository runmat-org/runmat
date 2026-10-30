use std::collections::HashMap;
use std::sync::Once;

use super::super::*;
use super::direction_from_value;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::values::position_meters;
use crate::class_registry::declaration::{immutable_properties, static_methods};
use crate::execution::value_codec::ImmutableValueClassSchema;
use runmat_analysis_core::{
    validate_coordinate_frame, AxialFrame, CartesianFrame, Position, SphericalFrame,
};

const COORDINATE_FRAME_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_COORDINATE_FRAME_CLASS.display_name();
const GLOBAL_FRAME_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_GLOBAL_COORDINATE_FRAME_CLASS
        .display_name();
const MATERIAL_FRAME_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_MATERIAL_COORDINATE_FRAME_CLASS
        .display_name();
const ELEMENT_FRAME_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_ELEMENT_COORDINATE_FRAME_CLASS
        .display_name();
const BOUNDARY_FRAME_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_BOUNDARY_NORMAL_TANGENT_FRAME_CLASS
        .display_name();
const CARTESIAN_FRAME_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_CARTESIAN_COORDINATE_FRAME_CLASS
        .display_name();
const CYLINDRICAL_FRAME_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_CYLINDRICAL_COORDINATE_FRAME_CLASS
        .display_name();
const SPHERICAL_FRAME_CLASS: &str =
    runmat_builtins::fea_contract::model_types::classes::FEA_SPHERICAL_COORDINATE_FRAME_CLASS
        .display_name();
const EMPTY_FIELDS: &[&str] = &[];
const CARTESIAN_FIELDS: &[&str] = &["frame_id", "origin", "x_axis", "y_axis"];
const AXIAL_FIELDS: &[&str] = &["frame_id", "origin", "axis", "radial_reference"];
const SPHERICAL_FIELDS: &[&str] = &["frame_id", "origin", "polar_axis", "azimuth_reference"];

macro_rules! simple_frame_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr) => {
        #[runtime_builtin(
                            name = $name,
                            binding_variant = "default",
                            context = true,
                            builtin_path = "crate::builtins::fea::constraints::support::frame"
                        )]
        pub async fn $function(
            runtime: Option<crate::context::RuntimeContext>,
        ) -> BuiltinResult<Value> {
            crate::builtins::fea::constraints::fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            ensure_frame_classes_registered();
            let object = crate::execution::object::construct_object_from_borrowed_fields(
                runtime.as_ref(),
                $class,
                std::iter::empty::<(&str, &Value)>(),
            )?;
            Ok(object.into_value_and_lease().0)
        }
    };
}

simple_frame_builtin!(
    coordinate_frame_global_builtin,
    FEA_COORDINATE_FRAME_GLOBAL_FRAME_DESCRIPTOR,
    "fea.CoordinateFrame.globalFrame",
    GLOBAL_FRAME_CLASS
);
simple_frame_builtin!(
    coordinate_frame_material_builtin,
    FEA_COORDINATE_FRAME_MATERIAL_DESCRIPTOR,
    "fea.CoordinateFrame.material",
    MATERIAL_FRAME_CLASS
);
simple_frame_builtin!(
    coordinate_frame_element_builtin,
    FEA_COORDINATE_FRAME_ELEMENT_DESCRIPTOR,
    "fea.CoordinateFrame.element",
    ELEMENT_FRAME_CLASS
);
simple_frame_builtin!(
    coordinate_frame_boundary_builtin,
    FEA_COORDINATE_FRAME_BOUNDARY_NORMAL_TANGENT_DESCRIPTOR,
    "fea.CoordinateFrame.boundaryNormalTangent",
    BOUNDARY_FRAME_CLASS
);

macro_rules! axis_frame_builtin {
    ($function:ident, $descriptor:ident, $name:literal, $class:expr, $build:expr, $primary:literal, $secondary:literal) => {
        #[runtime_builtin(
                            name = $name,
                            binding_variant = "default",
                            context = true,
                            builtin_path = "crate::builtins::fea::constraints::support::frame"
                        )]
        pub async fn $function(
            runtime: Option<crate::context::RuntimeContext>,
            id: runmat_value::Utf8Text,
            origin: Value,
            primary_axis: Value,
            secondary_axis: Value,
        ) -> BuiltinResult<Value> {
            crate::builtins::fea::constraints::fixed_outputs(&runmat_builtins::$descriptor, $name)?;
            let frame = ($build)(
                id.as_str().to_owned(),
                Position::new(position_meters($name, &origin)?),
                direction_from_value($name, &primary_axis)?,
                direction_from_value($name, &secondary_axis)?,
            );
            validate_coordinate_frame(&frame)
                .map_err(|error| builtin_error($name, &ERROR_INPUT, error.to_string()))?;
            require_orthogonal_axes($name, &frame)?;
            ensure_frame_classes_registered();
            let fields = [
                ("frame_id", Value::String(runmat_value::StringElement::Text(id))),
                ("origin", origin),
                ($primary, primary_axis),
                ($secondary, secondary_axis),
            ];
            let object = crate::execution::object::construct_object_from_borrowed_fields(
                runtime.as_ref(),
                $class,
                fields.iter().map(|(name, value)| (*name, value)),
            )?;
            Ok(object.into_value_and_lease().0)
        }
    };
}

axis_frame_builtin!(
    coordinate_frame_cartesian_builtin,
    FEA_COORDINATE_FRAME_CARTESIAN_DESCRIPTOR,
    "fea.CoordinateFrame.cartesian",
    CARTESIAN_FRAME_CLASS,
    |frame_id, origin, x_axis, y_axis| CoordinateFrame::Cartesian(CartesianFrame {
        frame_id,
        origin,
        x_axis,
        y_axis,
    }),
    "x_axis",
    "y_axis"
);
axis_frame_builtin!(
    coordinate_frame_cylindrical_builtin,
    FEA_COORDINATE_FRAME_CYLINDRICAL_DESCRIPTOR,
    "fea.CoordinateFrame.cylindrical",
    CYLINDRICAL_FRAME_CLASS,
    |frame_id, origin, axis, radial_reference| CoordinateFrame::Cylindrical(AxialFrame {
        frame_id,
        origin,
        axis,
        radial_reference,
    }),
    "axis",
    "radial_reference"
);
axis_frame_builtin!(
    coordinate_frame_spherical_builtin,
    FEA_COORDINATE_FRAME_SPHERICAL_DESCRIPTOR,
    "fea.CoordinateFrame.spherical",
    SPHERICAL_FRAME_CLASS,
    |frame_id, origin, polar_axis, azimuth_reference| CoordinateFrame::Spherical(SphericalFrame {
        frame_id,
        origin,
        polar_axis,
        azimuth_reference,
    }),
    "polar_axis",
    "azimuth_reference"
);

/// Cartesian and spherical bases use both directions as axes, so the
/// structural basis requires them orthogonal; reject the frame before it is
/// built instead of at solve time. Directions are already unit vectors.
const ORTHOGONALITY_TOLERANCE: f64 = 1.0e-10;

fn require_orthogonal_axes(builtin: &'static str, frame: &CoordinateFrame) -> BuiltinResult<()> {
    let (first, second, label) = match frame {
        CoordinateFrame::Cartesian(frame) => (frame.x_axis, frame.y_axis, "x and y axes"),
        CoordinateFrame::Spherical(frame) => (
            frame.polar_axis,
            frame.azimuth_reference,
            "polar axis and azimuth reference",
        ),
        _ => return Ok(()),
    };
    let cosine: f64 = first.iter().zip(second).map(|(a, b)| a * b).sum();
    if cosine.abs() > ORTHOGONALITY_TOLERANCE {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("coordinate-frame {label} must be orthogonal"),
        ));
    }
    Ok(())
}

pub(crate) fn coordinate_frame_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<CoordinateFrame> {
    let Value::Object(object) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "expected a typed fea.CoordinateFrame value",
        ));
    };
    let string = |name: &str| match object.properties().get(name) {
        Some(Value::String(runmat_value::StringElement::Text(value))) if !value.is_empty() => {
            Ok(value.to_string())
        }
        _ => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!(
                "{}.{} must be a non-empty string",
                object.class_name(),
                name
            ),
        )),
    };
    let nested = |name: &str| {
        object.properties().get(name).ok_or_else(|| {
            builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("{}.{} is required", object.class_name(), name),
            )
        })
    };
    let frame = match object.class_name().display_name() {
        GLOBAL_FRAME_CLASS => CoordinateFrame::Global,
        MATERIAL_FRAME_CLASS => CoordinateFrame::Material,
        ELEMENT_FRAME_CLASS => CoordinateFrame::Element,
        BOUNDARY_FRAME_CLASS => CoordinateFrame::BoundaryNormalTangent,
        CARTESIAN_FRAME_CLASS => CoordinateFrame::Cartesian(CartesianFrame {
            frame_id: string("frame_id")?,
            origin: Position::new(position_meters(builtin, nested("origin")?)?),
            x_axis: direction_from_value(builtin, nested("x_axis")?)?,
            y_axis: direction_from_value(builtin, nested("y_axis")?)?,
        }),
        CYLINDRICAL_FRAME_CLASS => CoordinateFrame::Cylindrical(AxialFrame {
            frame_id: string("frame_id")?,
            origin: Position::new(position_meters(builtin, nested("origin")?)?),
            axis: direction_from_value(builtin, nested("axis")?)?,
            radial_reference: direction_from_value(builtin, nested("radial_reference")?)?,
        }),
        SPHERICAL_FRAME_CLASS => CoordinateFrame::Spherical(SphericalFrame {
            frame_id: string("frame_id")?,
            origin: Position::new(position_meters(builtin, nested("origin")?)?),
            polar_axis: direction_from_value(builtin, nested("polar_axis")?)?,
            azimuth_reference: direction_from_value(builtin, nested("azimuth_reference")?)?,
        }),
        other => {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("unsupported coordinate-frame class `{other}`"),
            ))
        }
    };
    validate_coordinate_frame(&frame)
        .map_err(|error| builtin_error(builtin, &ERROR_INPUT, error.to_string()))?;
    require_orthogonal_axes(builtin, &frame)?;
    Ok(frame)
}

pub(super) fn ensure_frame_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            COORDINATE_FRAME_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let mut schemas = Vec::new();
        for (class_name, identity, fields, validate) in [
            (
                GLOBAL_FRAME_CLASS,
                "runmat.fea.frame.global",
                EMPTY_FIELDS,
                validate_empty_frame as _,
            ),
            (
                MATERIAL_FRAME_CLASS,
                "runmat.fea.frame.material",
                EMPTY_FIELDS,
                validate_empty_frame as _,
            ),
            (
                ELEMENT_FRAME_CLASS,
                "runmat.fea.frame.element",
                EMPTY_FIELDS,
                validate_empty_frame as _,
            ),
            (
                BOUNDARY_FRAME_CLASS,
                "runmat.fea.frame.boundary-normal-tangent",
                EMPTY_FIELDS,
                validate_empty_frame as _,
            ),
            (
                CARTESIAN_FRAME_CLASS,
                "runmat.fea.frame.cartesian",
                CARTESIAN_FIELDS,
                validate_cartesian_frame as _,
            ),
            (
                CYLINDRICAL_FRAME_CLASS,
                "runmat.fea.frame.cylindrical",
                AXIAL_FIELDS,
                validate_cylindrical_frame as _,
            ),
            (
                SPHERICAL_FRAME_CLASS,
                "runmat.fea.frame.spherical",
                SPHERICAL_FIELDS,
                validate_spherical_frame as _,
            ),
        ] {
            schemas.push(ImmutableValueClassSchema {
                class_name,
                type_identity: identity,
                schema_version: 1,
                allowed_fields: fields,
                required_fields: fields,
                validate,
            });
        }

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASSES.ensure(|| {
        for (class_name, fields) in [
            (GLOBAL_FRAME_CLASS, EMPTY_FIELDS),
            (MATERIAL_FRAME_CLASS, EMPTY_FIELDS),
            (ELEMENT_FRAME_CLASS, EMPTY_FIELDS),
            (BOUNDARY_FRAME_CLASS, EMPTY_FIELDS),
            (CARTESIAN_FRAME_CLASS, CARTESIAN_FIELDS),
            (CYLINDRICAL_FRAME_CLASS, AXIAL_FIELDS),
            (SPHERICAL_FRAME_CLASS, SPHERICAL_FIELDS),
        ] {
            register_value_class(class_name, immutable_properties(fields), HashMap::new());
        }
        register_value_class(
            COORDINATE_FRAME_CLASS,
            immutable_properties(EMPTY_FIELDS),
            static_methods([
                ("globalFrame", "fea.CoordinateFrame.globalFrame"),
                ("material", "fea.CoordinateFrame.material"),
                ("element", "fea.CoordinateFrame.element"),
                (
                    "boundaryNormalTangent",
                    "fea.CoordinateFrame.boundaryNormalTangent",
                ),
                ("cartesian", "fea.CoordinateFrame.cartesian"),
                ("cylindrical", "fea.CoordinateFrame.cylindrical"),
                ("spherical", "fea.CoordinateFrame.spherical"),
            ]),
        );
    });
}

fn validate_empty_frame(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    Ok(())
}

fn validate_cartesian_frame(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    validate_axis_frame_object(object, "x_axis", "y_axis")
}

fn validate_cylindrical_frame(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    validate_axis_frame_object(object, "axis", "radial_reference")
}

fn validate_spherical_frame(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    validate_axis_frame_object(object, "polar_axis", "azimuth_reference")
}

fn validate_axis_frame_object(
    object: &ObjectInstance,
    primary: &str,
    secondary: &str,
) -> Result<(), String> {
    let value = Value::Object(object.clone());
    coordinate_frame_from_value("fea.CoordinateFrame", &value)
        .map(|_| ())
        .map_err(|error| error.to_string())?;
    if primary == secondary {
        return Err("coordinate-frame axes must be distinct".into());
    }
    Ok(())
}
