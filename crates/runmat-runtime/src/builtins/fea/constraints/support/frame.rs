use std::collections::HashMap;
use std::sync::Once;

use super::super::*;
use super::direction_from_value;
use crate::builtins::fea::values::position_meters;
use crate::class_registry::declaration::{
    immutable_properties, register_value_class, static_methods,
};
use crate::execution::value_codec::{register_immutable_value_class, ImmutableValueClassSchema};
use runmat_analysis_core::{
    validate_coordinate_frame, AxialFrame, CartesianFrame, Position, SphericalFrame,
};

const COORDINATE_FRAME_CLASS: &str = "fea.CoordinateFrame";
const GLOBAL_FRAME_CLASS: &str = "fea.GlobalCoordinateFrame";
const MATERIAL_FRAME_CLASS: &str = "fea.MaterialCoordinateFrame";
const ELEMENT_FRAME_CLASS: &str = "fea.ElementCoordinateFrame";
const BOUNDARY_FRAME_CLASS: &str = "fea.BoundaryNormalTangentFrame";
const CARTESIAN_FRAME_CLASS: &str = "fea.CartesianCoordinateFrame";
const CYLINDRICAL_FRAME_CLASS: &str = "fea.CylindricalCoordinateFrame";
const SPHERICAL_FRAME_CLASS: &str = "fea.SphericalCoordinateFrame";
const EMPTY_FIELDS: &[&str] = &[];
const CARTESIAN_FIELDS: &[&str] = &["frame_id", "origin", "x_axis", "y_axis"];
const AXIAL_FIELDS: &[&str] = &["frame_id", "origin", "axis", "radial_reference"];
const SPHERICAL_FIELDS: &[&str] = &["frame_id", "origin", "polar_axis", "azimuth_reference"];

macro_rules! simple_frame_builtin {
    ($function:ident, $descriptor:ident, $signatures:ident, $name:literal, $label:literal, $class:expr) => {
        #[runtime_builtin(
                            name = $name,
                            builtin_path = "crate::builtins::fea::constraints::support::frame",
                            context = true,
                            binding_variant = "default"
                        )]
        pub async fn $function(
            runtime: Option<crate::context::RuntimeContext>,
        ) -> BuiltinResult<Value> {
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
    GLOBAL_FRAME_DESCRIPTOR,
    GLOBAL_FRAME_SIGNATURES,
    "fea.CoordinateFrame.globalFrame",
    "frame = fea.CoordinateFrame.globalFrame()",
    GLOBAL_FRAME_CLASS
);
simple_frame_builtin!(
    coordinate_frame_material_builtin,
    MATERIAL_FRAME_DESCRIPTOR,
    MATERIAL_FRAME_SIGNATURES,
    "fea.CoordinateFrame.material",
    "frame = fea.CoordinateFrame.material()",
    MATERIAL_FRAME_CLASS
);
simple_frame_builtin!(
    coordinate_frame_element_builtin,
    ELEMENT_FRAME_DESCRIPTOR,
    ELEMENT_FRAME_SIGNATURES,
    "fea.CoordinateFrame.element",
    "frame = fea.CoordinateFrame.element()",
    ELEMENT_FRAME_CLASS
);
simple_frame_builtin!(
    coordinate_frame_boundary_builtin,
    BOUNDARY_FRAME_DESCRIPTOR,
    BOUNDARY_FRAME_SIGNATURES,
    "fea.CoordinateFrame.boundaryNormalTangent",
    "frame = fea.CoordinateFrame.boundaryNormalTangent()",
    BOUNDARY_FRAME_CLASS
);

macro_rules! axis_frame_builtin {
    ($function:ident, $descriptor:ident, $signatures:ident, $name:literal, $label:literal, $class:expr, $build:expr, $primary:literal, $secondary:literal) => {
        #[runtime_builtin(
                            name = $name,
                            builtin_path = "crate::builtins::fea::constraints::support::frame",
                            context = true,
                            binding_variant = "default"
                        )]
        pub async fn $function(
            runtime: Option<crate::context::RuntimeContext>,
            id: runmat_value::Utf8Text,
            origin: Value,
            primary_axis: Value,
            secondary_axis: Value,
        ) -> BuiltinResult<Value> {
            let frame = ($build)(
                id.as_str().to_owned(),
                Position::new(position_meters($name, &origin)?),
                direction_from_value($name, &primary_axis)?,
                direction_from_value($name, &secondary_axis)?,
            );
            validate_coordinate_frame(&frame)
                .map_err(|error| builtin_error($name, &ERROR_INPUT, error.to_string()))?;
            ensure_frame_classes_registered();
            let fields = [
                ("frame_id", Value::String(id)),
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
    CARTESIAN_FRAME_DESCRIPTOR,
    CARTESIAN_FRAME_SIGNATURES,
    "fea.CoordinateFrame.cartesian",
    "frame = fea.CoordinateFrame.cartesian(id, origin, xAxis, yAxis)",
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
    CYLINDRICAL_FRAME_DESCRIPTOR,
    CYLINDRICAL_FRAME_SIGNATURES,
    "fea.CoordinateFrame.cylindrical",
    "frame = fea.CoordinateFrame.cylindrical(id, origin, axis, radialReference)",
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
    SPHERICAL_FRAME_DESCRIPTOR,
    SPHERICAL_FRAME_SIGNATURES,
    "fea.CoordinateFrame.spherical",
    "frame = fea.CoordinateFrame.spherical(id, origin, polarAxis, azimuthReference)",
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
        Some(Value::String(value)) if !value.is_empty() => Ok(value.to_string()),
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
    Ok(frame)
}

pub(super) fn ensure_frame_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            COORDINATE_FRAME_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
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
            register_immutable_value_class(ImmutableValueClassSchema {
                class_name,
                type_identity: identity,
                schema_version: 1,
                allowed_fields: fields,
                required_fields: fields,
                validate,
            })
            .unwrap_or_else(|error| panic!("failed to register typed coordinate frame: {error}"));
        }
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

fn validate_empty_frame(_: &ObjectInstance) -> Result<(), String> {
    Ok(())
}

fn validate_cartesian_frame(object: &ObjectInstance) -> Result<(), String> {
    validate_axis_frame_object(object, "x_axis", "y_axis")
}

fn validate_cylindrical_frame(object: &ObjectInstance) -> Result<(), String> {
    validate_axis_frame_object(object, "axis", "radial_reference")
}

fn validate_spherical_frame(object: &ObjectInstance) -> Result<(), String> {
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
