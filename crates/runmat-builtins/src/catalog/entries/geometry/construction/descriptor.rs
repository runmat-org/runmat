//! Canonical descriptors for the programmatic exact-geometry builder.
//!
//! These facts were owned by `runmat-runtime/src/builtins/geometry/generated/metadata.rs`;
//! the runtime binding re-exports them from here.

use crate::*;

pub const GEOMETRY_FEATURE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("geometry.Feature");
pub const GEOMETRY_GENERATED_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("geometry.Generated");
pub const GEOMETRY_REGION_OUTPUT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("geometry.RegionOutput");

macro_rules! geometry_descriptor {
    (
        $name:ident,
        $error:ident,
        $label:literal,
        $output_name:literal,
        $output:literal,
        $code:literal,
        $identifier:literal,
        $message:literal,
        $when:literal,
        [$(($input:literal, $ty:ident, $arity:ident, $description:literal)),* $(,)?],
        []
    ) => {
        pub const $error: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: $code,
            identifier: Some($identifier),
            // Every failure of the builder reports this one identifier.
            when: concat!(
                "The wrong number of arguments is passed. ",
                $when,
                " A value violates the feature-graph contract."
            ),
            message: $message,
        };
        pub const $name: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: &[$(BuiltinParamDescriptor {
                    name: $input,
                    ty: BuiltinParamType::$ty,
                    arity: BuiltinParamArity::$arity,
                    default: None,
                    description: $description,
                }),*],
                outputs: &[BuiltinParamDescriptor {
                    name: $output_name,
                    ty: BuiltinParamType::Any,
                    arity: BuiltinParamArity::Required,
                    default: None,
                    description: $output,
                }],
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[$error],
        };
    };
}

geometry_descriptor!(
    GEOMETRY_BOX_DESCRIPTOR,
    GEOMETRY_BOX_ERROR_INPUT,
    "feature = geometry.box(width, depth, height)",
    "feature",
    "geometry.Feature for an axis-aligned box with one corner at the origin.",
    "RM.GEOMETRY.BOX.INVALID_INPUT",
    "RunMat:geometry:box:InvalidInput",
    "geometry.box: invalid input",
    "A dimension is not a real numeric scalar.",
    [
        (
            "width",
            NumericScalar,
            Required,
            "Positive extent along +x, in meters."
        ),
        (
            "depth",
            NumericScalar,
            Required,
            "Positive extent along +y, in meters."
        ),
        (
            "height",
            NumericScalar,
            Required,
            "Positive extent along +z, in meters."
        ),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_CYLINDER_DESCRIPTOR,
    GEOMETRY_CYLINDER_ERROR_INPUT,
    "feature = geometry.cylinder(radius, height, axis, origin)",
    "feature",
    "geometry.Feature for a right circular cylinder.",
    "RM.GEOMETRY.CYLINDER.INVALID_INPUT",
    "RunMat:geometry:cylinder:InvalidInput",
    "geometry.cylinder: invalid input",
    "A dimension is not a real numeric scalar, or axis or origin is not a finite three-element real vector.",
    [
        ("radius", NumericScalar, Required, "Positive radius in meters."),
        ("height", NumericScalar, Required, "Positive length along the axis, in meters."),
        ("axis", NumericArray, Required, "Finite nonzero three-element axis direction."),
        ("origin", NumericArray, Required, "Center of the base circle, in meters."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_SPHERE_DESCRIPTOR,
    GEOMETRY_SPHERE_ERROR_INPUT,
    "feature = geometry.sphere(radius, center)",
    "feature",
    "geometry.Feature for a sphere.",
    "RM.GEOMETRY.SPHERE.INVALID_INPUT",
    "RunMat:geometry:sphere:InvalidInput",
    "geometry.sphere: invalid input",
    "The radius is not a real numeric scalar, or center is not a finite three-element real vector.",
    [
        (
            "radius",
            NumericScalar,
            Required,
            "Positive radius in meters."
        ),
        ("center", NumericArray, Required, "Sphere center in meters."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_TRANSFORM_DESCRIPTOR,
    GEOMETRY_TRANSFORM_ERROR_INPUT,
    "feature = geometry.transform(source, matrix)",
    "feature",
    "geometry.Feature whose result is the transformed source.",
    "RM.GEOMETRY.TRANSFORM.INVALID_INPUT",
    "RunMat:geometry:transform:InvalidInput",
    "geometry.transform: invalid input",
    "source is not a geometry.Feature, or matrix is not a finite real 4-by-4 matrix whose final row is [0 0 0 1].",
    [
        ("source", Any, Required, "geometry.Feature to transform."),
        (
            "matrix",
            NumericArray,
            Required,
            "4-by-4 homogeneous affine transform: T(1:3,1:3) linear part, T(1:3,4) translation in meters, final row [0 0 0 1]."
        ),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_INSTANCE_DESCRIPTOR,
    GEOMETRY_INSTANCE_ERROR_INPUT,
    "feature = geometry.instance(source, transform1, transform2, ...)",
    "feature",
    "geometry.Feature containing one transformed occurrence of source per transform.",
    "RM.GEOMETRY.INSTANCE.INVALID_INPUT",
    "RunMat:geometry:instance:InvalidInput",
    "geometry.instance: invalid input",
    "source is not a geometry.Feature, no transform is given, or a transform is not a finite real 4-by-4 matrix whose final row is [0 0 0 1].",
    [
        ("source", Any, Required, "geometry.Feature to place."),
        ("transform", NumericArray, Variadic, "One or more 4-by-4 homogeneous affine transforms; translation in T(1:3,4), in meters."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_EXTRUDE_DESCRIPTOR,
    GEOMETRY_EXTRUDE_ERROR_INPUT,
    "feature = geometry.extrude(profile, distance, direction)",
    "feature",
    "geometry.Feature for the linear sweep of profile.",
    "RM.GEOMETRY.EXTRUDE.INVALID_INPUT",
    "RunMat:geometry:extrude:InvalidInput",
    "geometry.extrude: invalid input",
    "profile is not a geometry.Feature, distance is not a real numeric scalar, or direction is not a finite three-element real vector.",
    [
        ("profile", Any, Required, "geometry.Feature whose result is a face or sheet."),
        ("distance", NumericScalar, Required, "Positive sweep length in meters."),
        ("direction", NumericArray, Required, "Finite nonzero three-element sweep direction."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_REVOLVE_DESCRIPTOR,
    GEOMETRY_REVOLVE_ERROR_INPUT,
    "feature = geometry.revolve(profile, axis, origin, angle)",
    "feature",
    "geometry.Feature for the rotational sweep of profile.",
    "RM.GEOMETRY.REVOLVE.INVALID_INPUT",
    "RunMat:geometry:revolve:InvalidInput",
    "geometry.revolve: invalid input",
    "profile is not a geometry.Feature, axis or origin is not a finite three-element real vector, or angle is not a real numeric scalar.",
    [
        ("profile", Any, Required, "geometry.Feature whose result is a face or sheet."),
        ("axis", NumericArray, Required, "Finite nonzero three-element rotation-axis direction."),
        ("origin", NumericArray, Required, "Point on the rotation axis, in meters."),
        ("angle", NumericScalar, Required, "Sweep angle in radians, in (0, 2*pi]."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_HOLE_DESCRIPTOR,
    GEOMETRY_HOLE_ERROR_INPUT,
    "feature = geometry.hole(target, radius, depth, axis, origin)",
    "feature",
    "geometry.Feature for target with a cylindrical hole removed.",
    "RM.GEOMETRY.HOLE.INVALID_INPUT",
    "RunMat:geometry:hole:InvalidInput",
    "geometry.hole: invalid input",
    "target is not a geometry.Feature, a dimension is not a real numeric scalar, or axis or origin is not a finite three-element real vector.",
    [
        ("target", Any, Required, "geometry.Feature to cut."),
        ("radius", NumericScalar, Required, "Positive hole radius in meters."),
        ("depth", NumericScalar, Required, "Positive hole depth along the axis, in meters."),
        ("axis", NumericArray, Required, "Finite nonzero three-element drilling direction."),
        ("origin", NumericArray, Required, "Center of the hole entry circle, in meters."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_POCKET_DESCRIPTOR,
    GEOMETRY_POCKET_ERROR_INPUT,
    "feature = geometry.pocket(target, profile, depth, direction)",
    "feature",
    "geometry.Feature for target with the swept profile removed.",
    "RM.GEOMETRY.POCKET.INVALID_INPUT",
    "RunMat:geometry:pocket:InvalidInput",
    "geometry.pocket: invalid input",
    "target or profile is not a geometry.Feature, depth is not a real numeric scalar, direction is not a finite three-element real vector, or the operand graphs define one feature identity differently.",
    [
        ("target", Any, Required, "geometry.Feature to cut."),
        ("profile", Any, Required, "geometry.Feature whose result is a face or sheet."),
        ("depth", NumericScalar, Required, "Positive sweep length in meters."),
        ("direction", NumericArray, Required, "Finite nonzero three-element sweep direction."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_FILLET_DESCRIPTOR,
    GEOMETRY_FILLET_ERROR_INPUT,
    "feature = geometry.fillet(target, edges, radius)",
    "feature",
    "geometry.Feature for target with the selected edges rounded.",
    "RM.GEOMETRY.FILLET.INVALID_INPUT",
    "RunMat:geometry:fillet:InvalidInput",
    "geometry.fillet: invalid input",
    "target is not a geometry.Feature, edges is not an edge geometry.RegionOutput, or radius is not a real numeric scalar.",
    [
        ("target", Any, Required, "geometry.Feature to modify."),
        ("edges", Any, Required, "Edge selection from geometry.edges or geometry.edgesIntroducedBy."),
        ("radius", NumericScalar, Required, "Positive fillet radius in meters."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_CHAMFER_DESCRIPTOR,
    GEOMETRY_CHAMFER_ERROR_INPUT,
    "feature = geometry.chamfer(target, edges, distance)",
    "feature",
    "geometry.Feature for target with the selected edges beveled.",
    "RM.GEOMETRY.CHAMFER.INVALID_INPUT",
    "RunMat:geometry:chamfer:InvalidInput",
    "geometry.chamfer: invalid input",
    "target is not a geometry.Feature, edges is not an edge geometry.RegionOutput, or distance is not a real numeric scalar.",
    [
        ("target", Any, Required, "geometry.Feature to modify."),
        ("edges", Any, Required, "Edge selection from geometry.edges or geometry.edgesIntroducedBy."),
        ("distance", NumericScalar, Required, "Positive symmetric chamfer distance in meters."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_UNION_DESCRIPTOR,
    GEOMETRY_UNION_ERROR_INPUT,
    "feature = geometry.union(feature1, feature2, ...)",
    "feature",
    "geometry.Feature for the Boolean union of all operands.",
    "RM.GEOMETRY.UNION.INVALID_INPUT",
    "RunMat:geometry:union:InvalidInput",
    "geometry.union: invalid input",
    "Fewer than two operands are given, an operand is not a geometry.Feature, or the operand graphs define one feature identity differently.",
    [("feature", Any, Variadic, "Two or more geometry.Feature operands.")],
    []
);
geometry_descriptor!(
    GEOMETRY_INTERSECT_DESCRIPTOR,
    GEOMETRY_INTERSECT_ERROR_INPUT,
    "feature = geometry.intersect(feature1, feature2, ...)",
    "feature",
    "geometry.Feature for the volume common to all operands.",
    "RM.GEOMETRY.INTERSECT.INVALID_INPUT",
    "RunMat:geometry:intersect:InvalidInput",
    "geometry.intersect: invalid input",
    "Fewer than two operands are given, an operand is not a geometry.Feature, or the operand graphs define one feature identity differently.",
    [("feature", Any, Variadic, "Two or more geometry.Feature operands.")],
    []
);
geometry_descriptor!(
    GEOMETRY_SUBTRACT_DESCRIPTOR,
    GEOMETRY_SUBTRACT_ERROR_INPUT,
    "feature = geometry.subtract(base, tool1, tool2, ...)",
    "feature",
    "geometry.Feature for base with every tool removed.",
    "RM.GEOMETRY.SUBTRACT.INVALID_INPUT",
    "RunMat:geometry:subtract:InvalidInput",
    "geometry.subtract: invalid input",
    "No tool is given, an operand is not a geometry.Feature, or the operand graphs define one feature identity differently.",
    [
        ("base", Any, Required, "geometry.Feature to cut."),
        ("tool", Any, Variadic, "One or more geometry.Feature tools."),
    ],
    []
);
geometry_descriptor!(
    GEOMETRY_FINISH_DESCRIPTOR,
    GEOMETRY_FINISH_ERROR_INPUT,
    "part = geometry.finish(feature)",
    "part",
    "geometry.Generated product with the feature as its result and no named regions.",
    "RM.GEOMETRY.FINISH.INVALID_INPUT",
    "RunMat:geometry:finish:InvalidInput",
    "geometry.finish: invalid input",
    "The argument is not a geometry.Feature.",
    [(
        "feature",
        Any,
        Required,
        "Final geometry.Feature of the part."
    )],
    []
);
geometry_descriptor!(
    GEOMETRY_SOLID_DESCRIPTOR,
    GEOMETRY_SOLID_ERROR_INPUT,
    "output = geometry.solid(feature)",
    "output",
    "Solid geometry.RegionOutput that must resolve to exactly one solid.",
    "RM.GEOMETRY.SOLID.INVALID_INPUT",
    "RunMat:geometry:solid:InvalidInput",
    "geometry.solid: invalid input",
    "The argument is not a geometry.Feature.",
    [(
        "feature",
        Any,
        Required,
        "geometry.Feature whose result solid is selected."
    )],
    []
);
geometry_descriptor!(
    GEOMETRY_FACES_DESCRIPTOR,
    GEOMETRY_FACES_ERROR_INPUT,
    "output = geometry.faces(feature)",
    "output",
    "Face geometry.RegionOutput that must resolve to one or more faces.",
    "RM.GEOMETRY.FACES.INVALID_INPUT",
    "RunMat:geometry:faces:InvalidInput",
    "geometry.faces: invalid input",
    "The argument is not a geometry.Feature.",
    [(
        "feature",
        Any,
        Required,
        "geometry.Feature whose result faces are selected."
    )],
    []
);
geometry_descriptor!(
    GEOMETRY_EDGES_DESCRIPTOR,
    GEOMETRY_EDGES_ERROR_INPUT,
    "output = geometry.edges(feature)",
    "output",
    "Edge geometry.RegionOutput that must resolve to one or more edges.",
    "RM.GEOMETRY.EDGES.INVALID_INPUT",
    "RunMat:geometry:edges:InvalidInput",
    "geometry.edges: invalid input",
    "The argument is not a geometry.Feature.",
    [(
        "feature",
        Any,
        Required,
        "geometry.Feature whose result edges are selected."
    )],
    []
);
geometry_descriptor!(
    GEOMETRY_FACES_INTRODUCED_BY_DESCRIPTOR,
    GEOMETRY_FACES_INTRODUCED_BY_ERROR_INPUT,
    "output = geometry.facesIntroducedBy(feature)",
    "output",
    "Face geometry.RegionOutput selected by the feature's topology history.",
    "RM.GEOMETRY.FACES_INTRODUCED_BY.INVALID_INPUT",
    "RunMat:geometry:facesIntroducedBy:InvalidInput",
    "geometry.facesIntroducedBy: invalid input",
    "The argument is not a geometry.Feature.",
    [(
        "feature",
        Any,
        Required,
        "geometry.Feature whose introduced faces are selected."
    )],
    []
);
geometry_descriptor!(
    GEOMETRY_EDGES_INTRODUCED_BY_DESCRIPTOR,
    GEOMETRY_EDGES_INTRODUCED_BY_ERROR_INPUT,
    "output = geometry.edgesIntroducedBy(feature)",
    "output",
    "Edge geometry.RegionOutput selected by the feature's topology history.",
    "RM.GEOMETRY.EDGES_INTRODUCED_BY.INVALID_INPUT",
    "RunMat:geometry:edgesIntroducedBy:InvalidInput",
    "geometry.edgesIntroducedBy: invalid input",
    "The argument is not a geometry.Feature.",
    [(
        "feature",
        Any,
        Required,
        "geometry.Feature whose introduced edges are selected."
    )],
    []
);
geometry_descriptor!(
    GEOMETRY_WITH_REGION_DESCRIPTOR,
    GEOMETRY_WITH_REGION_ERROR_INPUT,
    "part = geometry.withRegion(part, name, output)",
    "part",
    "geometry.Generated product with one additional named region output.",
    "RM.GEOMETRY.WITH_REGION.INVALID_INPUT",
    "RunMat:geometry:withRegion:InvalidInput",
    "geometry.withRegion: invalid input",
    "part is not a geometry.Generated value, output is not a geometry.RegionOutput or names a feature outside the part, or name is not scalar text, is blank, longer than 256 bytes, contains a control character, or is already declared.",
    [
        ("part", Any, Required, "geometry.Generated product from geometry.finish."),
        ("name", StringScalar, Required, "Unique region-output name."),
        ("output", Any, Required, "geometry.RegionOutput from a selection builtin."),
    ],
    []
);

pub const GEOMETRY_CONSTRUCTION_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "The geometry builder stores SI lengths, angles, vectors, and transforms as binary64. Integer-class scalars cross that boundary when exactly representable; integer-class vectors and matrices are rejected. No operation computes in an integer domain or returns integer data.",
    };
