use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

macro_rules! descriptor {
    (
        $name:ident,
        $label:literal,
        $output:literal,
        $code:literal,
        $identifier:literal,
        $message:literal,
        [$(($input:literal, $ty:ident, $arity:ident, $description:literal)),* $(,)?]
    ) => {
        pub(super) const $name: BuiltinDescriptor = BuiltinDescriptor {
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
                    name: "result",
                    ty: BuiltinParamType::Any,
                    arity: BuiltinParamArity::Required,
                    default: None,
                    description: $output,
                }],
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &[BuiltinErrorDescriptor {
                code: $code,
                identifier: Some($identifier),
                when: "An argument is malformed, incompatible, non-finite, or violates the typed geometry contract.",
                message: $message,
            }],
        };
    };
}

descriptor!(
    BOX_DESCRIPTOR,
    "feature = geometry.box(width, depth, height)",
    "Immutable geometry.Feature box in SI units.",
    "RM.GEOMETRY.BOX.INVALID_INPUT",
    "RunMat:geometry:box:InvalidInput",
    "geometry.box: invalid input",
    [
        (
            "width",
            NumericScalar,
            Required,
            "Positive width in meters."
        ),
        (
            "depth",
            NumericScalar,
            Required,
            "Positive depth in meters."
        ),
        (
            "height",
            NumericScalar,
            Required,
            "Positive height in meters."
        ),
    ]
);
descriptor!(
    CYLINDER_DESCRIPTOR,
    "feature = geometry.cylinder(radius, height, axis, origin)",
    "Immutable geometry.Feature cylinder in SI units.",
    "RM.GEOMETRY.CYLINDER.INVALID_INPUT",
    "RunMat:geometry:cylinder:InvalidInput",
    "geometry.cylinder: invalid input",
    [
        (
            "radius",
            NumericScalar,
            Required,
            "Positive radius in meters."
        ),
        (
            "height",
            NumericScalar,
            Required,
            "Positive height in meters."
        ),
        (
            "axis",
            NumericArray,
            Required,
            "Finite three-component axis."
        ),
        (
            "origin",
            NumericArray,
            Required,
            "Finite three-component origin in meters."
        ),
    ]
);
descriptor!(
    SPHERE_DESCRIPTOR,
    "feature = geometry.sphere(radius, center)",
    "Immutable geometry.Feature sphere in SI units.",
    "RM.GEOMETRY.SPHERE.INVALID_INPUT",
    "RunMat:geometry:sphere:InvalidInput",
    "geometry.sphere: invalid input",
    [
        (
            "radius",
            NumericScalar,
            Required,
            "Positive radius in meters."
        ),
        (
            "center",
            NumericArray,
            Required,
            "Finite three-component center in meters."
        ),
    ]
);
descriptor!(
    TRANSFORM_DESCRIPTOR,
    "feature = geometry.transform(source, matrix)",
    "Transformed immutable geometry.Feature.",
    "RM.GEOMETRY.TRANSFORM.INVALID_INPUT",
    "RunMat:geometry:transform:InvalidInput",
    "geometry.transform: invalid input",
    [
        ("source", Any, Required, "Source geometry.Feature."),
        (
            "matrix",
            NumericArray,
            Required,
            "Finite 4-by-4 column-major affine transform."
        ),
    ]
);
descriptor!(
    INSTANCE_DESCRIPTOR,
    "feature = geometry.instance(source, transform, ...)",
    "Immutable geometry.Feature containing exact occurrences.",
    "RM.GEOMETRY.INSTANCE.INVALID_INPUT",
    "RunMat:geometry:instance:InvalidInput",
    "geometry.instance: invalid input",
    [
        ("source", Any, Required, "Source geometry.Feature."),
        (
            "transform",
            NumericArray,
            Variadic,
            "One or more finite 4-by-4 column-major affine transforms."
        ),
    ]
);
descriptor!(
    UNION_DESCRIPTOR,
    "feature = geometry.union(feature, feature, ...)",
    "Boolean-union geometry.Feature.",
    "RM.GEOMETRY.UNION.INVALID_INPUT",
    "RunMat:geometry:union:InvalidInput",
    "geometry.union: invalid input",
    [(
        "feature",
        Any,
        Variadic,
        "Two or more geometry.Feature operands."
    )]
);
descriptor!(
    INTERSECT_DESCRIPTOR,
    "feature = geometry.intersect(feature, feature, ...)",
    "Boolean-intersection geometry.Feature.",
    "RM.GEOMETRY.INTERSECT.INVALID_INPUT",
    "RunMat:geometry:intersect:InvalidInput",
    "geometry.intersect: invalid input",
    [(
        "feature",
        Any,
        Variadic,
        "Two or more geometry.Feature operands."
    )]
);
descriptor!(
    SUBTRACT_DESCRIPTOR,
    "feature = geometry.subtract(base, tool, ...)",
    "Boolean-subtraction geometry.Feature.",
    "RM.GEOMETRY.SUBTRACT.INVALID_INPUT",
    "RunMat:geometry:subtract:InvalidInput",
    "geometry.subtract: invalid input",
    [
        ("base", Any, Required, "Base geometry.Feature."),
        (
            "tool",
            Any,
            Variadic,
            "One or more tool geometry.Feature values."
        ),
    ]
);
descriptor!(
    FINISH_DESCRIPTOR,
    "generated = geometry.finish(feature)",
    "Immutable geometry.Generated product.",
    "RM.GEOMETRY.FINISH.INVALID_INPUT",
    "RunMat:geometry:finish:InvalidInput",
    "geometry.finish: invalid input",
    [("feature", Any, Required, "Final geometry.Feature result.")]
);
descriptor!(
    SOLID_DESCRIPTOR,
    "output = geometry.solid(feature)",
    "Exactly-one solid geometry.RegionOutput.",
    "RM.GEOMETRY.SOLID.INVALID_INPUT",
    "RunMat:geometry:solid:InvalidInput",
    "geometry.solid: invalid input",
    [("feature", Any, Required, "Source geometry.Feature.")]
);
descriptor!(
    FACES_DESCRIPTOR,
    "output = geometry.faces(feature)",
    "One-or-more face geometry.RegionOutput.",
    "RM.GEOMETRY.FACES.INVALID_INPUT",
    "RunMat:geometry:faces:InvalidInput",
    "geometry.faces: invalid input",
    [("feature", Any, Required, "Source geometry.Feature.")]
);
descriptor!(
    FACES_INTRODUCED_DESCRIPTOR,
    "output = geometry.facesIntroducedBy(feature)",
    "Face geometry.RegionOutput selected by topology history.",
    "RM.GEOMETRY.FACES_INTRODUCED_BY.INVALID_INPUT",
    "RunMat:geometry:facesIntroducedBy:InvalidInput",
    "geometry.facesIntroducedBy: invalid input",
    [(
        "feature",
        Any,
        Required,
        "Feature whose introduced faces are selected."
    )]
);
descriptor!(
    EDGES_INTRODUCED_DESCRIPTOR,
    "output = geometry.edgesIntroducedBy(feature)",
    "Edge geometry.RegionOutput selected by topology history.",
    "RM.GEOMETRY.EDGES_INTRODUCED_BY.INVALID_INPUT",
    "RunMat:geometry:edgesIntroducedBy:InvalidInput",
    "geometry.edgesIntroducedBy: invalid input",
    [(
        "feature",
        Any,
        Required,
        "Feature whose introduced edges are selected."
    )]
);
descriptor!(
    WITH_REGION_DESCRIPTOR,
    "generated = geometry.withRegion(generated, name, output)",
    "geometry.Generated product with one additional named region output.",
    "RM.GEOMETRY.WITH_REGION.INVALID_INPUT",
    "RunMat:geometry:withRegion:InvalidInput",
    "geometry.withRegion: invalid input",
    [
        (
            "generated",
            Any,
            Required,
            "Source geometry.Generated product."
        ),
        ("name", StringScalar, Required, "Unique output name."),
        (
            "output",
            Any,
            Required,
            "Typed geometry.RegionOutput value."
        ),
    ]
);
