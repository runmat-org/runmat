//! MATLAB-compatible colormap array generators.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Type,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::surface::ColorMap;
use runmat_value::{IntValue, Tensor, Value};

use super::state::current_colormap_length;
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const MAX_COLORMAP_LENGTH: usize = 1_000_000;

const OUTPUT_CMAP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "m-by-3 RGB colormap array with values in [0, 1].",
}];

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const INPUTS_LENGTH: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("current colormap length or 256"),
    description: "Number of colors as a nonnegative integer.",
}];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORMAP_ARRAY.INVALID_ARGUMENT",
    identifier: Some("RunMat:colormapArray:InvalidArgument"),
    when: "The requested colormap length or RGB colormap array is invalid.",
    message: "colormap array: invalid argument",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_ARGUMENT];

const COLORCUBE_INTEGER_LENGTH_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "m",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented nonnegative scalar length accepts every built-in integer class.",
    }];

pub const COLORCUBE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "c = colorcube(integer_m)",
        inputs: &COLORCUBE_INTEGER_LENGTH_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "m is read exactly from authoritative host integer storage, validated against RunMat's allocation guard, and used only as the row count. The generated m-by-3 RGB colormap is double; resident scalar controls are rejected without provider dispatch.",
    }];

pub const PARULA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "c = parula(integer_m)",
        inputs: &COLORCUBE_INTEGER_LENGTH_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility target explicitly lists every built-in integer class for m. The exact nonnegative length is read from authoritative host storage and used only as a guarded row count; the generated m-by-3 RGB colormap is double, and resident controls reject without provider dispatch.",
    }];

fn colormap_type(_args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    Type::Tensor { shape: None }
}

#[rustfmt::skip]
macro_rules! define_colormap_builtin {
    (
        $descriptor:ident,
        $signatures:ident,
        $fn_name:ident,
        $name:literal,
        $variant:path,
        $summary:literal,
        $keywords:literal,
        $label_none:literal,
        $label_length:literal
        $(, $integer_capabilities:path)?
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 2] = [
            BuiltinSignatureDescriptor {
                label: $label_none,
                inputs: &INPUTS_NONE,
                outputs: &OUTPUT_CMAP,
            },
            BuiltinSignatureDescriptor {
                label: $label_length,
                inputs: &INPUTS_LENGTH,
                outputs: &OUTPUT_CMAP,
            },
        ];

        pub const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };

        #[runtime_builtin(
            name = $name,
            category = "plotting",
            summary = $summary,
            keywords = $keywords,
            type_resolver(colormap_type),
            descriptor(crate::builtins::plotting::colormap_arrays::$descriptor),
            $(integer_capabilities($integer_capabilities),)?
            builtin_path = "crate::builtins::plotting::colormap_arrays"
        )]
        pub fn $fn_name(args: Vec<Value>) -> BuiltinResult<Value> {
            let len = parse_length_arg(&args, $name)?;
            Ok(Value::Tensor(colormap_tensor($variant, len)))
        }
    };
}

define_colormap_builtin!(
    PARULA_DESCRIPTOR,
    PARULA_SIGNATURES,
    parula_builtin,
    "parula",
    ColorMap::Parula,
    "Return the parula colormap as an RGB array.",
    "parula,colormap,plotting,rgb",
    "c = parula()",
    "c = parula(m)",
    crate::builtins::plotting::colormap_arrays::PARULA_INTEGER_CAPABILITIES
);
define_colormap_builtin!(
    COLORCUBE_DESCRIPTOR,
    COLORCUBE_SIGNATURES,
    colorcube_builtin,
    "colorcube",
    ColorMap::ColorCube,
    "Return the colorcube colormap as an RGB array.",
    "colorcube,colormap,plotting,rgb",
    "c = colorcube()",
    "c = colorcube(m)",
    crate::builtins::plotting::colormap_arrays::COLORCUBE_INTEGER_CAPABILITIES
);
define_colormap_builtin!(
    VIRIDIS_DESCRIPTOR,
    VIRIDIS_SIGNATURES,
    viridis_builtin,
    "viridis",
    ColorMap::Viridis,
    "Return the viridis colormap as an RGB array.",
    "viridis,colormap,plotting,rgb",
    "c = viridis()",
    "c = viridis(m)"
);
define_colormap_builtin!(
    PLASMA_DESCRIPTOR,
    PLASMA_SIGNATURES,
    plasma_builtin,
    "plasma",
    ColorMap::Plasma,
    "Return the plasma colormap as an RGB array.",
    "plasma,colormap,plotting,rgb",
    "c = plasma()",
    "c = plasma(m)"
);
define_colormap_builtin!(
    INFERNO_DESCRIPTOR,
    INFERNO_SIGNATURES,
    inferno_builtin,
    "inferno",
    ColorMap::Inferno,
    "Return the inferno colormap as an RGB array.",
    "inferno,colormap,plotting,rgb",
    "c = inferno()",
    "c = inferno(m)"
);
define_colormap_builtin!(
    MAGMA_DESCRIPTOR,
    MAGMA_SIGNATURES,
    magma_builtin,
    "magma",
    ColorMap::Magma,
    "Return the magma colormap as an RGB array.",
    "magma,colormap,plotting,rgb",
    "c = magma()",
    "c = magma(m)"
);
define_colormap_builtin!(
    TURBO_DESCRIPTOR,
    TURBO_SIGNATURES,
    turbo_builtin,
    "turbo",
    ColorMap::Turbo,
    "Return the turbo colormap as an RGB array.",
    "turbo,colormap,plotting,rgb",
    "c = turbo()",
    "c = turbo(m)"
);
define_colormap_builtin!(
    JET_DESCRIPTOR,
    JET_SIGNATURES,
    jet_builtin,
    "jet",
    ColorMap::Jet,
    "Return the jet colormap as an RGB array.",
    "jet,colormap,plotting,rgb",
    "c = jet()",
    "c = jet(m)"
);
define_colormap_builtin!(
    HOT_DESCRIPTOR,
    HOT_SIGNATURES,
    hot_builtin,
    "hot",
    ColorMap::Hot,
    "Return the hot colormap as an RGB array.",
    "hot,colormap,plotting,rgb",
    "c = hot()",
    "c = hot(m)"
);
define_colormap_builtin!(
    COOL_DESCRIPTOR,
    COOL_SIGNATURES,
    cool_builtin,
    "cool",
    ColorMap::Cool,
    "Return the cool colormap as an RGB array.",
    "cool,colormap,plotting,rgb",
    "c = cool()",
    "c = cool(m)"
);
define_colormap_builtin!(
    SPRING_DESCRIPTOR,
    SPRING_SIGNATURES,
    spring_builtin,
    "spring",
    ColorMap::Spring,
    "Return the spring colormap as an RGB array.",
    "spring,colormap,plotting,rgb",
    "c = spring()",
    "c = spring(m)"
);
define_colormap_builtin!(
    SUMMER_DESCRIPTOR,
    SUMMER_SIGNATURES,
    summer_builtin,
    "summer",
    ColorMap::Summer,
    "Return the summer colormap as an RGB array.",
    "summer,colormap,plotting,rgb",
    "c = summer()",
    "c = summer(m)"
);
define_colormap_builtin!(
    AUTUMN_DESCRIPTOR,
    AUTUMN_SIGNATURES,
    autumn_builtin,
    "autumn",
    ColorMap::Autumn,
    "Return the autumn colormap as an RGB array.",
    "autumn,colormap,plotting,rgb",
    "c = autumn()",
    "c = autumn(m)"
);
define_colormap_builtin!(
    WINTER_DESCRIPTOR,
    WINTER_SIGNATURES,
    winter_builtin,
    "winter",
    ColorMap::Winter,
    "Return the winter colormap as an RGB array.",
    "winter,colormap,plotting,rgb",
    "c = winter()",
    "c = winter(m)"
);
define_colormap_builtin!(
    GRAY_DESCRIPTOR,
    GRAY_SIGNATURES,
    gray_builtin,
    "gray",
    ColorMap::Gray,
    "Return the gray colormap as an RGB array.",
    "gray,colormap,plotting,rgb",
    "c = gray()",
    "c = gray(m)"
);
define_colormap_builtin!(
    BONE_DESCRIPTOR,
    BONE_SIGNATURES,
    bone_builtin,
    "bone",
    ColorMap::Bone,
    "Return the bone colormap as an RGB array.",
    "bone,colormap,plotting,rgb",
    "c = bone()",
    "c = bone(m)"
);
define_colormap_builtin!(
    COPPER_DESCRIPTOR,
    COPPER_SIGNATURES,
    copper_builtin,
    "copper",
    ColorMap::Copper,
    "Return the copper colormap as an RGB array.",
    "copper,colormap,plotting,rgb",
    "c = copper()",
    "c = copper(m)"
);
define_colormap_builtin!(
    PINK_DESCRIPTOR,
    PINK_SIGNATURES,
    pink_builtin,
    "pink",
    ColorMap::Pink,
    "Return the pink colormap as an RGB array.",
    "pink,colormap,plotting,rgb",
    "c = pink()",
    "c = pink(m)"
);
define_colormap_builtin!(
    LINES_DESCRIPTOR,
    LINES_SIGNATURES,
    lines_builtin,
    "lines",
    ColorMap::Lines,
    "Return the lines colormap as an RGB array.",
    "lines,colormap,plotting,rgb",
    "c = lines()",
    "c = lines(m)"
);

pub(crate) fn colormap_tensor(map: ColorMap, len: usize) -> Tensor {
    let mut colors = Vec::with_capacity(len);
    for idx in 0..len {
        colors.push(sample_colormap(&map, idx, len));
    }
    rgb_tensor_from_colors(&colors)
}

pub(crate) fn parse_rgb_colormap_tensor(
    tensor: &Tensor,
    builtin: &'static str,
) -> BuiltinResult<(ColorMap, usize)> {
    if tensor.cols != 3 || tensor_utils::tensor_element_len(tensor) != tensor.rows.saturating_mul(3)
    {
        return Err(invalid(
            builtin,
            "RGB colormap arrays must be m-by-3 numeric matrices",
        ));
    }
    if tensor.rows == 0 {
        return Err(invalid(
            builtin,
            "RGB colormap arrays must contain at least one color",
        ));
    }
    if tensor.rows > MAX_COLORMAP_LENGTH {
        return Err(invalid(builtin, "RGB colormap array is too large"));
    }

    let mut colors = Vec::with_capacity(tensor.rows);
    let uint8_map = tensor.numeric_dtype() == runmat_value::NumericDType::U8;
    for row in 0..tensor.rows {
        let mut color = [
            tensor_utils::tensor_value_f64(tensor, row),
            tensor_utils::tensor_value_f64(tensor, tensor.rows + row),
            tensor_utils::tensor_value_f64(tensor, 2 * tensor.rows + row),
        ];
        if uint8_map {
            for component in &mut color {
                *component /= f64::from(u8::MAX);
            }
        }
        if !color.iter().all(|value| value.is_finite()) {
            return Err(invalid(builtin, "RGB colormap entries must be finite"));
        }
        if !color.iter().all(|value| (0.0..=1.0).contains(value)) {
            return Err(invalid(builtin, "RGB colormap entries must be in [0, 1]"));
        }
        colors.push(color);
    }

    let listed = colors
        .iter()
        .map(|color| [color[0] as f32, color[1] as f32, color[2] as f32])
        .collect::<Vec<_>>();
    let map = ColorMap::from_rgb_rows(listed)
        .expect("validated RGB colormap array should produce a listed colormap");
    Ok((map, tensor.rows))
}

fn parse_length_arg(args: &[Value], builtin: &'static str) -> BuiltinResult<usize> {
    match args {
        [] => Ok(current_colormap_length()),
        [value] => colormap_length(value, builtin),
        _ => Err(invalid(builtin, "expected zero or one input argument")),
    }
}

fn colormap_length(value: &Value, builtin: &'static str) -> BuiltinResult<usize> {
    if let Some(length) = exact_integer_scalar(value) {
        return colormap_integer_length(&length, builtin);
    }
    let raw = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            tensor_utils::tensor_value_f64(tensor, 0)
        }
        _ => return Err(invalid(builtin, "colormap length must be a numeric scalar")),
    };
    if !raw.is_finite() || raw.fract() != 0.0 || raw < 0.0 {
        return Err(invalid(
            builtin,
            "colormap length must be a nonnegative integer",
        ));
    }
    if raw > MAX_COLORMAP_LENGTH as f64 {
        return Err(invalid(builtin, "colormap length is too large"));
    }
    Ok(raw as usize)
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn colormap_integer_length(value: &IntValue, builtin: &'static str) -> BuiltinResult<usize> {
    let Some(length) = value.try_to_usize() else {
        return Err(invalid(
            builtin,
            "colormap length must be a nonnegative integer",
        ));
    };
    if length > MAX_COLORMAP_LENGTH {
        return Err(invalid(builtin, "colormap length is too large"));
    }
    Ok(length)
}

fn rgb_tensor_from_colors(colors: &[[f64; 3]]) -> Tensor {
    let rows = colors.len();
    let mut data = Vec::with_capacity(rows.saturating_mul(3));
    for color in colors {
        data.push(color[0]);
    }
    for color in colors {
        data.push(color[1]);
    }
    for color in colors {
        data.push(color[2]);
    }
    Tensor::new(data, vec![rows, 3]).expect("RGB colormap matrix")
}

fn sample_colormap(map: &ColorMap, idx: usize, len: usize) -> [f64; 3] {
    let t = if len <= 1 {
        0.0
    } else {
        idx as f32 / (len - 1) as f32
    };
    let rgb = map.map_value(t);
    [
        rgb.x.clamp(0.0, 1.0) as f64,
        rgb.y.clamp(0.0, 1.0) as f64,
        rgb.z.clamp(0.0, 1.0) as f64,
    ]
}

fn invalid(builtin: &'static str, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {builtin}: {}",
        ERROR_INVALID_ARGUMENT.message,
        detail.as_ref()
    ))
    .with_builtin(builtin);
    if let Some(identifier) = ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::figure::figure_builtin;
    use crate::builtins::plotting::state::set_colormap_with_length;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn descriptors_cover_core_forms() {
        for (name, descriptor) in [
            ("parula", &PARULA_DESCRIPTOR),
            ("colorcube", &COLORCUBE_DESCRIPTOR),
            ("viridis", &VIRIDIS_DESCRIPTOR),
            ("plasma", &PLASMA_DESCRIPTOR),
            ("inferno", &INFERNO_DESCRIPTOR),
            ("magma", &MAGMA_DESCRIPTOR),
            ("turbo", &TURBO_DESCRIPTOR),
            ("jet", &JET_DESCRIPTOR),
            ("hot", &HOT_DESCRIPTOR),
            ("cool", &COOL_DESCRIPTOR),
            ("spring", &SPRING_DESCRIPTOR),
            ("summer", &SUMMER_DESCRIPTOR),
            ("autumn", &AUTUMN_DESCRIPTOR),
            ("winter", &WINTER_DESCRIPTOR),
            ("gray", &GRAY_DESCRIPTOR),
            ("bone", &BONE_DESCRIPTOR),
            ("copper", &COPPER_DESCRIPTOR),
            ("pink", &PINK_DESCRIPTOR),
            ("lines", &LINES_DESCRIPTOR),
        ] {
            let labels = descriptor
                .signatures
                .iter()
                .map(|sig| sig.label)
                .collect::<Vec<_>>();
            assert!(labels.contains(&format!("c = {name}()").as_str()));
            assert!(labels.contains(&format!("c = {name}(m)").as_str()));
        }
    }

    #[test]
    fn named_colormaps_return_m_by_3_rgb_arrays() {
        for (name, builtin) in [
            (
                "parula",
                parula_builtin as fn(Vec<Value>) -> BuiltinResult<Value>,
            ),
            ("colorcube", colorcube_builtin),
            ("viridis", viridis_builtin),
            ("plasma", plasma_builtin),
            ("inferno", inferno_builtin),
            ("magma", magma_builtin),
            ("turbo", turbo_builtin),
            ("jet", jet_builtin),
            ("hot", hot_builtin),
            ("cool", cool_builtin),
            ("spring", spring_builtin),
            ("summer", summer_builtin),
            ("autumn", autumn_builtin),
            ("winter", winter_builtin),
            ("gray", gray_builtin),
            ("bone", bone_builtin),
            ("copper", copper_builtin),
            ("pink", pink_builtin),
            ("lines", lines_builtin),
        ] {
            let Value::Tensor(cmap) = builtin(vec![Value::Num(6.0)]).expect(name) else {
                panic!("expected tensor");
            };
            assert_eq!(
                (cmap.rows, cmap.cols, cmap.materialize_f64().len()),
                (6, 3, 18)
            );
            assert!(
                cmap.materialize_f64()
                    .iter()
                    .all(|v| (0.0..=1.0).contains(v)),
                "{name} produced out-of-range RGB entries"
            );
        }
    }

    #[test]
    fn default_length_tracks_current_colormap_length() {
        let _guard = setup();
        let _ = figure_builtin(vec![]).expect("figure");
        set_colormap_with_length(ColorMap::ColorCube, 17);
        let Value::Tensor(cmap) = parula_builtin(vec![]).expect("parula default") else {
            panic!("expected tensor");
        };
        assert_eq!((cmap.rows, cmap.cols), (17, 3));
    }

    #[test]
    fn length_argument_validates_nonnegative_integer() {
        let err = parula_builtin(vec![Value::Num(1.5)]).expect_err("fractional length");
        assert!(err.message().contains("nonnegative integer"));
        let err = colorcube_builtin(vec![Value::Num(-1.0)]).expect_err("negative length");
        assert!(err.message().contains("nonnegative integer"));
    }

    #[test]
    fn length_argument_reads_typed_integer_tensor_exactly() {
        let length = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![MAX_COLORMAP_LENGTH as u64]),
            vec![1, 1],
        )
        .expect("typed length");
        assert_eq!(
            colormap_length(&Value::Tensor(length), "parula").unwrap(),
            MAX_COLORMAP_LENGTH
        );

        let too_large = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![MAX_COLORMAP_LENGTH as u64 + 1]),
            vec![1, 1],
        )
        .expect("too large");
        assert!(colormap_length(&Value::Tensor(too_large), "parula").is_err());

        let negative = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![-1]),
            vec![1, 1],
        )
        .expect("negative");
        assert!(colormap_length(&Value::Tensor(negative), "parula").is_err());
    }

    #[test]
    fn colorcube_accepts_all_integer_length_classes_and_returns_double() {
        let storages = [
            runmat_value::IntegerStorage::I8(vec![6]),
            runmat_value::IntegerStorage::I16(vec![6]),
            runmat_value::IntegerStorage::I32(vec![6]),
            runmat_value::IntegerStorage::I64(vec![6]),
            runmat_value::IntegerStorage::U8(vec![6]),
            runmat_value::IntegerStorage::U16(vec![6]),
            runmat_value::IntegerStorage::U32(vec![6]),
            runmat_value::IntegerStorage::U64(vec![6]),
        ];

        for storage in storages {
            let length = Tensor::new_integer(storage, vec![1, 1]).expect("length");
            let Value::Tensor(cmap) =
                colorcube_builtin(vec![Value::Tensor(length)]).expect("colorcube")
            else {
                panic!("expected colormap tensor");
            };
            assert_eq!((cmap.rows, cmap.cols), (6, 3));
            assert_eq!(cmap.numeric_dtype(), runmat_value::NumericDType::F64);
        }
    }

    #[test]
    fn parula_accepts_all_integer_length_classes_and_returns_double() {
        let storages = [
            runmat_value::IntegerStorage::I8(vec![6]),
            runmat_value::IntegerStorage::I16(vec![6]),
            runmat_value::IntegerStorage::I32(vec![6]),
            runmat_value::IntegerStorage::I64(vec![6]),
            runmat_value::IntegerStorage::U8(vec![6]),
            runmat_value::IntegerStorage::U16(vec![6]),
            runmat_value::IntegerStorage::U32(vec![6]),
            runmat_value::IntegerStorage::U64(vec![6]),
        ];

        for storage in storages {
            let length = Tensor::new_integer(storage, vec![1, 1]).expect("length");
            let Value::Tensor(cmap) = parula_builtin(vec![Value::Tensor(length)]).expect("parula")
            else {
                panic!("expected colormap tensor");
            };
            assert_eq!((cmap.rows, cmap.cols), (6, 3));
            assert_eq!(cmap.numeric_dtype(), runmat_value::NumericDType::F64);
        }
    }

    #[test]
    fn colorcube_rejects_resident_length_without_provider_dispatch() {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        });
        let err = colorcube_builtin(vec![resident]).expect_err("resident length");
        assert!(err.message().contains("numeric scalar"));
    }

    #[test]
    fn colorcube_integer_capability_is_documented_structural_host_only() {
        let capability = &COLORCUBE_INTEGER_CAPABILITIES[0];
        assert_eq!(capability.inputs[0].classes.len(), 8);
        assert_eq!(
            capability.inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            capability.computation_domain,
            BuiltinIntegerComputationDomain::Structural
        );
        assert_eq!(capability.backend, BuiltinIntegerBackendRule::HostOnly);
        assert_eq!(
            capability.output_class,
            BuiltinIntegerOutputClassRule::Double
        );
    }

    #[test]
    fn parse_rgb_colormap_tensor_recognizes_generated_maps_and_custom_gradient() {
        let parula = colormap_tensor(ColorMap::Parula, 8);
        let (map, len) = parse_rgb_colormap_tensor(&parula, "colormap").expect("parse parula");
        assert!(matches!(map, ColorMap::Listed(_)));
        assert_eq!(len, 8);

        let custom =
            Tensor::new(vec![0.0, 1.0, 0.0, 0.5, 1.0, 0.0], vec![2, 3]).expect("custom colormap");
        let (map, len) = parse_rgb_colormap_tensor(&custom, "colormap").expect("parse custom");
        assert_eq!(len, 2);
        assert!(matches!(map, ColorMap::Listed(_)));

        let typed = Tensor::new_integer(
            runmat_value::IntegerStorage::U8(vec![0, 255, 0, 128, 255, 0]),
            vec![2, 3],
        )
        .expect("typed RGB colormap");
        let (map, len) = parse_rgb_colormap_tensor(&typed, "colormap").expect("parse typed");
        assert_eq!(len, 2);
        let ColorMap::Listed(rows) = map else {
            panic!("expected listed map");
        };
        assert_eq!(rows.as_ref()[0], [0.0, 0.0, 1.0]);
        assert_eq!(rows.as_ref()[1][0], 1.0);
        assert!((rows.as_ref()[1][1] - 128.0_f32 / 255.0_f32).abs() < f32::EPSILON);
        assert_eq!(rows.as_ref()[1][2], 0.0);
    }
}
