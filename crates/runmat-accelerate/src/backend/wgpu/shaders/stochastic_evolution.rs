use runmat_accelerate_api::stochastic_evolution::StochasticEvolutionArithmeticClass;
use runmat_types::{NumericClass, RealFloatingRecipe};

fn wgsl_type(class: NumericClass) -> Option<&'static str> {
    match class {
        NumericClass::Single => Some("f32"),
        NumericClass::Double => Some("f64"),
        _ => None,
    }
}

fn convert(expression: &str, from: NumericClass, to: NumericClass) -> String {
    if from == to {
        expression.to_owned()
    } else {
        format!(
            "{}({expression})",
            wgsl_type(to).expect("validated floating class")
        )
    }
}

fn binary(
    recipe: RealFloatingRecipe,
    left: (&str, NumericClass),
    right: (&str, NumericClass),
) -> String {
    let computation = recipe.computation_class();
    let left = convert(left.0, left.1, computation);
    let right = convert(right.0, right.1, computation);
    let expression = match recipe.operator() {
        runmat_types::OperatorKind::Add => format!("({left} + {right})"),
        runmat_types::OperatorKind::ElementwiseMultiply => format!("({left} * {right})"),
        _ => unreachable!("validated stochastic arithmetic contains only add and multiply"),
    };
    convert(&expression, computation, recipe.result_class())
}

pub fn source(arithmetic: StochasticEvolutionArithmeticClass) -> Option<String> {
    let state_type = wgsl_type(arithmetic.state_class())?;
    let noise_type = wgsl_type(arithmetic.noise_class())?;
    let result_type = wgsl_type(arithmetic.result_class())?;
    let drift_type = wgsl_type(arithmetic.drift_class())?;
    let scale_type = wgsl_type(arithmetic.scale_class())?;
    let scaled = binary(
        arithmetic.scale_noise(),
        ("scale", arithmetic.scale_class()),
        ("noise", arithmetic.noise_class()),
    );
    let scaled_class = arithmetic.scale_noise().result_class();
    let scaled_type = wgsl_type(scaled_class)?;
    let delta = binary(
        arithmetic.drift_add(),
        ("drift", arithmetic.drift_class()),
        ("scaled", scaled_class),
    );
    let exponential_class = arithmetic.drift_add().result_class();
    let exponential_type = wgsl_type(exponential_class)?;
    let output = binary(
        arithmetic.state_product(),
        ("state", arithmetic.state_class()),
        ("factor", exponential_class),
    );
    let control_fields = match (arithmetic.drift_class(), arithmetic.scale_class()) {
        (NumericClass::Single, NumericClass::Single) => {
            "drift: f32,\n    scale: f32,\n    _control_pad: vec2<u32>,"
        }
        (NumericClass::Single, NumericClass::Double) => {
            "drift: f32,\n    _control_pad: u32,\n    scale: f64,"
        }
        (NumericClass::Double, NumericClass::Single) => {
            "drift: f64,\n    scale: f32,\n    _control_pad: u32,"
        }
        (NumericClass::Double, NumericClass::Double) => "drift: f64,\n    scale: f64,",
        _ => return None,
    };
    Some(format!(
        r#"
struct State {{ data: array<{state_type}>, }};
struct Noise {{ data: array<{noise_type}>, }};
struct Output {{ data: array<{result_type}>, }};
struct Params {{
    offset: u32,
    count: u32,
    total: u32,
    _pad: u32,
    {control_fields}
}};
@group(0) @binding(0) var<storage, read> Input: State;
@group(0) @binding(1) var<storage, read> SuppliedNoise: Noise;
@group(0) @binding(2) var<storage, read_write> Result: Output;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let local = gid.x;
    if local >= params.count {{ return; }}
    let global = params.offset + local;
    if global >= params.total {{ return; }}
    let state: {state_type} = Input.data[global];
    let noise: {noise_type} = SuppliedNoise.data[local];
    let drift: {drift_type} = params.drift;
    let scale: {scale_type} = params.scale;
    let scaled: {scaled_type} = {scaled};
    let delta: {exponential_type} = {delta};
    let factor: {exponential_type} = exp(delta);
    Result.data[global] = {output};
}}
"#
    ))
}

#[cfg(test)]
mod tests {
    use super::source;
    use runmat_accelerate_api::scalar_program::ScalarLiteral;
    use runmat_accelerate_api::stochastic_evolution::StochasticEvolutionArithmetic;
    use runmat_types::NumericClass;

    #[test]
    fn mixed_precision_source_uses_canonical_stage_casts_and_supplied_noise() {
        let arithmetic = StochasticEvolutionArithmetic::new(
            NumericClass::Single,
            NumericClass::Double,
            ScalarLiteral::F32Bits(0.25_f32.to_bits()),
            ScalarLiteral::F64Bits(0.5_f64.to_bits()),
        )
        .unwrap();
        let shader = source(arithmetic.class_identity()).unwrap();
        assert!(shader.contains("array<f32>"));
        assert!(shader.contains("array<f64>"));
        assert!(shader.contains("SuppliedNoise.data[local]"));
        assert!(!shader.contains("philox"));
    }
}
