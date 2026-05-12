use runmat_builtins::Value;

#[derive(Debug, Clone, Copy)]
pub struct ValueToF64Error;

pub fn value_to_f64(v: &Value) -> Result<f64, ValueToF64Error> {
    match v {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::Complex(re, im) if im.abs() < 1e-12 => Ok(*re),
        Value::ComplexTensor(ct) if ct.data.len() == 1 && ct.data[0].1.abs() < 1e-12 => {
            Ok(ct.data[0].0)
        }
        _ => Err(ValueToF64Error),
    }
}
