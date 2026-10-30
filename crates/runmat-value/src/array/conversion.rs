use super::*;

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Int(IntValue::I32(i))
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(IntValue::I64(i))
    }
}
impl From<u32> for Value {
    fn from(i: u32) -> Self {
        Value::Int(IntValue::U32(i))
    }
}
impl From<u64> for Value {
    fn from(i: u64) -> Self {
        Value::Int(IntValue::U64(i))
    }
}
impl From<i16> for Value {
    fn from(i: i16) -> Self {
        Value::Int(IntValue::I16(i))
    }
}
impl From<i8> for Value {
    fn from(i: i8) -> Self {
        Value::Int(IntValue::I8(i))
    }
}
impl From<u16> for Value {
    fn from(i: u16) -> Self {
        Value::Int(IntValue::U16(i))
    }
}
impl From<u8> for Value {
    fn from(i: u8) -> Self {
        Value::Int(IntValue::U8(i))
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Num(f)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::string(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::string(s)
    }
}

impl From<Utf8Text> for Value {
    fn from(text: Utf8Text) -> Self {
        Value::String(StringElement::Text(text))
    }
}

impl From<StringElement> for Value {
    fn from(element: StringElement) -> Self {
        Value::String(element)
    }
}

impl From<Tensor> for Value {
    fn from(m: Tensor) -> Self {
        Value::Tensor(m)
    }
}

// Remove blanket From<Vec<Value>> to avoid losing shape information

// TryFrom implementations for extracting native types
impl TryFrom<&Value> for i32 {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Int(i) => Ok(i.to_i64() as i32),
            Value::Num(n) => Ok(*n as i32),
            _ => Err(format!("cannot convert {v:?} to i32")),
        }
    }
}

impl TryFrom<&Value> for f64 {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Num(n) => Ok(*n),
            Value::Int(i) => Ok(i.to_f64()),
            _ => Err(format!("cannot convert {v:?} to f64")),
        }
    }
}

impl TryFrom<&Value> for bool {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Bool(b) => Ok(*b),
            Value::Int(i) => Ok(!i.is_zero()),
            Value::Num(n) => Ok(*n != 0.0),
            _ => Err(format!("cannot convert {v:?} to bool")),
        }
    }
}

impl TryFrom<&Value> for String {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::String(s) => s
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "cannot convert missing string to text".to_owned()),
            Value::StringArray(sa) => {
                if sa.data.len() == 1 {
                    sa.data[0]
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "cannot convert missing string to text".to_owned())
                } else {
                    Err("cannot convert string array to scalar string".to_string())
                }
            }
            Value::CharArray(ca) => {
                // Convert full char array to one string if it is a single row; else error
                if ca.shape.len() <= 2 && ca.rows == 1 {
                    ca.try_row_string()
                        .map_err(|error| error.to_string())?
                        .ok_or_else(|| {
                            "cannot convert multi-row char array to scalar string".into()
                        })
                } else {
                    Err("cannot convert multi-row char array to scalar string".to_string())
                }
            }
            Value::Int(i) => Ok(i.decimal_string()),
            Value::Num(n) => Ok(n.to_string()),
            Value::Bool(b) => Ok(b.to_string()),
            _ => Err(format!("cannot convert {v:?} to String")),
        }
    }
}

impl TryFrom<&Value> for Tensor {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tensor(m) => Ok(m.clone()),
            _ => Err(format!("cannot convert {v:?} to Tensor")),
        }
    }
}

impl TryFrom<&Value> for Value {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        Ok(v.clone())
    }
}

impl TryFrom<&Value> for Vec<Value> {
    type Error = String;
    fn try_from(v: &Value) -> Result<Self, Self::Error> {
        match v {
            Value::Cell(c) => Ok(c.data().to_vec()),
            _ => Err(format!("cannot convert {v:?} to Vec<Value>")),
        }
    }
}
