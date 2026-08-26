#[derive(Debug, Clone, PartialEq)]
pub enum JavaValue {
    Null,
    Boolean(bool),
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Char(u16),
    String(String),
    Array {
        component: crate::JavaParameterType,
        elements: Vec<JavaValue>,
    },
    Object {
        handle: JavaObjectHandle,
        class_name: String,
    },
}
use crate::JavaObjectHandle;
