#[derive(Debug, Clone, PartialEq)]
pub enum JavaValue {
    Null,
    Boolean(bool),
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    UnsignedLong(u64),
    Float(f32),
    Double(f64),
    Char(u16),
    String(String),
    Callback(u64),
    Array {
        component: crate::JavaParameterType,
        elements: Vec<JavaValue>,
    },
    Object {
        handle: JavaObjectHandle,
        class_name: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct JavaCallbackInvocation {
    pub method_name: String,
    pub arguments: Vec<JavaValue>,
    pub returns_value: bool,
}
use crate::JavaObjectHandle;
