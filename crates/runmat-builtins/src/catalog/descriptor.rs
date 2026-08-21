use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinOutputMode {
    Fixed,
    ByRequestedOutputCount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinCompletionPolicy {
    Public,
    MethodOnly,
    HiddenInternal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinParamArity {
    Required,
    Optional,
    Variadic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinParamType {
    Any,
    NumericScalar,
    IntegerScalar,
    StringScalar,
    NumericArray,
    LogicalArray,
    SizeArg,
    LikePrototype,
    AxesHandle,
    StyleSpec,
    PropertyName,
    PropertyValue,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinParamDescriptor {
    pub name: &'static str,
    pub ty: BuiltinParamType,
    pub arity: BuiltinParamArity,
    pub default: Option<&'static str>,
    pub description: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinSignatureDescriptor {
    pub label: &'static str,
    pub inputs: &'static [BuiltinParamDescriptor],
    pub outputs: &'static [BuiltinParamDescriptor],
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinErrorDescriptor {
    pub code: &'static str,
    pub identifier: Option<&'static str>,
    pub when: &'static str,
    pub message: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuiltinDescriptor {
    pub signatures: &'static [BuiltinSignatureDescriptor],
    pub output_mode: BuiltinOutputMode,
    pub completion_policy: BuiltinCompletionPolicy,
    pub errors: &'static [BuiltinErrorDescriptor],
}
