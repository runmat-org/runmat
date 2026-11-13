use crate::*;

pub const JAVA_ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JAVA.UNAVAILABLE",
    identifier: Some("RunMat:Foreign:UnsupportedOnWasm"),
    when: "A JVM operation is called in a browser without a native Java host.",
    message: "Java interoperability is unavailable on this host.",
};
pub const JAVA_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JAVA.INVALID_CALL",
    identifier: Some("RunMat:Foreign:InvalidCall"),
    when: "A Java builtin receives an invalid public argument form.",
    message: "Invalid Java interoperability operation.",
};
pub const JAVA_ERROR_HOST_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JAVA.HOST_UNAVAILABLE",
    identifier: Some("RunMat:Foreign:HostUnavailable"),
    when: "A native Java call has no active runtime context.",
    message: "Java operation has no active runtime context.",
};
const ERROR_ADAPTER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JAVA.ADAPTER_UNAVAILABLE",
    identifier: Some("RunMat:Foreign:AdapterUnavailable"),
    when: "The runtime has no session Java adapter.",
    message: "Java adapter is unavailable.",
};
const ERROR_EDT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JAVA.EDT_UNAVAILABLE",
    identifier: Some("RunMat:Java:EdtUnavailable"),
    when: "An EDT operation runs without Desktop Java UI support.",
    message: "Java EDT execution requires a Desktop host with Java UI support.",
};
const ERROR_JAVA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JAVA.FOREIGN",
    identifier: None,
    when: "JVM discovery, class loading, reflection, conversion, callback, or invocation fails.",
    message: "The Java error retains its source class, cause chain, and stack frames.",
};
const FOREIGN_ERRORS: &[BuiltinErrorDescriptor] = &[
    JAVA_ERROR_UNAVAILABLE,
    JAVA_ERROR_INVALID,
    JAVA_ERROR_HOST_UNAVAILABLE,
    ERROR_ADAPTER,
    ERROR_JAVA,
];
const EDT_ERRORS: &[BuiltinErrorDescriptor] = &[
    JAVA_ERROR_UNAVAILABLE,
    JAVA_ERROR_INVALID,
    JAVA_ERROR_HOST_UNAVAILABLE,
    ERROR_ADAPTER,
    ERROR_EDT,
    ERROR_JAVA,
];
const LOCAL_ERRORS: &[BuiltinErrorDescriptor] = &[];

const CLASS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "className",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fully qualified Java class name.",
};
const METHOD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "methodName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Java method name.",
};
const RECEIVER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "objectOrClass",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Session-owned Java object or fully qualified class name.",
};
const ARGUMENTS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Arguments converted at the Java foreign boundary.",
};
const DIMENSIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dimensions",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Checked nonnegative Java array dimensions.",
};
const PATH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "path",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Session dynamic-classpath text entry or cell of text entries.",
};
const FEATURE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "feature",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "jvm, desktop, awt, or swing.",
};
const VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value inspected for Java foreign identity.",
};
const OPTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Java environment or classpath option.",
};
const OPTION_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "optionValue",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Option value.",
};
const OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result converted from the session JVM.",
}];
const NONE: &[BuiltinParamDescriptor] = &[];

const OBJECT_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "object = javaObject(className, arguments...)",
    inputs: &[CLASS, ARGUMENTS],
    outputs: OUTPUT,
}];
const OBJECT_EDT_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "object = javaObjectEDT(className, arguments...)",
    inputs: &[CLASS, ARGUMENTS],
    outputs: OUTPUT,
}];
const METHOD_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "result = javaMethod(methodName, objectOrClass, arguments...)",
    inputs: &[METHOD, RECEIVER, ARGUMENTS],
    outputs: OUTPUT,
}];
const METHOD_EDT_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "result = javaMethodEDT(methodName, objectOrClass, arguments...)",
    inputs: &[METHOD, RECEIVER, ARGUMENTS],
    outputs: OUTPUT,
}];
const ARRAY_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "array = javaArray(className, dimensions...)",
    inputs: &[CLASS, DIMENSIONS],
    outputs: OUTPUT,
}];
const ADD_PATH_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "javaaddpath(path, ...)",
        inputs: &[PATH, ARGUMENTS],
        outputs: NONE,
    },
    BuiltinSignatureDescriptor {
        label: "javaaddpath(path, ..., '-end')",
        inputs: &[PATH, ARGUMENTS],
        outputs: NONE,
    },
];
const REMOVE_PATH_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "javarmpath(path, ...)",
    inputs: &[PATH, ARGUMENTS],
    outputs: NONE,
}];
const CLASSPATH_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "paths = javaclasspath()",
        inputs: NONE,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "paths = javaclasspath('-dynamic' | '-static' | '-all')",
        inputs: &[OPTION],
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "javaclasspath(path, ...)",
        inputs: &[PATH, ARGUMENTS],
        outputs: NONE,
    },
];
const JENV_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "environment = jenv()",
        inputs: NONE,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "environment = jenv(option, optionValue)",
        inputs: &[OPTION, OPTION_VALUE],
        outputs: OUTPUT,
    },
];
const USEJAVA_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "available = usejava(feature)",
    inputs: &[FEATURE],
    outputs: OUTPUT,
}];
const ISJAVA_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "result = isjava(value)",
    inputs: &[VALUE],
    outputs: OUTPUT,
}];

const fn descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
    errors: &'static [BuiltinErrorDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors,
    }
}

pub const JAVA_OBJECT_DESCRIPTOR: BuiltinDescriptor = descriptor(OBJECT_SIGNATURES, FOREIGN_ERRORS);
pub const JAVA_OBJECT_EDT_DESCRIPTOR: BuiltinDescriptor =
    descriptor(OBJECT_EDT_SIGNATURES, EDT_ERRORS);
pub const JAVA_METHOD_DESCRIPTOR: BuiltinDescriptor = descriptor(METHOD_SIGNATURES, FOREIGN_ERRORS);
pub const JAVA_METHOD_EDT_DESCRIPTOR: BuiltinDescriptor =
    descriptor(METHOD_EDT_SIGNATURES, EDT_ERRORS);
pub const JAVA_ARRAY_DESCRIPTOR: BuiltinDescriptor = descriptor(ARRAY_SIGNATURES, FOREIGN_ERRORS);
pub const JAVA_ADD_PATH_DESCRIPTOR: BuiltinDescriptor =
    descriptor(ADD_PATH_SIGNATURES, FOREIGN_ERRORS);
pub const JAVA_REMOVE_PATH_DESCRIPTOR: BuiltinDescriptor =
    descriptor(REMOVE_PATH_SIGNATURES, FOREIGN_ERRORS);
pub const JAVA_CLASSPATH_DESCRIPTOR: BuiltinDescriptor =
    descriptor(CLASSPATH_SIGNATURES, FOREIGN_ERRORS);
pub const JENV_DESCRIPTOR: BuiltinDescriptor = descriptor(JENV_SIGNATURES, FOREIGN_ERRORS);
pub const USEJAVA_DESCRIPTOR: BuiltinDescriptor = descriptor(USEJAVA_SIGNATURES, FOREIGN_ERRORS);
pub const ISJAVA_DESCRIPTOR: BuiltinDescriptor = descriptor(ISJAVA_SIGNATURES, LOCAL_ERRORS);
