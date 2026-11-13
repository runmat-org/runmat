use crate::*;

pub const ROWFILTER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "rowfilter is constructed from variable names, Parquet metadata/datastores, tables, or timetables; direct typed-integer arrays have no public data or control role and reject before gather. Integer columns nested in a tabular source remain opaque authoritative payloads while the filter captures row-selection metadata.",
    };

pub const ROWFILTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "F = rowfilter()",
            inputs: &[],
            outputs: &[BuiltinParamDescriptor {
                name: "descriptor",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Scalar host value descriptor; not a live iterator or UI handle.",
            }],
        },
        BuiltinSignatureDescriptor {
            label: "F = rowfilter(source, predicate)",
            inputs: &[
                BuiltinParamDescriptor {
                    name: "source",
                    ty: BuiltinParamType::Any,
                    arity: BuiltinParamArity::Required,
                    default: None,
                    description:
                        "Selector payload stored in Variables; not evaluated by this constructor.",
                },
                BuiltinParamDescriptor {
                    name: "predicate",
                    ty: BuiltinParamType::Any,
                    arity: BuiltinParamArity::Optional,
                    default: Some("''"),
                    description: "Legacy predicate payload, stored without invocation.",
                },
            ],
            outputs: &[BuiltinParamDescriptor {
                name: "descriptor",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Scalar host value descriptor; not a live iterator or UI handle.",
            }],
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::super::super::shared::TABLE_ERROR_INVALID_ARGUMENT],
};
