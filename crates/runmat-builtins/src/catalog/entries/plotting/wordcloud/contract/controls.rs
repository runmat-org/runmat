use crate::*;

pub const WORDCLOUD_INTEGER_PROPERTIES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "wordcloud-integer-properties",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "wordcloud with typed-integer chart property values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WordcloudIntegerPropertiesExtension"),
    };

pub const WORDCLOUD_NUMERIC_PARENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "wordcloud-numeric-parent",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "wordcloud with a numeric parent handle is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WordcloudNumericParentExtension"),
    };

pub const WORDCLOUD_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    WORDCLOUD_INTEGER_PROPERTIES_EXTENSION,
    WORDCLOUD_NUMERIC_PARENT_EXTENSION,
];

pub const WORDCLOUD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "wc = wordcloud(words, integer_sizeData) or wordcloud(table_with_integer_sizes, ___)",
        inputs: &WORDCLOUD_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "All eight documented integer classes contribute nonnegative chart weights. Conversion occurs only at the explicit floating chart-size and renderer boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "wc = wordcloud(tbl, integer_wordVar, integer_sizeVar)",
        inputs: &WORDCLOUD_SELECTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer selectors are exact one-based indices and never enter chart arithmetic.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "wc = wordcloud(___, property, integer_value)",
        inputs: &WORDCLOUD_PROPERTY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed properties are independently gated. Structural counts use exact storage; colors, positions, and SizePower cross their explicit floating chart boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "wc = wordcloud(integer_parent, ___)",
        inputs: &WORDCLOUD_PARENT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A positive typed scalar is an exact alias into RunMat's figure registry only in RunMat mode; documented graphics-object parents remain unaffected.",
    },
];

const WORDCLOUD_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sizeData or selected table size variable",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public datatype table explicitly lists every integer class. Native storage remains authoritative until values enter the chart's floating size/layout domain.",
    }];

const WORDCLOUD_SELECTOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "wordVar or sizeVar",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public datatype table explicitly lists every integer class for numeric table-variable indices. RunMat validates the exact one-based selector against the bounded variable count.",
    }];

const WORDCLOUD_PROPERTY_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numeric chart property",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public property descriptions specify numeric or integer values without publishing native integer storage classes. RunMat gates typed property storage before chart selection or mutation.",
    }];

const WORDCLOUD_PARENT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "parent",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public parent form uses graphics container objects. RunMat numeric registry aliases are independently gated and decoded exactly before plot-state selection.",
    }];
