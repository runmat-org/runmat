use super::*;
use crate::*;

const DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Data",
};
const CONTROL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Unchanged control",
};
const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "value, option",
        inputs: &[DATA, CONTROL],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const FORM: BuiltinPreparationForm = BuiltinPreparationForm {
    id: BuiltinPreparationFormId(0),
    binding_variant: "default",
    signatures: &[0],
    data_arguments: &[0],
    operation: BuiltinPreparationOperation::Unary,
    provider_requirement: BuiltinPreparationProviderRequirement::InputSupported,
};

fn errors(forms: &'static [BuiltinPreparationForm]) -> Vec<&'static str> {
    let mut entry = crate::SIN_CATALOG_ENTRY;
    entry.descriptor = &DESCRIPTOR;
    entry.placement.preparation = BuiltinOperandPreparation::Forms(forms);
    validate_builtin_preparation(&entry)
}

#[test]
fn roles_reference_existing_signatures_without_claiming_control_operands() {
    assert!(errors(&[FORM]).is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        data_arguments: &[1],
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        data_arguments: &[2],
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        signatures: &[1],
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        binding_variant: "missing",
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        operation: BuiltinPreparationOperation::RuntimeOwned,
        provider_requirement: BuiltinPreparationProviderRequirement::F64,
        ..FORM
    }])
    .is_empty());
}

#[test]
fn duplicate_and_inconsistent_forms_are_rejected() {
    assert!(!errors(&[BuiltinPreparationForm {
        operation: BuiltinPreparationOperation::ElementwiseNary { data_arity: 1 },
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        operation: BuiltinPreparationOperation::ElementwiseNary { data_arity: 3 },
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[]).is_empty());
    assert!(!errors(&[FORM, FORM]).is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        signatures: &[0, 0],
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        data_arguments: &[0, 0],
        operation: BuiltinPreparationOperation::ElementwiseBinary,
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        operation: BuiltinPreparationOperation::MatrixMultiply,
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        operation: BuiltinPreparationOperation::Sink(BuiltinSinkPreparation::Gather),
        provider_requirement: BuiltinPreparationProviderRequirement::F64,
        ..FORM
    }])
    .is_empty());
}

#[test]
fn runtime_owned_does_not_change_accelerator_capability_and_forms_serialize() {
    let mut entry = crate::SIN_CATALOG_ENTRY;
    entry.placement.preparation = BuiltinOperandPreparation::RuntimeOwned;
    assert!(validate_builtin_preparation(&entry).is_empty());
    assert_eq!(
        entry.placement.accelerator,
        BuiltinAcceleratorPolicy::Optional
    );
    let encoded = serde_json::to_string(&FORM).unwrap();
    assert!(encoded.contains("data_arguments"));
    entry.descriptor = &DESCRIPTOR;
    entry.placement.preparation = BuiltinOperandPreparation::Forms(&[FORM]);
    entry.placement.accelerator = BuiltinAcceleratorPolicy::Forbidden;
    assert!(!validate_builtin_preparation(&entry).is_empty());
}

#[test]
fn floating_cast_targets_are_typed_and_legacy_form_identity_is_preserved() {
    assert!(errors(&[BuiltinPreparationForm {
        operation: BuiltinPreparationOperation::FloatingCast(runmat_types::NumericClass::Single),
        ..FORM
    }])
    .is_empty());
    assert!(errors(&[BuiltinPreparationForm {
        operation: BuiltinPreparationOperation::FloatingCast(runmat_types::NumericClass::Double),
        ..FORM
    }])
    .is_empty());
    assert!(!errors(&[BuiltinPreparationForm {
        operation: BuiltinPreparationOperation::FloatingCast(runmat_types::NumericClass::Int8),
        ..FORM
    }])
    .is_empty());
    let BuiltinOperandPreparation::Forms(forms) = SINGLE_CATALOG_ENTRY.placement.preparation else {
        panic!("declared forms");
    };
    assert_eq!(forms.len(), 3);
    assert_eq!(forms[0].id, BuiltinPreparationFormId(0));
    assert_eq!(forms[0].signatures, &[0, 1]);
    assert_eq!(forms[0].operation, BuiltinPreparationOperation::Unary);
    assert_eq!(forms[1].id, BuiltinPreparationFormId(1));
    assert_eq!(forms[1].signatures, &[0]);
    assert_eq!(
        forms[1].operation,
        BuiltinPreparationOperation::FloatingCast(runmat_types::NumericClass::Single)
    );
    assert_eq!(forms[2].id, BuiltinPreparationFormId(2));
    assert_eq!(forms[2].signatures, &[1]);
    assert_eq!(forms[2].data_arguments, &[0]);
    assert_eq!(
        forms[2].operation,
        BuiltinPreparationOperation::RuntimeOwned
    );
    assert_eq!(
        forms[2].provider_requirement,
        BuiltinPreparationProviderRequirement::InputSupported
    );
    assert!(validate_builtin_preparation(&SINGLE_CATALOG_ENTRY).is_empty());
    let BuiltinOperandPreparation::Forms(forms) = DOUBLE_CATALOG_ENTRY.placement.preparation else {
        panic!("declared double forms");
    };
    assert_eq!(forms.len(), 3);
    assert_eq!(forms[0].id, BuiltinPreparationFormId(0));
    assert_eq!(forms[0].signatures, &[0, 1]);
    assert_eq!(forms[0].operation, BuiltinPreparationOperation::Unary);
    assert_eq!(forms[1].id, BuiltinPreparationFormId(1));
    assert_eq!(forms[1].signatures, &[0]);
    assert_eq!(forms[1].data_arguments, &[0]);
    assert_eq!(
        forms[1].operation,
        BuiltinPreparationOperation::FloatingCast(runmat_types::NumericClass::Double)
    );
    assert_eq!(
        forms[1].provider_requirement,
        BuiltinPreparationProviderRequirement::F64
    );
    assert_eq!(forms[2].id, BuiltinPreparationFormId(2));
    assert_eq!(forms[2].signatures, &[1]);
    assert_eq!(forms[2].data_arguments, &[0]);
    assert_eq!(
        forms[2].operation,
        BuiltinPreparationOperation::RuntimeOwned
    );
    assert_eq!(
        forms[2].provider_requirement,
        BuiltinPreparationProviderRequirement::InputSupported
    );
    assert!(validate_builtin_preparation(&DOUBLE_CATALOG_ENTRY).is_empty());
}
