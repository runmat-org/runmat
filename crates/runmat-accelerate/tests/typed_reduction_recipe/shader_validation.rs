//! Capability-minimal validation of the checked reduction WGSL stages.
use super::{array, axes, config, recipe, ReductionRecipe};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequiredCaps {
    Basic,
    Float64,
    Float64IeeeWords,
}

impl RequiredCaps {
    fn for_recipe(recipe: &ReductionRecipe) -> Self {
        let descriptor = recipe.physical().descriptor();
        let has_f64 = descriptor.output_precision == ProviderPrecision::F64
            || descriptor.partial.precision == ProviderPrecision::F64
            || descriptor
                .inputs
                .iter()
                .any(|input| input.precision == ProviderPrecision::F64);
        if !has_f64 {
            Self::Basic
        } else if !recipe.contract().identity
            && descriptor.partial.precision == ProviderPrecision::F64
        {
            Self::Float64IeeeWords
        } else {
            Self::Float64
        }
    }

    fn enabled(self) -> Capabilities {
        match self {
            Self::Basic => Capabilities::empty(),
            Self::Float64 => Capabilities::FLOAT64,
            Self::Float64IeeeWords => Capabilities::FLOAT64 | Capabilities::SHADER_INT64,
        }
    }

    fn insufficient(self) -> [Option<Capabilities>; 2] {
        match self {
            Self::Basic => [None, None],
            Self::Float64 => [Some(Capabilities::empty()), None],
            Self::Float64IeeeWords => [
                Some(Capabilities::FLOAT64),
                Some(Capabilities::SHADER_INT64),
            ],
        }
    }
}

pub(super) fn validate_shaders(recipe: &ReductionRecipe) {
    let required = RequiredCaps::for_recipe(recipe);
    for (stage, source) in [
        ("single", recipe.single_source()),
        ("first", recipe.pass1_source()),
        ("final", recipe.pass2_source()),
    ] {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{stage}: {}", error.emit_to_string(source)));
        let flags = ValidationFlags::all();
        Validator::new(flags, required.enabled())
            .validate(&module)
            .unwrap_or_else(|error| panic!("{stage} ({required:?}): {error:?}\n{source}"));
        for missing in required.insufficient().into_iter().flatten() {
            assert!(
                Validator::new(flags, missing).validate(&module).is_err(),
                "{stage}: {required:?} shader unexpectedly validates with {missing:?}"
            );
        }
        let partial = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("RRPartial"))
            .unwrap()
            .1;
        let naga::TypeInner::Struct { members, span } = &partial.inner else {
            panic!("partial state struct");
        };
        assert_eq!(*span, recipe.partial_layout().stride);
        assert_eq!(members[1].offset, recipe.partial_layout().count_offset);
        assert_eq!(members[2].offset, recipe.partial_layout().flags_offset);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn nonidentity_precision_profiles_require_exact_optional_capabilities() {
    let single = recipe(
        "min",
        vec![array(true, &[25, 2])],
        axes("min", Value::Num(1.0), false),
        None,
        config(),
    );
    assert_eq!(RequiredCaps::for_recipe(&single), RequiredCaps::Basic);
    validate_shaders(&single);

    let double = recipe(
        "min",
        vec![array(false, &[25, 2])],
        axes("min", Value::Num(1.0), false),
        None,
        config(),
    );
    assert_eq!(
        RequiredCaps::for_recipe(&double),
        RequiredCaps::Float64IeeeWords
    );
    validate_shaders(&double);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn f64_identity_shader_needs_float64_but_not_ieee_words() {
    let identity = recipe(
        "sum",
        vec![array(false, &[2, 3])],
        vec![Value::Num(4.0)],
        None,
        config(),
    );
    assert!(identity.contract().identity);
    assert_eq!(RequiredCaps::for_recipe(&identity), RequiredCaps::Float64);
    for source in [
        identity.single_source(),
        identity.pass1_source(),
        identity.pass2_source(),
    ] {
        assert!(!source.contains("runmat_f64_low"));
        assert!(!source.contains("@IEEE_"));
    }
    validate_shaders(&identity);
}
