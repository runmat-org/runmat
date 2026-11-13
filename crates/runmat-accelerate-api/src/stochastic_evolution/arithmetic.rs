use crate::scalar_program::ScalarLiteral;
use runmat_types::{NumericClass, OperatorKind, RealFloatingRecipe};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StochasticEvolutionArithmetic {
    class: StochasticEvolutionArithmeticClass,
    drift: ScalarLiteral,
    scale: ScalarLiteral,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StochasticEvolutionArithmeticClass {
    state_class: NumericClass,
    noise_class: NumericClass,
    drift_class: NumericClass,
    scale_class: NumericClass,
    scale_noise: RealFloatingRecipe,
    drift_add: RealFloatingRecipe,
    state_product: RealFloatingRecipe,
}

impl StochasticEvolutionArithmetic {
    pub fn new(
        state_class: NumericClass,
        noise_class: NumericClass,
        drift: ScalarLiteral,
        scale: ScalarLiteral,
    ) -> Option<Self> {
        let scale_noise = RealFloatingRecipe::for_classes(
            OperatorKind::ElementwiseMultiply,
            scale.class(),
            noise_class,
        )?;
        let drift_add = RealFloatingRecipe::for_classes(
            OperatorKind::Add,
            drift.class(),
            scale_noise.result_class(),
        )?;
        let state_product = RealFloatingRecipe::for_classes(
            OperatorKind::ElementwiseMultiply,
            state_class,
            drift_add.result_class(),
        )?;
        [state_class, noise_class, drift.class(), scale.class()]
            .into_iter()
            .all(|class| matches!(class, NumericClass::Single | NumericClass::Double))
            .then_some(Self {
                class: StochasticEvolutionArithmeticClass {
                    state_class,
                    noise_class,
                    drift_class: drift.class(),
                    scale_class: scale.class(),
                    scale_noise,
                    drift_add,
                    state_product,
                },
                drift,
                scale,
            })
    }
    pub const fn class_identity(self) -> StochasticEvolutionArithmeticClass {
        self.class
    }
    pub const fn state_class(self) -> NumericClass {
        self.class.state_class
    }
    pub const fn noise_class(self) -> NumericClass {
        self.class.noise_class
    }
    pub const fn drift(self) -> ScalarLiteral {
        self.drift
    }
    pub const fn scale(self) -> ScalarLiteral {
        self.scale
    }
    pub const fn scale_noise(self) -> RealFloatingRecipe {
        self.class.scale_noise
    }
    pub const fn drift_add(self) -> RealFloatingRecipe {
        self.class.drift_add
    }
    pub const fn exponential_class(self) -> NumericClass {
        self.class.drift_add.result_class()
    }
    pub const fn state_product(self) -> RealFloatingRecipe {
        self.class.state_product
    }
    pub const fn result_class(self) -> NumericClass {
        self.class.state_product.result_class()
    }
    pub fn requires_f64(self) -> bool {
        self.class.requires_f64()
    }
}

impl StochasticEvolutionArithmeticClass {
    pub const fn state_class(self) -> NumericClass {
        self.state_class
    }
    pub const fn noise_class(self) -> NumericClass {
        self.noise_class
    }
    pub const fn drift_class(self) -> NumericClass {
        self.drift_class
    }
    pub const fn scale_class(self) -> NumericClass {
        self.scale_class
    }
    pub const fn scale_noise(self) -> RealFloatingRecipe {
        self.scale_noise
    }
    pub const fn drift_add(self) -> RealFloatingRecipe {
        self.drift_add
    }
    pub const fn state_product(self) -> RealFloatingRecipe {
        self.state_product
    }
    pub const fn result_class(self) -> NumericClass {
        self.state_product.result_class()
    }
    pub fn requires_f64(self) -> bool {
        [
            self.state_class,
            self.noise_class,
            self.drift_class,
            self.scale_class,
            self.scale_noise.computation_class(),
            self.drift_add.computation_class(),
            self.state_product.computation_class(),
            self.result_class(),
        ]
        .into_iter()
        .any(|class| class == NumericClass::Double)
    }
}
