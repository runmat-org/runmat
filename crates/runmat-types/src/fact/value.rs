use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueFact {
    pub kind: ValueKindFact,
    pub shape: ShapeFact,
    pub storage: StorageFact,
    pub layout: LayoutFact,
    pub contiguity: ContiguityFact,
    pub view: ViewFact,
    pub residency: ResidencyFact,
    pub alias: AliasFact,
    pub mutation: MutationFact,
    pub certainty: CertaintyFact,
    pub invalidation: InvalidationVector,
}

impl ValueFact {
    pub fn proven(kind: ValueKindFact, shape: ShapeFact, storage: StorageFact) -> Self {
        Self {
            kind,
            shape,
            storage,
            layout: LayoutFact::ColumnMajor,
            contiguity: ContiguityFact::Contiguous,
            view: ViewFact::Materialized,
            residency: ResidencyFact::Host,
            alias: AliasFact::Unique,
            mutation: MutationFact::ValueSemantics,
            certainty: CertaintyFact::Proven,
            invalidation: InvalidationVector::default(),
        }
    }

    pub fn scalar(kind: ValueKindFact) -> Self {
        Self::proven(kind, ShapeFact::Scalar, StorageFact::Scalar)
    }

    pub fn unknown(reason: DynamicReason) -> Self {
        Self {
            kind: ValueKindFact::Unknown,
            shape: ShapeFact::Unknown,
            storage: StorageFact::Unknown,
            layout: LayoutFact::Unknown,
            contiguity: ContiguityFact::Unknown,
            view: ViewFact::Unknown,
            residency: ResidencyFact::Unknown,
            alias: AliasFact::Unknown,
            mutation: MutationFact::Unknown,
            certainty: CertaintyFact::Dynamic(reason),
            invalidation: InvalidationVector::default(),
        }
    }

    pub fn never() -> Self {
        let mut fact = Self::unknown(DynamicReason::Unspecified);
        fact.kind = ValueKindFact::Never;
        fact
    }

    pub fn numeric(&self) -> Option<NumericFact> {
        match self.kind {
            ValueKindFact::Numeric(value) => Some(value),
            _ => None,
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.element_count() == Some(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueKindFact {
    Never,
    Unknown,
    Void,
    Numeric(NumericFact),
    Logical,
    Character,
    String,
    Symbolic,
    Cell(CellFact),
    Struct(StructFact),
    Object(ObjectFact),
    ClassReference(ClassReferenceFact),
    Callable(CallableFact),
    OutputList(OutputListFact),
    Exception(ExceptionFact),
    Execution(ExecutionFact),
    Foreign(ForeignFact),
}
