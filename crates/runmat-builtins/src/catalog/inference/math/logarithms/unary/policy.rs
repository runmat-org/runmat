use crate::LogarithmKind;

#[derive(Clone, Copy)]
pub(super) struct UnaryLogarithmPolicy {
    kind: LogarithmKind,
}

impl UnaryLogarithmPolicy {
    pub(super) const fn for_kind(kind: LogarithmKind) -> Self {
        Self { kind }
    }

    pub(super) const fn name(self) -> &'static str {
        match self.kind {
            LogarithmKind::Natural => "log",
            LogarithmKind::OnePlus => "log1p",
            LogarithmKind::Binary => "log2",
            LogarithmKind::Common => "log10",
        }
    }

    pub(super) const fn real_boundary(self) -> f64 {
        match self.kind {
            LogarithmKind::OnePlus => -1.0,
            LogarithmKind::Natural | LogarithmKind::Binary | LogarithmKind::Common => 0.0,
        }
    }

    pub(super) const fn accepts_symbolic(self) -> bool {
        matches!(self.kind, LogarithmKind::Natural)
    }

    pub(super) const fn accepts_tabular(self) -> bool {
        !matches!(self.kind, LogarithmKind::OnePlus)
    }

    pub(super) const fn arity_diagnostic(self) -> &'static str {
        self.diagnostic(DiagnosticKind::Arity)
    }

    pub(super) const fn sparse_diagnostic(self) -> &'static str {
        self.diagnostic(DiagnosticKind::Sparse)
    }

    pub(super) const fn complex_integer_diagnostic(self) -> &'static str {
        self.diagnostic(DiagnosticKind::ComplexInteger)
    }

    pub(super) const fn input_diagnostic(self) -> &'static str {
        self.diagnostic(DiagnosticKind::Input)
    }

    const fn diagnostic(self, diagnostic: DiagnosticKind) -> &'static str {
        match (self.kind, diagnostic) {
            (LogarithmKind::OnePlus, DiagnosticKind::Arity) => "RM-CATALOG-LOG1P-ARITY",
            (LogarithmKind::OnePlus, DiagnosticKind::Sparse) => "RM-CATALOG-LOG1P-SPARSE",
            (LogarithmKind::OnePlus, DiagnosticKind::ComplexInteger) => {
                "RM-CATALOG-LOG1P-COMPLEX-INTEGER"
            }
            (LogarithmKind::OnePlus, DiagnosticKind::Input) => "RM-CATALOG-LOG1P-INPUT",
            (_, DiagnosticKind::Arity) => "RM-CATALOG-LOGARITHM-ARITY",
            (_, DiagnosticKind::Sparse) => "RM-CATALOG-LOGARITHM-SPARSE",
            (_, DiagnosticKind::ComplexInteger) => "RM-CATALOG-LOGARITHM-COMPLEX-INTEGER",
            (_, DiagnosticKind::Input) => "RM-CATALOG-LOGARITHM-INPUT",
        }
    }
}

#[derive(Clone, Copy)]
enum DiagnosticKind {
    Arity,
    Sparse,
    ComplexInteger,
    Input,
}
