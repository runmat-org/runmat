use super::Compiler;
use crate::bytecode::{BytecodeSubscriptSelector, BytecodeSubscriptStep, Instr};
use crate::compiler::CompileError;
use runmat_mir::{MirIndexComponent, MirIndexing, MirSubscriptChain, MirSubscriptStep};

impl Compiler {
    pub(super) fn compile_subscript_chain(
        &mut self,
        chain: &MirSubscriptChain,
        capture_slot: Option<usize>,
    ) -> Result<(), CompileError> {
        self.compile_subscript_chain_inner(chain, capture_slot, false)
    }

    pub(super) fn compile_subscript_chain_to_register(
        &mut self,
        chain: &MirSubscriptChain,
    ) -> Result<(), CompileError> {
        self.compile_subscript_chain_inner(chain, None, true)
    }

    fn compile_subscript_chain_inner(
        &mut self,
        chain: &MirSubscriptChain,
        capture_slot: Option<usize>,
        to_sequence_register: bool,
    ) -> Result<(), CompileError> {
        self.compile_mir_operand(&chain.root)?;
        let mut steps = Vec::with_capacity(chain.steps.len());
        for step in &chain.steps {
            match step {
                MirSubscriptStep::Member(member) => {
                    steps.push(BytecodeSubscriptStep::Member(member.clone()));
                }
                MirSubscriptStep::DynamicMember(member) => {
                    self.compile_mir_operand(member)?;
                    steps.push(BytecodeSubscriptStep::DynamicMember);
                }
                MirSubscriptStep::Index(indexing) => {
                    let (selectors, consumed_prefix) =
                        self.compile_subscript_selectors(&steps, indexing)?;
                    if consumed_prefix {
                        steps.clear();
                    }
                    steps.push(match indexing.kind {
                        runmat_hir::IndexKind::Paren => {
                            BytecodeSubscriptStep::Parentheses { selectors }
                        }
                        runmat_hir::IndexKind::Brace => BytecodeSubscriptStep::Braces { selectors },
                    });
                }
                MirSubscriptStep::DottedInvoke { member, indexing } => {
                    let (arguments, consumed_prefix) =
                        self.compile_subscript_selectors(&steps, indexing)?;
                    if consumed_prefix {
                        steps.clear();
                    }
                    steps.push(BytecodeSubscriptStep::DottedInvoke {
                        member: member.clone(),
                        arguments,
                    });
                }
            }
        }
        if let Some(sequence_slot) = capture_slot {
            self.emit(Instr::CaptureSubscriptPath {
                steps,
                sequence_slot,
                context: chain.context,
            });
        } else {
            self.emit(Instr::ReadSubscriptPath {
                steps,
                selection: chain.sequence_use,
                context: chain.context,
                to_sequence_register,
            });
        }
        Ok(())
    }

    fn compile_subscript_selectors(
        &mut self,
        prefix: &[BytecodeSubscriptStep],
        indexing: &MirIndexing,
    ) -> Result<(Vec<BytecodeSubscriptSelector>, bool), CompileError> {
        let contextual = indexing
            .components
            .iter()
            .any(|component| matches!(component, MirIndexComponent::ContextualExpr(_)));
        if contextual {
            self.emit(Instr::BeginSubscriptEndReceiver {
                prefix: prefix.to_vec(),
            });
        }
        let mut selectors = Vec::with_capacity(indexing.components.len());
        let mut operand_count = 0usize;
        for (component, selector) in indexing.components.iter().enumerate() {
            match selector {
                MirIndexComponent::Colon => selectors.push(BytecodeSubscriptSelector::Colon),
                MirIndexComponent::Expr(operand) => {
                    self.compile_mir_operand(operand)?;
                    selectors.push(BytecodeSubscriptSelector::Value);
                    operand_count += 1;
                }
                MirIndexComponent::ContextualExpr(region) => {
                    let previous = self
                        .subscript_end_component
                        .replace((component, indexing.components.len()));
                    self.compile_mir_expression_region(region)?;
                    self.subscript_end_component = previous;
                    selectors.push(BytecodeSubscriptSelector::Value);
                    operand_count += 1;
                }
            }
        }
        if contextual {
            self.emit(Instr::FinishSubscriptEndReceiver {
                selector_count: operand_count,
                prefix_operand_count: prefix
                    .iter()
                    .map(BytecodeSubscriptStep::operand_count)
                    .sum(),
            });
        }
        Ok((selectors, contextual))
    }
}
