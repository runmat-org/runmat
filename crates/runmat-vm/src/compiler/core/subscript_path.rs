use super::{
    Compiler, IndexEndContext, IDENT_MIR_CELL_EXPAND_PLAN_INVALID,
    IDENT_MIR_CELL_INDEX_PLAN_INVALID, IDENT_MIR_INDEX_CONTEXT_INVALID,
    IDENT_MIR_PAREN_CELL_PLAN_INVALID, IDENT_MIR_SUBSCRIPT_CHAIN_INVALID,
};
use crate::bytecode::{
    BytecodeDottedInvokeArgument, BytecodeSubscriptSelector, BytecodeSubscriptStep, Instr,
};
use crate::compiler::CompileError;
use runmat_mir::{
    MirDottedInvokeArgument, MirIndexComponent, MirIndexing, MirSubscriptChain, MirSubscriptStep,
};

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
        chain.validate().map_err(|error| {
            let identifier = match error {
                runmat_mir::MirSubscriptChainError::InvalidReadContext => {
                    IDENT_MIR_INDEX_CONTEXT_INVALID
                }
                runmat_mir::MirSubscriptChainError::ParenthesesWithCellPlan => {
                    IDENT_MIR_PAREN_CELL_PLAN_INVALID
                }
                runmat_mir::MirSubscriptChainError::BracesWithoutCellPlan => {
                    IDENT_MIR_CELL_INDEX_PLAN_INVALID
                }
                runmat_mir::MirSubscriptChainError::InvalidCellExpandAll => {
                    IDENT_MIR_CELL_EXPAND_PLAN_INVALID
                }
                runmat_mir::MirSubscriptChainError::Empty
                | runmat_mir::MirSubscriptChainError::InvalidArgumentSpans
                | runmat_mir::MirSubscriptChainError::InvalidDottedArgumentRegion
                | runmat_mir::MirSubscriptChainError::InvalidPreparedRegion => {
                    IDENT_MIR_SUBSCRIPT_CHAIN_INVALID
                }
            };
            self.compile_error(format!("invalid MIR subscript chain: {error}"))
                .with_identifier(identifier)
        })?;
        self.compile_mir_operand(&chain.root)?;
        let mut steps = Vec::with_capacity(chain.steps.len());
        for (position, step) in chain.steps.iter().enumerate() {
            match step {
                MirSubscriptStep::Member(member) => {
                    steps.push(BytecodeSubscriptStep::Member(member.clone()));
                }
                MirSubscriptStep::DynamicMember(member) => {
                    self.compile_mir_operand(member)?;
                    steps.push(BytecodeSubscriptStep::DynamicMember);
                }
                MirSubscriptStep::PreparedDynamicMember(region) => {
                    self.compile_mir_prepared_expression_region(region)?;
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
                MirSubscriptStep::DottedInvoke {
                    member,
                    arguments,
                    semantic_fallback,
                } => {
                    let contextual = arguments.iter().any(|argument| match argument {
                        MirDottedInvokeArgument::Contextual(_) => true,
                        MirDottedInvokeArgument::Call(region) => region.contains_unscoped_end(),
                        MirDottedInvokeArgument::Colon => false,
                    });
                    let variable_width = arguments.iter().any(|argument| {
                        matches!(argument, MirDottedInvokeArgument::Call(region)
                            if !matches!(region.argument, runmat_mir::MirCallArg::Single(_)))
                    });
                    if contextual && variable_width {
                        return Err(self
                            .compile_error("receiver-relative end cannot share an index list with variable-width expansion")
                            .with_identifier(IDENT_MIR_INDEX_CONTEXT_INVALID));
                    }
                    if contextual {
                        // `end` belongs to the stored member's indexing receiver,
                        // not the containing object. Materialize that member once
                        // before evaluating its selectors, as for an ordinary
                        // member/index chain; retain it through nested awaits.
                        steps.push(BytecodeSubscriptStep::Member(member.clone()));
                    }
                    // Preparing the prefix is an execution boundary: the next
                    // invocation's arguments must not run before this receiver
                    // has been evaluated. The existing prepared receiver also
                    // roots contextual `end` through nested awaits.
                    let prefix_operand_count =
                        steps.iter().map(BytecodeSubscriptStep::operand_count).sum();
                    self.emit(Instr::BeginSubscriptEndReceiver {
                        prefix: std::mem::take(&mut steps),
                    });
                    let mut compiled = Vec::with_capacity(arguments.len());
                    for (component, argument) in arguments.iter().enumerate() {
                        match argument {
                            MirDottedInvokeArgument::Colon => {
                                compiled.push(BytecodeDottedInvokeArgument::Colon)
                            }
                            MirDottedInvokeArgument::Contextual(region) => {
                                let previous =
                                    self.index_end_context.replace(IndexEndContext::Subscript {
                                        component,
                                        component_count: arguments.len(),
                                    });
                                let result = self.compile_mir_expression_region(region);
                                self.index_end_context = previous;
                                result?;
                                compiled.push(BytecodeDottedInvokeArgument::Call(
                                    runmat_runtime::call::arguments::ArgumentSpec::Single,
                                ));
                            }
                            MirDottedInvokeArgument::Call(region) => {
                                // Call-shaped arguments can also be selectors
                                // on a data member. Bind their `end` to this
                                // receiver without scalarizing captured lists.
                                let previous =
                                    self.index_end_context.replace(IndexEndContext::Subscript {
                                        component,
                                        component_count: arguments.len(),
                                    });
                                let result = (|| {
                                    self.compile_mir_expression_steps(&region.steps)?;
                                    self.compile_mir_call_arg(&region.argument)
                                })();
                                self.index_end_context = previous;
                                result?;
                                let (mut specs, _) = self
                                    .mir_call_arg_specs(std::slice::from_ref(&region.argument))?;
                                compiled.push(BytecodeDottedInvokeArgument::Call(specs.remove(0)));
                            }
                        }
                    }
                    let argument_operand_count = compiled
                        .iter()
                        .map(|argument| match argument {
                            BytecodeDottedInvokeArgument::Colon => 0,
                            BytecodeDottedInvokeArgument::Call(spec) => spec.stack_operand_count(),
                        })
                        .sum();
                    if contextual {
                        self.emit(Instr::FinishSubscriptEndReceiver {
                            prefix_operand_count,
                            selector_count: argument_operand_count,
                        });
                    } else {
                        self.emit(Instr::FinishDottedInvokeReceiver {
                            prefix_operand_count,
                            argument_operand_count,
                        });
                    }
                    let path = if contextual {
                        vec![BytecodeSubscriptStep::Parentheses {
                            selectors: compiled
                                .into_iter()
                                .map(|argument| match argument {
                                    BytecodeDottedInvokeArgument::Colon => {
                                        BytecodeSubscriptSelector::Colon
                                    }
                                    BytecodeDottedInvokeArgument::Call(
                                        runmat_runtime::call::arguments::ArgumentSpec::Single,
                                    ) => BytecodeSubscriptSelector::Value,
                                    BytecodeDottedInvokeArgument::Call(spec) => {
                                        BytecodeSubscriptSelector::Expansion(spec)
                                    }
                                })
                                .collect(),
                        }]
                    } else {
                        vec![BytecodeSubscriptStep::DottedInvoke {
                            member: member.clone(),
                            arguments: compiled,
                            semantic_fallback: semantic_fallback.clone(),
                        }]
                    };
                    if position + 1 == chain.steps.len() {
                        self.emit_terminal_subscript_path(
                            chain,
                            path,
                            capture_slot,
                            to_sequence_register,
                        );
                    } else {
                        self.emit(Instr::ReadSubscriptPath {
                            steps: path,
                            selection: runmat_types::SequenceUse::RequireSingle,
                            context: chain.context,
                            to_sequence_register: false,
                        });
                    }
                }
            }
        }
        if !steps.is_empty() {
            self.emit_terminal_subscript_path(chain, steps, capture_slot, to_sequence_register);
        }
        Ok(())
    }

    fn emit_terminal_subscript_path(
        &mut self,
        chain: &MirSubscriptChain,
        steps: Vec<BytecodeSubscriptStep>,
        capture_slot: Option<usize>,
        to_sequence_register: bool,
    ) {
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
        if contextual
            && indexing
                .components
                .iter()
                .any(|component| matches!(component, MirIndexComponent::Expansion(_)))
        {
            return Err(self
                .compile_error("receiver-relative end cannot share an index list with variable-width expansion")
                .with_identifier(IDENT_MIR_INDEX_CONTEXT_INVALID));
        }
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
                MirIndexComponent::PreparedExpr(region) => {
                    self.compile_mir_prepared_expression_region(region)?;
                    selectors.push(BytecodeSubscriptSelector::Value);
                    operand_count += 1;
                }
                MirIndexComponent::Expansion(region) => {
                    self.compile_mir_expression_steps(&region.steps)?;
                    self.compile_mir_call_arg(&region.argument)?;
                    let (mut specs, _) =
                        self.mir_call_arg_specs(std::slice::from_ref(&region.argument))?;
                    let spec = specs.remove(0);
                    operand_count = operand_count
                        .checked_add(spec.stack_operand_count())
                        .ok_or_else(|| {
                            self.compile_error("expanded index operand count overflows")
                        })?;
                    selectors.push(BytecodeSubscriptSelector::Expansion(spec));
                }
                MirIndexComponent::ContextualExpr(region) => {
                    let previous = self.index_end_context.replace(IndexEndContext::Subscript {
                        component,
                        component_count: indexing.components.len(),
                    });
                    let result = self.compile_mir_expression_region(region);
                    self.index_end_context = previous;
                    result?;
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
