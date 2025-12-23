//! SSA Integration Tests
//!
//! Tests the full SSA pipeline: bytecode → SSA → optimization

use runmat_ignition::Instr;
use runmat_turbine::dominators::DomTree;
use runmat_turbine::ssa_builder::bytecode_to_ssa;
use runmat_turbine::ssa_opt::{optimize, OptLevel};

/// Helper to create a simple bytecode sequence
fn simple_arithmetic_bytecode() -> Vec<Instr> {
    // x = 1 + 2; y = x * 3
    vec![
        Instr::LoadConst(1.0), // 0
        Instr::LoadConst(2.0), // 1
        Instr::Add,            // 2: 1 + 2 = 3
        Instr::StoreVar(0),    // 3: x = 3
        Instr::LoadVar(0),     // 4: load x
        Instr::LoadConst(3.0), // 5
        Instr::Mul,            // 6: x * 3 = 9
        Instr::StoreVar(1),    // 7: y = 9
        Instr::Return,         // 8
    ]
}

/// Bytecode with common subexpression
fn cse_candidate_bytecode() -> Vec<Instr> {
    // a = x + y; b = x + y; c = a + b
    vec![
        Instr::LoadVar(0),  // 0: load x
        Instr::LoadVar(1),  // 1: load y
        Instr::Add,         // 2: x + y
        Instr::StoreVar(2), // 3: a = x + y
        Instr::LoadVar(0),  // 4: load x (again)
        Instr::LoadVar(1),  // 5: load y (again)
        Instr::Add,         // 6: x + y (CSE candidate)
        Instr::StoreVar(3), // 7: b = x + y
        Instr::LoadVar(2),  // 8: load a
        Instr::LoadVar(3),  // 9: load b
        Instr::Add,         // 10: a + b
        Instr::StoreVar(4), // 11: c = a + b
        Instr::Return,      // 12
    ]
}

/// Bytecode with constant folding opportunity
fn const_fold_bytecode() -> Vec<Instr> {
    // x = 2 + 3; y = x * 4; z = y - 1
    vec![
        Instr::LoadConst(2.0), // 0
        Instr::LoadConst(3.0), // 1
        Instr::Add,            // 2: 2 + 3 = 5
        Instr::StoreVar(0),    // 3: x = 5
        Instr::LoadVar(0),     // 4
        Instr::LoadConst(4.0), // 5
        Instr::Mul,            // 6: 5 * 4 = 20
        Instr::StoreVar(1),    // 7: y = 20
        Instr::LoadVar(1),     // 8
        Instr::LoadConst(1.0), // 9
        Instr::Sub,            // 10: 20 - 1 = 19
        Instr::StoreVar(2),    // 11: z = 19
        Instr::Return,         // 12
    ]
}

/// Bytecode with a simple loop
fn loop_bytecode() -> Vec<Instr> {
    // for i = 1:10; x = x + i; end
    vec![
        Instr::LoadConst(0.0), // 0: x = 0
        Instr::StoreVar(0),    // 1
        Instr::LoadConst(1.0), // 2: i = 1
        Instr::StoreVar(1),    // 3
        // loop header:
        Instr::LoadVar(1),      // 4: load i
        Instr::LoadConst(10.0), // 5
        Instr::LessEqual,       // 6: i <= 10
        Instr::JumpIfFalse(17), // 7: exit if false -> jump to Return at 17
        // loop body:
        Instr::LoadVar(0),     // 8: load x
        Instr::LoadVar(1),     // 9: load i
        Instr::Add,            // 10: x + i
        Instr::StoreVar(0),    // 11: x = x + i
        Instr::LoadVar(1),     // 12: load i
        Instr::LoadConst(1.0), // 13
        Instr::Add,            // 14: i + 1
        Instr::StoreVar(1),    // 15: i = i + 1
        Instr::Jump(4),        // 16: back to header at index 4
        Instr::Return,         // 17: exit
    ]
}

#[test]
fn test_bytecode_to_ssa_simple() {
    let bytecode = simple_arithmetic_bytecode();
    let func = bytecode_to_ssa(&bytecode, 2, "simple");

    // Should have at least one block
    assert!(!func.blocks.is_empty());

    // Verify we have instructions
    let total_instrs: usize = func.blocks.iter().map(|b| b.instrs.len()).sum();
    assert!(total_instrs > 0);

    // Check dump works
    let dump = func.dump();
    assert!(dump.contains("simple"));
    assert!(dump.contains("ConstF64"));
}

#[test]
fn test_ssa_optimization_none() {
    let bytecode = simple_arithmetic_bytecode();
    let mut func = bytecode_to_ssa(&bytecode, 2, "opt_none");

    let before_instrs: usize = func.blocks.iter().map(|b| b.instrs.len()).sum();

    optimize(&mut func, OptLevel::None);

    let after_instrs: usize = func.blocks.iter().map(|b| b.instrs.len()).sum();

    // OptLevel::None should not change anything
    assert_eq!(before_instrs, after_instrs);
}

#[test]
fn test_ssa_optimization_size() {
    let bytecode = const_fold_bytecode();
    let mut func = bytecode_to_ssa(&bytecode, 3, "opt_size");

    optimize(&mut func, OptLevel::Size);

    // Should still work (DCE might remove some dead code)
    assert!(!func.blocks.is_empty());
}

#[test]
fn test_ssa_optimization_speed() {
    let bytecode = cse_candidate_bytecode();
    let mut func = bytecode_to_ssa(&bytecode, 5, "opt_speed");

    let before_dump = func.dump();

    optimize(&mut func, OptLevel::Speed);

    let after_dump = func.dump();

    // After CSE, there should be some Copy operations
    // (or fewer Add operations)
    println!("Before:\n{before_dump}");
    println!("After:\n{after_dump}");

    // Just verify it runs without panicking
    assert!(!func.blocks.is_empty());
}

#[test]
fn test_ssa_optimization_aggressive() {
    let bytecode = loop_bytecode();
    let mut func = bytecode_to_ssa(&bytecode, 2, "opt_aggressive");

    optimize(&mut func, OptLevel::Aggressive);

    // Should still have valid structure
    assert!(!func.blocks.is_empty());

    // Should have multiple blocks (loop creates branches)
    assert!(func.blocks.len() >= 2);
}

#[test]
fn test_dominator_tree() {
    let bytecode = loop_bytecode();
    let func = bytecode_to_ssa(&bytecode, 2, "dom_test");

    let dom = DomTree::compute(&func);

    // Entry block should dominate all other blocks
    let entry = func.entry;
    for block in &func.blocks {
        assert!(dom.dominates(entry, block.id));
    }

    // Preorder should contain all blocks
    assert_eq!(dom.preorder().len(), func.blocks.len());
}

#[test]
fn test_loop_detection() {
    use runmat_turbine::loop_analysis::LoopInfo;

    let bytecode = loop_bytecode();
    let func = bytecode_to_ssa(&bytecode, 2, "loop_test");
    let dom = DomTree::compute(&func);

    let loops = LoopInfo::compute(&func, &dom);

    // Should detect at least one loop
    let loop_count = loops.loops().count();
    assert!(
        loop_count >= 1,
        "Expected at least 1 loop, found {loop_count}"
    );
}

#[test]
fn test_ssa_dump_format() {
    let bytecode = simple_arithmetic_bytecode();
    let func = bytecode_to_ssa(&bytecode, 2, "dump_test");

    let dump = func.dump();

    // Should contain function name
    assert!(dump.contains("dump_test"));

    // Should contain block labels
    assert!(dump.contains("bb0"));

    // Should contain value names
    assert!(dump.contains("%"));

    // Should contain return
    assert!(dump.contains("ret"));
}

#[test]
fn test_all_opt_levels_dont_panic() {
    let bytecodes = vec![
        ("simple", simple_arithmetic_bytecode()),
        ("cse", cse_candidate_bytecode()),
        ("const_fold", const_fold_bytecode()),
        ("loop", loop_bytecode()),
    ];

    let levels = vec![
        OptLevel::None,
        OptLevel::Size,
        OptLevel::Speed,
        OptLevel::Aggressive,
    ];

    for (name, bytecode) in &bytecodes {
        for level in &levels {
            let mut func = bytecode_to_ssa(bytecode, 10, format!("{}_{:?}", name, level));
            optimize(&mut func, *level);

            // Just verify it doesn't panic and produces valid output
            assert!(
                !func.blocks.is_empty(),
                "{} at {:?} produced empty blocks",
                name,
                level
            );
            assert!(
                !func.dump().is_empty(),
                "{} at {:?} produced empty dump",
                name,
                level
            );
        }
    }
}

/// Demonstrate SSA optimization impact on instruction count
#[test]
fn test_ssa_optimization_impact() {
    // CSE test: same expression computed twice
    let cse_bytecode = cse_candidate_bytecode();

    let mut none_func = bytecode_to_ssa(&cse_bytecode, 5, "cse_none");
    let none_before: usize = none_func.blocks.iter().map(|b| b.instrs.len()).sum();
    optimize(&mut none_func, OptLevel::None);
    let none_after: usize = none_func.blocks.iter().map(|b| b.instrs.len()).sum();

    let mut speed_func = bytecode_to_ssa(&cse_bytecode, 5, "cse_speed");
    let speed_before: usize = speed_func.blocks.iter().map(|b| b.instrs.len()).sum();
    optimize(&mut speed_func, OptLevel::Speed);
    let speed_after: usize = speed_func.blocks.iter().map(|b| b.instrs.len()).sum();

    println!("CSE Bytecode Optimization Impact:");
    println!(
        "  OptLevel::None:  {} -> {} instructions",
        none_before, none_after
    );
    println!(
        "  OptLevel::Speed: {} -> {} instructions",
        speed_before, speed_after
    );

    // Speed should have same or fewer instructions due to CSE + DCE
    assert!(
        speed_after <= none_after,
        "OptLevel::Speed should not increase instruction count"
    );

    // Constant folding test
    let fold_bytecode = const_fold_bytecode();

    let mut fold_none = bytecode_to_ssa(&fold_bytecode, 3, "fold_none");
    optimize(&mut fold_none, OptLevel::None);
    let fold_none_instrs: usize = fold_none.blocks.iter().map(|b| b.instrs.len()).sum();

    let mut fold_speed = bytecode_to_ssa(&fold_bytecode, 3, "fold_speed");
    optimize(&mut fold_speed, OptLevel::Speed);
    let fold_speed_instrs: usize = fold_speed.blocks.iter().map(|b| b.instrs.len()).sum();

    println!("\nConstant Folding Impact:");
    println!("  OptLevel::None:  {} instructions", fold_none_instrs);
    println!("  OptLevel::Speed: {} instructions", fold_speed_instrs);

    // Constant folding should reduce Add/Mul/Sub to ConstF64
    assert!(
        fold_speed_instrs <= fold_none_instrs,
        "Constant folding should not increase instructions"
    );
}
