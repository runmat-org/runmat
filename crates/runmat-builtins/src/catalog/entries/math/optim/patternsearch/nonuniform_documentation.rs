use crate::*;

pub(super) const SECTION: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading: "Nonuniform pattern search",
    paragraphs: &[
        "Algorithm='nups' alternates GPS and random orthogonal polling in 16-iteration cycles. Each phase receives between two and fourteen iterations according to its measured improvement per objective point. 'nups-gps' uses only GPS; 'nups-mads' uses only the random orthogonal phase. Interior polls use coordinate rays and the positive all-ones ray, followed by their negatives. An orthogonal frame is retained across that pair, rather than redrawn for its negative half.",
        "Linear boundary geometry adds feasible generating rays. Trial steps are shortened against the original bounds and linear rows. Linear equalities use tangent GPS generators: 'nups' uses its GPS phase only, and 'nups-mads' rejects linear equalities. Nonlinear constraints use the same augmented-Lagrangian subproblems as classic search.",
        "A poll retains small improvements while continuing until sufficient relative decrease or exhaustion of its directions. Selected parallel execution completes the poll. PollMethod, PollOrderAlgorithm, UseCompletePoll, MeshRotate, AccelerateMesh, MeshExpansionFactor, and MeshContractionFactor are dormant NUPS controls. A poll method cannot be SearchFcn; Latin-hypercube search remains available.",
        "RunMat uses relative decrease 1e-4, an initial 8/8 phase allocation, and the effective initial mesh as its coarse-mesh threshold. Future allocations round the measured rate proportion and clamp each phase to two through fourteen iterations; two zero rates restore 8/8. Unsuccessful polls halve the mesh. Sufficient decrease doubles it; a third consecutive small-improvement poll multiplies it by 1.5; other improvements double a coarse mesh or leave a finer mesh unchanged. These numerical choices do not specify a MATLAB-identical random trajectory.",
        "A successful auxiliary search doubles the mesh without attributing its gain to a polling phase or changing that phase's polarity. Mesh and machine-precision convergence require a completed spanning pair, or a complete equality-tangent cone, at the current accepted point. An accepted move, changed boundary controls, changed scaling, or nonlinear merit update invalidates earlier coverage. Partial time/evaluation exhaustion cannot certify convergence.",
        "Random draws and rngstate use the supplied execution context's stream, or the process fallback stream when no context is supplied. A foreign ambient context does not replace that stream. The same rule applies to classic random ordering, LTMADS, and Latin-hypercube search. No random turn remains held across objective, constraint, or observer callbacks.",
    ],
};

pub(super) const EXAMPLE: BuiltinExample = BuiltinExample {
    id: "nonuniform-bounded",
    title: "Use nonuniform GPS with an active upper bound",
    program: "opts = optimoptions('patternsearch','Algorithm','nups-gps','Display','off','ScaleMesh',false,'MeshTolerance',1e-3);\n[x,fval,flag,out] = patternsearch(@(x) (x-2)^2,0,[],[],[],[],0,1,[],opts)",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(x>=0 && x<=1+1e-6); assert(abs(x-1)<1e-4); assert(abs(fval-(x-2)^2)<1e-12); assert(flag>0); assert(strcmp(out.pollmethod,'nups-gps'));",
    },
};
