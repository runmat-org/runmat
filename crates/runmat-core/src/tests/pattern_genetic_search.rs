// The auxiliary genetic search uses the public parser/VM path and the
// enclosing Pattern observation owner, not a second solver fixture.

#[test]
fn public_pattern_genetic_search_preserves_seed_fresh_requests_and_later_fitness_events() {
    runmat_gc::gc_test_context(|| {
        let example = runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY
            .documentation
            .examples
            .iter()
            .find(|example| example.id == "known-seed-genetic-search")
            .unwrap();
        let runmat_builtins::BuiltinExampleVerification::Assertions { source } =
            example.verification
        else {
            panic!("genetic-search example requires independent assertions");
        };
        run(
            "pattern-genetic-seed.m",
            &format!("{}\n{source}", example.program),
        );
        run("pattern-genetic-forced-cache.m", FRESH_ELITE);
        run("pattern-genetic-later-fitness.m", LATER_FITNESS);
        run("pattern-genetic-hybrid.m", PATTERN_HYBRID);
        run(
            "pattern-genetic-inherited-vectorization.m",
            INHERITED_VECTORIZATION,
        );
        run("pattern-genetic-dynamic-window.m", DYNAMIC_WINDOW);
        run("pattern-genetic-original-error.m", ORIGINAL_ERROR);
    });
}
fn run(name: &str, program: &str) {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let outcome = execute_text_request_named_source(&mut session, name, program)
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
    assert!(!outcome.has_errors(), "{name}: {:?}", outcome.diagnostics);
}
const FRESH_ELITE: &str = r#"
global psga_calls psga_events;
psga_calls=0; psga_events={};
local=optimoptions('ga','Display','off','PopulationSize',2,'EliteCount',2, ...
 'InitialPopulationMatrix',[0;0],'MaxGenerations',1,'OutputFcn',@audit);
opts=optimoptions('patternsearch','Display','off','ScaleMesh',false,'Cache','on', ...
 'SearchFcn',{@searchga,1,local},'MaxIterations',1,'MaxFunctionEvaluations',3);
[x,f,flag,out]=patternsearch(@objective,0,[],[],[],[],[],[],[],opts);
assert(x==0 && f==0 && flag==0);
assert(psga_calls==3 && out.funccount==3);
assert(numel(psga_events)==3 && strcmp(psga_events{1},'init') && strcmp(psga_events{2},'iter') && strcmp(psga_events{3},'done'));
function f=objective(x)
global psga_calls; psga_calls=psga_calls+1; f=x(1)^2;
end
function [state,options,changed]=audit(options,state,phase)
global psga_calls psga_events;
psga_events{end+1}=phase;
assert(state.FunEval==psga_calls-1);
assert(isequal(state.Population,[0;0]) && isequal(state.Score,[0;0]));
changed=false;
end
"#;
const LATER_FITNESS: &str = r#"
global psga_calls psga_saved psga_events;
psga_calls=0; psga_saved=[]; psga_events={};
local=optimoptions('ga','Display','off','PopulationSize',2,'EliteCount',2, ...
 'CreationFcn',@create,'InitialPopulationMatrix',[0],'MaxGenerations',0,'OutputFcn',@audit);
opts=optimoptions('patternsearch','Display','off','ScaleMesh',false, ...
 'SearchFcn',{@searchga,1,local},'MaxIterations',1,'MaxFunctionEvaluations',3);
[x,f,flag,out]=patternsearch(@objective,0,[],[],[],[],[],[],[],opts);
assert(x==0 && f==0 && flag==0 && psga_calls==3 && out.funccount==3);
assert(numel(psga_events)==2 && strcmp(psga_events{1},'init') && strcmp(psga_events{2},'done'));
function f=objective(x)
global psga_calls; psga_calls=psga_calls+1; f=x(1)^2;
end
function population=create(n,problem,options)
global psga_saved; psga_saved=problem.FitnessFcn; population=zeros(2,n);
end
function [state,options,changed]=audit(options,state,phase)
global psga_saved psga_events;
psga_events{end+1}=phase;
if strcmp(phase,'init'), assert(feval(psga_saved,0)==0); end
changed=false;
end
"#;
const PATTERN_HYBRID: &str = r#"
global psga_calls psga_zero_calls; psga_calls=0; psga_zero_calls=0;
h=optimset('Display','off','MaxIter',0,'MaxFunEvals',1);
local=optimoptions('ga','Display','off','PopulationSize',2,'EliteCount',2, ...
 'InitialPopulationMatrix',[0;0],'MaxGenerations',0,'HybridFcn',{'fminsearch',h});
opts=optimoptions('patternsearch','Display','off','ScaleMesh',false, ...
 'SearchFcn',{@searchga,1,local},'MaxIterations',1,'MaxFunctionEvaluations',3);
[x,f,flag,out]=patternsearch(@objective,0,[],[],[],[],[],[],[],opts);
assert(x==0 && f==0 && flag==0 && psga_calls==3 && out.funccount==3 && psga_zero_calls==2);
function f=objective(x)
global psga_calls psga_zero_calls; psga_calls=psga_calls+1;
if x(1)==0, psga_zero_calls=psga_zero_calls+1; end
f=x(1)^2;
end
"#;
const INHERITED_VECTORIZATION: &str = r#"
global psga_points; psga_points=0;
opts=optimoptions('patternsearch','Display','off','SearchFcn','searchga', ...
 'UseVectorized',true,'UseParallel',true,'MaxIterations',1,'MaxFunctionEvaluations',2);
[x,f,flag,out]=patternsearch(@objective,0,[],[],[],[],[],[],[],opts);
assert(x==0 && f==0 && flag==0 && out.funccount==2 && psga_points==2);
function f=objective(X)
global psga_points; psga_points=psga_points+size(X,1); f=sum(X.^2,2);
end
"#;
const DYNAMIC_WINDOW: &str = r#"
global psga_events psga_local;
psga_events=0;
psga_local=optimoptions('ga','Display','off','PopulationSize',2,'EliteCount',2, ...
 'InitialPopulationMatrix',[0;0],'MaxGenerations',0,'OutputFcn',@genetic_audit);
opts=optimoptions('patternsearch','Display','off','ScaleMesh',false, ...
 'MaxIterations',2,'OutputFcn',@activate);
[x,f,flag]=patternsearch(@(x) x^2,0,[],[],[],[],[],[],[],opts);
assert(x==0 && f==0 && psga_events==0);
function [stop,options,changed]=activate(values,options,phase)
global psga_local;
stop=false; changed=false;
if strcmp(phase,'iter') && values.iteration==1
 options.SearchFcn={@searchga,1,psga_local}; changed=true;
end
end
function [state,options,changed]=genetic_audit(options,state,phase)
global psga_events; psga_events=psga_events+1; changed=false;
end
"#;
const ORIGINAL_ERROR: &str = r#"
global psga_calls; psga_calls=0;
local=optimoptions('ga','Display','off','PopulationSize',2,'EliteCount',2, ...
 'InitialPopulationMatrix',[0;0],'MaxGenerations',1);
opts=optimoptions('patternsearch','Display','off','SearchFcn',{@searchga,1,local}, ...
 'MaxIterations',1,'MaxFunctionEvaluations',10);
caught=false;
try
 patternsearch(@objective,0,[],[],[],[],[],[],[],opts);
catch err
 caught=true; assert(strcmp(err.identifier,'PatternGA:Original'));
end
assert(caught && psga_calls==2);
function f=objective(x)
global psga_calls; psga_calls=psga_calls+1;
if psga_calls==2, error('PatternGA:Original','original genetic-search objective'); end
f=x^2;
end
"#;
