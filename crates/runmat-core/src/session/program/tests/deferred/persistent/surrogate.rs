//! Parsed public ingress exercises the actual selected-program producer and
//! restored current-source reentry, not a mocked callable resolver.
use super::*;

async fn run(session: &mut RunMatSession, source: &str) -> Result<(), crate::RunError> {
    let outcome = super::run(session, source).await?;
    assert!(
        !outcome.has_errors(),
        "public surrogate execution: {:?}",
        outcome.diagnostics
    );
    Ok(())
}

#[tokio::test]
async fn surrogate_public_vectorized_partial_batch_and_known_measurements_keep_actual_counts() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    run(
        &mut session,
        r#"
global surrogateWidths; surrogateWidths=[];
function y=surrogateObserved(x)
global surrogateWidths;
surrogateWidths(end+1)=size(x,1);
y=x.^2;
end
opts=optimoptions('surrogateopt','Display','off','PlotFcn',[],...
'InitialPoints',[-2;-1;0;1;2],'UseVectorized',true,...
'BatchUpdateInterval',3,'MaxFunctionEvaluations',5);
[x,f,flag,out,trials]=surrogateopt(@surrogateObserved,-3,3,opts);
assert(x==0 && f==0 && flag==0 && out.funccount==5 && out.constrviolation==0);
assert(isequal(surrogateWidths,[3 2]));
assert(isequal(trials.X,[-2;-1;0;1;2]) && isequal(trials.Fval,[4;1;0;1;4]));
initial=struct('X',[-1;0;1],'Fval',[1;0;1],'Ineq',zeros(3,0));
opts=optimoptions('surrogateopt','Display','off','PlotFcn',[],...
'InitialPoints',initial,'MaxFunctionEvaluations',0);
[x,f,flag,out,trials]=surrogateopt(@(x)error('unexpected objective call'),-2,2,opts);
assert(x==0 && f==0 && flag==0 && out.funccount==0);
assert(isequal(trials.X,[-1;0;1]) && isequal(trials.Fval,[1;0;1]));
"#,
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn surrogate_checkpoint_cold_resume_retains_original_code_capture_and_total_point_budget() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("surrogate-original.mat");
    let path = path.to_str().unwrap().replace('\'', "''");
    let mut producer = RunMatSession::with_options(false, false).unwrap();
    run(
        &mut producer,
        "function y=savedSurrogateLeaf(x); y=x(1)^2+7; end",
    )
    .await
    .unwrap();
    run(
        &mut producer,
        &format!(
            r#"
offset=3; objective=@(x)savedSurrogateLeaf(x)+offset;
opts=optimoptions('surrogateopt','Display','off','PlotFcn',[],...
'InitialPoints',[0;1;2],'MaxFunctionEvaluations',1,'CheckpointFile','{path}');
[x,f,flag,out,trials]=surrogateopt(objective,-3,3,opts);
assert(x==0 && f==10 && flag==0 && out.funccount==1);
assert(isequal(trials.X,0) && trials.Fval==10);
"#
        ),
    )
    .await
    .unwrap();
    drop(producer);
    let mut cold = RunMatSession::with_options(false, false).unwrap();
    run(&mut cold, "function y=savedSurrogateLeaf(x); y=-1000; end")
        .await
        .unwrap();
    run(
        &mut cold,
        &format!(
            r#"
offset=1000;
more=struct('Display','off','PlotFcn',[],'MaxFunctionEvaluations',3,'CheckpointFile','');
[x,f,flag,out,trials]=surrogateopt('{path}',more);
assert(x==0 && f==10 && flag==0 && out.funccount==3);
assert(isequal(trials.X,[0;1;2]) && isequal(trials.Fval,[10;11;14]));
"#
        ),
    )
    .await
    .unwrap();
}
