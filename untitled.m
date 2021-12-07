open PT2Regler.slx
set_param('PT2Regler/Gain','Gain','Kr') 
modelworkspace = get_param('PT2Regler','ModelWorkspace');
assignin(modelworkspace,'Kr',Simulink.Parameter(0.5));
set_param('PT2Regler/Transfer Fcn','Numerator','N') 
model_workspace = get_param('PT2Regler','ModelWorkspace');
assignin(model_workspace,'N',Simulink.Parameter([0.9 1]));
set_param('PT2Regler/Transfer Fcn','Denominator','D') 
model__workspace = get_param('PT2Regler','ModelWorkspace');
assignin(model__workspace,'D',Simulink.Parameter([0 0.2 1]));
set_param('PT2Regler/Transfer Fcn1','Numerator','N1') 
model_workspace1 = get_param('PT2Regler','ModelWorkspace');
assignin(model_workspace1,'N1',Simulink.Parameter([0.5 1]));
set_param('PT2Regler/Transfer Fcn1','Denominator','D1') 
model__workspace2 = get_param('PT2Regler','ModelWorkspace');
assignin(model__workspace2,'D1',Simulink.Parameter([0 0.25 1]));
set_param('PT2Regler','ParameterArgumentNames','Kr,N,D,N1,D1');
