open System.slx
set_param('System/Weg','InitialCondition','x0') 
modelworkspace = get_param('System','ModelWorkspace');
assignin(modelworkspace,'x0',Simulink.Parameter(0.3));
set_param('System','ParameterArgumentNames','x0');