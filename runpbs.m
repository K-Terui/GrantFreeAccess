%コア数は0から数える
clear
% job1 = batch('main_GFA','Pool',111,'AutoAddClientPath',false);
job2 = batch('main_GFA_MultiAntennas','Pool',111,'AutoAddClientPath',false);

%pbsから直接引き出し
%findJob(parcluster('klab-cluster remote R2020a'), 'ID', ?);?にはjob番号