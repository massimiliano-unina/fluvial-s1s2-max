close all, clear all, clc; 

orbits = {'139', '168'}

for n_orbit = 1:2
    orbit = orbits{n_orbit}; 
    if orbit == '139'
        N = 2; 
    else
        N = 3;
    end
    for TS = 0:N
        
        IMD = imread(['D:\08_month_40m\',orbit,'_orbit\TS_',num2str(N),'\hrl_2015\IMD_2015_020m_eu_03035_d05_Merge_wgs84.tif']);
        TCD = imread(['D:\08_month_40m\',orbit,'_orbit\TS_',num2str(N),'\hrl_2015\TCD_2015_020m_eu_03035_d05_Merge_wgs84.tif']);
        WAW = imread(['D:\08_month_40m\',orbit,'_orbit\TS_',num2str(N),'\hrl_2015\WAW_2015_020m_eu_03035_d06_Merge_wgs84.tif']);
        
    end
end