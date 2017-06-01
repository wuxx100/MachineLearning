year=13;
dataFolder=['navData/',int2str(year),'/'];
for day=1:317
    dayFolder=[dataFolder,int2str(day)];

        dataFiles=dir(dayFolder);
        numOfData=length(dataFiles);
        for i=1:numOfData
                dataFile=dataFiles(i);
        if strfind(dataFile.name,'13n')
                disp(dataFile.name);
                [navmes,ionpar] = READ_RIN_NAV([dayFolder,'/',dataFile.name]);
                navmes=navmes';
                ionpar=ionpar';
                st=strsplit(dataFile.name,'.');
                stationName=st{1};
                navmesFileName=[dayFolder,'/',stationName,'_','navmes.csv'];
                ionparFileName=[dayFolder,'/',stationName,'_','ionpar.csv'];
                disp(size(navmes))
                csvwrite(navmesFileName,navmes);
                csvwrite(ionparFileName,ionpar);

        end
        end
end
