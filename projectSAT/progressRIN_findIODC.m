clear
clc
year=13;

for day=4:300
    

    
    readFolder=['navData/',int2str(year),'/',int2str(day),'/'];

    writeFolder='navData/';

        dataFiles=dir(readFolder);
        numOfData=length(dataFiles);
        for i=1:numOfData
            dataFile=dataFiles(i);
            if strfind(dataFile.name,'navmes.csv')
                disp([readFolder,'/',dataFile.name]);
                navme=csvread([readFolder,'/',dataFile.name]);
                nowSize=size(navme);
                for j =1:nowSize(1)
                    data=navme(j,:);
                    prnNum=data(1);
                    nowyear=data(2);
                    dayForMonth=data(4);
                    monthForYear=data(3);
                    dayForYear=datenum(nowyear,monthForYear,dayForMonth)-datenum(nowyear,1,1)+1;
                    fileName=[writeFolder,int2str(nowyear),'/',int2str(dayForYear),'/','findIODC/','prn',num2str(prnNum),'.csv'];
                    dlmwrite(fileName,data,'delimiter',',','-append');
                end
            end

        end

end
