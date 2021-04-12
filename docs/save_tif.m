function save_tif(name_image, file, output_name)
%This function saves the image name_image in input in a TIFF file, using
%the information of the file in input. The num Input is the number that
%you can use in the output file. 

%  i = 2;
 h=imfinfo(file);
%  name_image = single(imread(file))/(2^(15));
% % %         name_image = name_image(end/3+1:end,end/3+1:end); %Venice
% %        name_image = name_image(end/3+1:end,1:(2/3)*end); % Guinea
%  name_image = name_image(end/2+1:end,end/2+1:end); %Tunisie

     t = Tiff(output_name,'w');
    t.setTag('Photometric',Tiff.Photometric.MinIsBlack); % assume grayscale
    tagstruct.StripOffsets = h.StripOffsets;
%     t.setTag('StripOffsets',a);
    t.setTag('BitsPerSample',32);
%     t.setTag('SamplesPerPixel',1);
    t.setTag('SamplesPerPixel',3);%h.SamplesPerPixel);   
    t.setTag('RowsPerStrip', h.RowsPerStrip);
    tagstruct.StripByteCounts = h.StripByteCounts;

    tagstruct.XResolution = h.XResolution;
    tagstruct.YResolution = h.YResolution;
    tagstruct.ResolutionUnit = h.ResolutionUnit;
    t.setTag('Thresholding',h.Thresholding);
    tagstruct.Offset = h.Offset;
    t.setTag('SampleFormat',Tiff.SampleFormat.IEEEFP);
    t.setTag('ImageLength',size(name_image,1));%size(i2,1)
    t.setTag('ImageWidth',size(name_image,2));%size(i2,1)
    t.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);
    t.write(single(name_image));
    t.close();
end
