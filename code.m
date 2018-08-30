%This code demonstrates the techniques of color image processing and
%detection of objects in an image using color; Color histograms and image
%compression methods are also studied
%Code written by Sai Saradha K.L. (MS, Computer Engineering, Fall 2016)

 
clc;
clear all;
 
%Collecting samples from the images with traffic cones
cone1_o=imgaussfilt(imread('images/hw2_cone_training_1.jpg'),1.5);
figure, imshow(cone1_o,[]);
title('Original Cone1');
cone1=rgb2hsv(imcrop(cone1_o));
figure, imshow(cone1,[]);
cone2=rgb2hsv(imcrop(cone1_o));
figure, imshow(cone2,[]);
 
cone2_o=imgaussfilt(imread('images/hw2_cone_training_2.jpg'),1.5);
figure, imshow(cone2_o,[]);
title('Original Cone2');
cone3=rgb2hsv(imcrop(cone2_o));
figure, imshow(cone3,[]);
cone4=rgb2hsv(imcrop(cone2_o));
figure, imshow(cone4,[]);
cone5=rgb2hsv(imcrop(cone2_o));
figure, imshow(cone5,[]);
 
cone3_o=imgaussfilt(imread('images/hw2_cone_training_3.jpg'),1.5);
figure, imshow(cone3_o,[]);
title('Original Cone3');
cone6=rgb2hsv(imcrop(cone3_o));
figure, imshow(cone6,[]);
cone7=rgb2hsv(imcrop(cone3_o));
figure, imshow(cone7,[]);
cone8=rgb2hsv(imcrop(cone3_o));
figure, imshow(cone8,[]);
cone9=rgb2hsv(imcrop(cone3_o));
figure, imshow(cone9,[]);
cone10=rgb2hsv(imcrop(cone3_o));
figure, imshow(cone10,[]);
 
cone4_o=imgaussfilt(imread('images/hw2_cone_training_4.jpg'),1.5);
figure, imshow(cone4_o,[]);
title('Original Cone4');
cone11=rgb2hsv(imcrop(cone4_o));
figure, imshow(cone11,[]);
 
cone5_o=imgaussfilt(imread('images/hw2_cone_training_5.jpg'),1.5);
figure, imshow(cone5_o,[]);
title('Original Cone5');
cone12=rgb2hsv(imcrop(cone5_o));
figure, imshow(cone12,[]);
 
cone_cell={cone1, cone2, cone3, cone4, cone5, cone6, cone7, cone8, cone9, cone10, cone11, cone12};
 
%Now we have twelve samples. We need to do the following on these images:
%We need only H and S components, so we separate them out and then
%string them out as a vector
h_cone=zeros;
s_cone=zeros;
h_cone_all=zeros;
s_cone_all=zeros;
h_resh=zeros;
s_resh=zeros;
 
for i=1:size(cone_cell,2)
    h_resh=cone_cell{1,i}(:,:,1);
    [r,c]=size(h_resh);
    s_resh=cone_cell{1,i}(:,:,2);
    [r1,c1]=size(s_resh);
    h_cone=reshape(cone_cell{1,i}(:,:,1),1,r*c);
    s_cone=reshape(cone_cell{1,i}(:,:,2),1,r1*c1);
    h_cone_all=[h_cone_all h_cone];
    s_cone_all=[s_cone_all s_cone];
end
 
%Finding the mean and Standard Deviation of the components:
h_mean=mean(h_cone_all);
s_mean=mean(s_cone_all);
h_std=std2(h_cone_all);
s_std=std2(s_cone_all);
 
%Range for the cone:
h_lt=h_mean-(1*h_std);
h_ut=h_mean+(1*h_std);
s_lt=s_mean-(1*s_std);
s_ut=s_mean+(1*s_std);
 
%Segmenting the images based on this mean and Std:
%Testing on the training images first:
tr_img_1=rgb2hsv(cone1_o);
tr_img_2=rgb2hsv(cone2_o);
tr_img_3=rgb2hsv(cone3_o);
tr_img_4=rgb2hsv(cone4_o);
tr_img_5=rgb2hsv(cone5_o);
 
tr_img={tr_img_1,tr_img_2,tr_img_3,tr_img_4,tr_img_5};
num_img=size(tr_img,2);
 
for i=1:num_img
    [r c ~]=size(tr_img{1,i});
    for j=1:r
        for k=1:c
            if((h_lt<=tr_img{1,i}(j,k,1))&&(tr_img{1,i}(j,k,1)<=h_ut)&& (s_lt<=tr_img{1,i}(j,k,2))&&(tr_img{1,i}(j,k,2)<=s_ut))
                tr_img{1,i}(j,k,:)=1;
            else
                tr_img{1,i}(j,k,:)=0;
            end
        end
    end
    figure, imshow(tr_img{1,i});
end
 
%Now, testing on the test image:
%---- Write code
 
%Y, Cb, Cr color space:
cone1_ycbcr=rgb2ycbcr(imcrop(cone1_o));
cone2_ycbcr=rgb2ycbcr(imcrop(cone1_o));
cone3_ycbcr=rgb2ycbcr(imcrop(cone2_o));
cone4_ycbcr=rgb2ycbcr(imcrop(cone2_o));
cone5_ycbcr=rgb2ycbcr(imcrop(cone2_o));
cone6_ycbcr=rgb2ycbcr(imcrop(cone3_o));
cone7_ycbcr=rgb2ycbcr(imcrop(cone3_o));
cone8_ycbcr=rgb2ycbcr(imcrop(cone3_o));
cone9_ycbcr=rgb2ycbcr(imcrop(cone3_o));
cone10_ycbcr=rgb2ycbcr(imcrop(cone3_o));
cone11_ycbcr=rgb2ycbcr(imcrop(cone4_o));
cone12_ycbcr=rgb2ycbcr(imcrop(cone5_o));
 
cone_cell_ycbcr={cone1_ycbcr, cone2_ycbcr, cone3_ycbcr, cone4_ycbcr, cone5_ycbcr, cone6_ycbcr, cone7_ycbcr, cone8_ycbcr, cone9_ycbcr, cone10_ycbcr, cone11_ycbcr, cone12_ycbcr};
 
%Now we have twelve samples. We need to do the following on these images:
%We need only H and S components, so we separate them out and then
%string them out as a vector
h_cone_ycbcr=zeros;
s_cone_ycbcr=zeros;
h_cone_all_ycbcr=zeros;
s_cone_all_ycbcr=zeros;
h_resh_ycbcr=zeros;
s_resh_ycbcr=zeros;
 
for i=1:size(cone_cell_ycbcr,2)
    h_resh_ycbcr=cone_cell_ycbcr{i}(:,:,2);
    [r,c]=size(h_resh_ycbcr);
    s_resh_ycbcr=cone_cell_ycbcr{i}(:,:,3);
    [r1,c1]=size(s_resh_ycbcr);
    h_cone_ycbcr=reshape(cone_cell_ycbcr{i}(:,:,2),1,r*c);
    s_cone_ycbcr=reshape(cone_cell_ycbcr{i}(:,:,3),1,r1*c1);
    h_cone_all_ycbcr=[h_cone_all_ycbcr h_cone_ycbcr];
    s_cone_all_ycbcr=[s_cone_all_ycbcr s_cone_ycbcr];
end
 
%Finding the mean and Standard Deviation of the components:
h_mean_ycbcr=mean(h_cone_all_ycbcr);
s_mean_ycbcr=mean(s_cone_all_ycbcr);
h_std_ycbcr=std2(h_cone_all_ycbcr);
s_std_ycbcr=std2(s_cone_all_ycbcr);
 
%Mean and STD Range for the cone:
h_lt_ycbcr=h_mean_ycbcr-(3*h_std_ycbcr);
h_ut_ycbcr=h_mean_ycbcr+(3*h_std_ycbcr);
s_lt_ycbcr=s_mean_ycbcr-(3*s_std_ycbcr);
s_ut_ycbcr=s_mean_ycbcr+(3*s_std_ycbcr);
 
%Segmenting the images based on this mean and Std:
%Testing on the training images first:
tr_img_1_ycbcr=rgb2ycbcr(cone1_o);
tr_img_2_ycbcr=rgb2ycbcr(cone2_o);
tr_img_3_ycbcr=rgb2ycbcr(cone3_o);
tr_img_4_ycbcr=rgb2ycbcr(cone4_o);
tr_img_5_ycbcr=rgb2ycbcr(cone5_o);
 
tr_img_ycbcr={tr_img_1_ycbcr,tr_img_2_ycbcr,tr_img_3_ycbcr,tr_img_4_ycbcr,tr_img_5_ycbcr};
num_img_ycbcr=size(tr_img_ycbcr,2);
 
tr_img_ycbcr2=cell(1,num_img_ycbcr);
seg_img=cell(1,num_img_ycbcr);
for i=1:num_img_ycbcr
    [r c ~]=size(tr_img_ycbcr{1,i});
    tr_img_ycbcr2{1,i}=tr_img_ycbcr{1,i};
    for j=1:r
        for k=1:c
            if((h_lt_ycbcr<=tr_img_ycbcr{1,i}(j,k,2))&&(tr_img_ycbcr{1,i}(j,k,2)<=h_ut_ycbcr)&& (s_lt_ycbcr<=tr_img_ycbcr{1,i}(j,k,3))&&(tr_img_ycbcr{1,i}(j,k,3)<=s_ut_ycbcr))
                tr_img_ycbcr{1,i}(j,k,:)=1;
                tr_img_ycbcr2{1,i}(j,k,:)=tr_img_ycbcr2{1,i}(j,k,:);
            else
                tr_img_ycbcr{1,i}(j,k,:)=0;
                tr_img_ycbcr2{1,i}(j,k,:)=128;
            end
        end
    end
    figure, imshow(double(tr_img_ycbcr{1,i}));
    title('Segmented Binary Image');
    figure, imshow(tr_img_ycbcr2{1,i});
    title('Segmented image in Ycbcr space');
    seg_img{1,i}=ycbcr2rgb(tr_img_ycbcr2{1,i});
    figure, imshow(seg_img{1,i});
    title('Segmented image in RGB space');
end
 
%Now testing on test images:
test_img_1=imgaussfilt(imread('hw2_cone_testing_1.jpg'),1.5);
test_img_2=imgaussfilt(imread('hw2_cone_testing_2.jpg'),1.5);
test_img_1_ycbcr=rgb2ycbcr(test_img_1);
test_img_2_ycbcr=rgb2ycbcr(test_img_2);
test_img_ycbcr={test_img_1_ycbcr,test_img_2_ycbcr};
num_test_img_ycbcr=size(test_img_ycbcr,2);
 
test_img_ycbcr2=cell(1,num_test_img_ycbcr);
seg_img2=cell(1,num_test_img_ycbcr);
for i=1:num_test_img_ycbcr
    [r c ~]=size(test_img_ycbcr{1,i});
    test_img_ycbcr2{1,i}=test_img_ycbcr{1,i};
    for j=1:r
        for k=1:c
            if((h_lt_ycbcr<=test_img_ycbcr{1,i}(j,k,2))&&(test_img_ycbcr{1,i}(j,k,2)<=h_ut_ycbcr)&& (s_lt_ycbcr<=test_img_ycbcr{1,i}(j,k,3))&&(test_img_ycbcr{1,i}(j,k,3)<=s_ut_ycbcr))
                test_img_ycbcr{1,i}(j,k,:)=1;
                test_img_ycbcr2{1,i}(j,k,:)=test_img_ycbcr2{1,i}(j,k,:);
            else
                test_img_ycbcr{1,i}(j,k,:)=0;
                test_img_ycbcr2{1,i}(j,k,:)=128;
            end
        end
    end
    figure, imshow(double(test_img_ycbcr{1,i}));
    title('Segmented Binary Image - test');
    figure, imshow(test_img_ycbcr2{1,i});
    title('Segmented image in Ycbcr space - test');
    seg_img2{1,i}=ycbcr2rgb(test_img_ycbcr2{1,i});
    figure, imshow(seg_img2{1,i});
    title('Segmented image in RGB space - test');
end
 
%%
%Question - 2nd half - using probability to detect orange color
%Reading the mask images:
mask_img_1=imread('hw2_cone_training_map_1.png');
mask_img_2=imread('hw2_cone_training_map_2.png');
mask_img_3=imread('hw2_cone_training_map_3.png');
mask_img_4=imread('hw2_cone_training_map_4.png');
mask_img_5=imread('hw2_cone_training_map_5.png');
 
%Since mask image is in png format and training image is in jpeg format, to
%ensure that they are in the proper range:
%Check if this is required:
 
%Multiplying the original image with the mask
original_cell={cone1_o,cone2_o,cone3_o, cone4_o, cone5_o};
mask_cell={mask_img_1,mask_img_2,mask_img_3,mask_img_4,mask_img_5};
mask_comp_cell={imcomplement(mask_img_1),imcomplement(mask_img_2),imcomplement(mask_img_3),imcomplement(mask_img_4),imcomplement(mask_img_5)};
size_orig=size(original_cell,2);
size_mask=size(mask_cell,2);
cone_cell_prob=cell(1,size_orig);
nocone_cell_prob=cell(1,size_mask);
cone_cell_ycbcr_prob=cell(1,size_orig);
nocone_cell_ycbcr_prob=cell(1,size_mask);
for i=1:size_orig
      cone_cell_prob{1,i}=original_cell{1,i}.*repmat((uint8(mask_cell{1,i})),[1,1,3]);
%       figure, imshow(cone_cell_prob{1,i});
%       title('Cone only image in RGB space');
      nocone_cell_prob{1,i}=original_cell{1,i}.*repmat((uint8(mask_comp_cell{1,i})),[1,1,3]); 
%       figure, imshow(nocone_cell_prob{1,i});
%       title('No Cone image in RGB space');
      cone_cell_ycbcr_prob{1,i}=rgb2ycbcr(cone_cell_prob{1,i});
%       figure, imshow(cone_cell_ycbcr_prob{1,i});
%       title('Cone only image in YCbCr space');
      nocone_cell_ycbcr_prob{1,i}=rgb2ycbcr(nocone_cell_prob{1,i});
%       figure, imshow(nocone_cell_ycbcr_prob{1,i});
%       title('No Cone image in YCbCr space');
end
 
h_cone_all_ycbcr_prob=zeros;
s_cone_all_ycbcr_prob=zeros;
h_cone_all_ycbcr_prob_nc=zeros;
s_cone_all_ycbcr_prob_nc=zeros;
 
for i=1:size(cone_cell_ycbcr_prob,2)
    h_resh_ycbcr_prob=cone_cell_ycbcr_prob{1,i}(:,:,2);
    [r,c]=size(h_resh_ycbcr_prob);
    s_resh_ycbcr_prob=cone_cell_ycbcr_prob{1,i}(:,:,3);
    [r1,c1]=size(s_resh_ycbcr_prob);
    h_cone_ycbcr_prob=reshape(h_resh_ycbcr_prob,1,r*c);
    s_cone_ycbcr_prob=reshape(s_resh_ycbcr_prob,1,r1*c1);
    h_cone_all_ycbcr_prob=[h_cone_all_ycbcr_prob h_cone_ycbcr_prob];
    s_cone_all_ycbcr_prob=[s_cone_all_ycbcr_prob s_cone_ycbcr_prob];
end
 
for i=1:size(nocone_cell_ycbcr_prob,2)
    h_resh_ycbcr_prob_nc=nocone_cell_ycbcr_prob{1,i}(:,:,2);
    [r2,c2]=size(h_resh_ycbcr_prob_nc);
    s_resh_ycbcr_prob_nc=nocone_cell_ycbcr_prob{1,i}(:,:,3);
    [r3,c3]=size(s_resh_ycbcr_prob_nc);
    h_cone_ycbcr_prob_nc=reshape(h_resh_ycbcr_prob_nc,1,r2*c2);
    s_cone_ycbcr_prob_nc=reshape(s_resh_ycbcr_prob_nc,1,r3*c3);
    h_cone_all_ycbcr_prob_nc=[h_cone_all_ycbcr_prob_nc h_cone_ycbcr_prob_nc];
    s_cone_all_ycbcr_prob_nc=[s_cone_all_ycbcr_prob_nc s_cone_ycbcr_prob_nc];
end
 
%Eliminating the 0s in the vector for calculating the distribution:
h_cone_all_ycbcr_prob=h_cone_all_ycbcr_prob(h_cone_all_ycbcr_prob~=0);
s_cone_all_ycbcr_prob=s_cone_all_ycbcr_prob(s_cone_all_ycbcr_prob~=0);
h_cone_all_ycbcr_prob_nc=h_cone_all_ycbcr_prob_nc(h_cone_all_ycbcr_prob_nc~=0);
s_cone_all_ycbcr_prob_nc=s_cone_all_ycbcr_prob_nc(s_cone_all_ycbcr_prob_nc~=0);
 
%Formulating the distributions:
%Distribution for traffic cones:
Distplot= zeros(256);
Distplot_nc=zeros(256);
h_cone_all_ycbcr_prob = round(h_cone_all_ycbcr_prob);
s_cone_all_ycbcr_prob = round(s_cone_all_ycbcr_prob);
h_cone_all_ycbcr_prob_nc=round(h_cone_all_ycbcr_prob_nc);
s_cone_all_ycbcr_prob_nc=round(s_cone_all_ycbcr_prob_nc);
for i = 1:length(h_cone_all_ycbcr_prob)
   Distplot(h_cone_all_ycbcr_prob(i), s_cone_all_ycbcr_prob(i)) = Distplot(h_cone_all_ycbcr_prob(i), s_cone_all_ycbcr_prob(i)) + 1;
  
end
 prob_dist=Distplot./length(h_cone_all_ycbcr_prob);
figure, surf(Distplot);
 
%Distribution for non traffic cones:
for i = 1:length(h_cone_all_ycbcr_prob_nc)
   Distplot_nc(h_cone_all_ycbcr_prob_nc(i), s_cone_all_ycbcr_prob_nc(i)) = Distplot_nc(h_cone_all_ycbcr_prob_nc(i), s_cone_all_ycbcr_prob_nc(i)) + 1;
end
prob_dist_nc=Distplot_nc./length(h_cone_all_ycbcr_prob_nc);
figure, surf(Distplot_nc);
 
%Now segmenting the image based on the probability distribution:
%We now have five training images and two test images in YCbCr space
%tr_img_ycbcr={tr_img_1_ycbcr,tr_img_2_ycbcr,tr_img_3_ycbcr,tr_img_4_ycbcr,tr_img_5_ycbcr};
%test_img_ycbcr={test_img_1_ycbcr,test_img_2_ycbcr};
tr_img_1_ycbcr=rgb2ycbcr(cone1_o);
tr_img_2_ycbcr=rgb2ycbcr(cone2_o);
tr_img_3_ycbcr=rgb2ycbcr(cone3_o);
tr_img_4_ycbcr=rgb2ycbcr(cone4_o);
tr_img_5_ycbcr=rgb2ycbcr(cone5_o);
 
tr_img_ycbcr={tr_img_1_ycbcr,tr_img_2_ycbcr,tr_img_3_ycbcr,tr_img_4_ycbcr,tr_img_5_ycbcr};
num_img_ycbcr=size(tr_img_ycbcr,2);
 
tr_img_ycbcr2=cell(1,num_img_ycbcr);
seg_img=cell(1,num_img_ycbcr);
 
for i=1:num_img_ycbcr
    [r c ~]=size(tr_img_ycbcr{1,i});
    tr_img_ycbcr2{1,i}=tr_img_ycbcr{1,i};
    for j=1:r
        for k=1:c
            cb_val=tr_img_ycbcr{1,i}(j,k,2);
            cr_val=tr_img_ycbcr{1,i}(j,k,3);
            if(prob_dist(cb_val,cr_val)>prob_dist_nc(cb_val,cr_val))
                tr_img_ycbcr{1,i}(j,k,:)=1;
                tr_img_ycbcr2{1,i}(j,k,:)=tr_img_ycbcr2{1,i}(j,k,:);
            else
                tr_img_ycbcr{1,i}(j,k,:)=0;
                tr_img_ycbcr2{1,i}(j,k,:)=128;
            end
        end
    end
    tr_img_ycbcr{1,i}=bwareaopen(tr_img_ycbcr{1,i},5000);
    figure, imshow(double(tr_img_ycbcr{1,i}));
    title('Segmented Binary Image');
    figure, imshow(tr_img_ycbcr2{1,i});
    title('Segmented image in Ycbcr space');
    seg_img{1,i}=ycbcr2rgb(tr_img_ycbcr2{1,i});
    figure, imshow(seg_img{1,i});
    title('Segmented image in RGB space');
end
 
%Now testing on test images:
test_img_1=imgaussfilt(imread('hw2_cone_testing_1.jpg'),1.5);
test_img_2=imgaussfilt(imread('hw2_cone_testing_2.jpg'),1.5);
test_img_1_ycbcr=rgb2ycbcr(test_img_1);
test_img_2_ycbcr=rgb2ycbcr(test_img_2);
test_img_ycbcr={test_img_1_ycbcr,test_img_2_ycbcr};
num_test_img_ycbcr=size(test_img_ycbcr,2);
 
test_img_ycbcr2=cell(1,num_test_img_ycbcr);
seg_img2=cell(1,num_test_img_ycbcr);
for i=1:num_test_img_ycbcr
    [r c ~]=size(test_img_ycbcr{1,i});
    test_img_ycbcr2{1,i}=test_img_ycbcr{1,i};
    for j=1:r
        for k=1:c
            cb_val=test_img_ycbcr{1,i}(j,k,2);
            cr_val=test_img_ycbcr{1,i}(j,k,3);
            if(prob_dist(cb_val,cr_val)>=prob_dist_nc(cb_val,cr_val))
                test_img_ycbcr{1,i}(j,k,:)=1;
                test_img_ycbcr2{1,i}(j,k,:)=test_img_ycbcr2{1,i}(j,k,:);
            else
                test_img_ycbcr{1,i}(j,k,:)=0;
                test_img_ycbcr2{1,i}(j,k,:)=128;
            end
        end
    end
    test_img_ycbcr{1,i}=bwareaopen(test_img_ycbcr{1,i},1000);
    figure, imshow(double(test_img_ycbcr{1,i}));
    title('Segmented Binary Image - test');
    figure, imshow(test_img_ycbcr2{1,i});
    title('Segmented image in Ycbcr space - test');
    seg_img2{1,i}=ycbcr2rgb(test_img_ycbcr2{1,i});
    figure, imshow(seg_img2{1,i});
    title('Segmented image in RGB space - test');
end

%Routine to plot distribution of orange color from traffic cone images%
hsize = size(h_cone_all_ycbcr);
ssize = size(s_cone_all_ycbcr);
count_hs = zeros(256,256);
for i=1:hsize(1,2)
    if h_cone_all_ycbcr(1,i)~=0 && s_cone_all_ycbcr(1,i)~=0
       count_hs(h_cone_all_ycbcr(1,i),s_cone_all_ycbcr(1,i))=count_hs(h_cone_all_ycbcr(1,i),s_cone_all_ycbcr(1,i))+1;  
    end
end
 
l=1;
for i=1:256
    for j=1:256
        if count_hs(i,j)~=0;
            X(l,1)=i;
            Y(l,1)= j;
            Z(l,1)=count_hs(i,j);
            l=l+1;
        end
     end
end
 figure, m = scatter3(Y,X,Z);
 xlim([0 250]),ylim([0 250]);
 axis('square');view(-30,20);
 title('Distribution of C_{b} and C_{r} values for Traffic Cone orange','FontSize',13);
 xlabel('C_{b}','FontSize',15);ylabel('C_{r}','FontSize',15);zlabel('Count','FontSize',15);
 m.MarkerFaceColor = [255/255 140/255 0/255];
 
 %Routine to plot PDF of orange for traffic cone and non-traffic cone%
 hcsize = size(h_cone_all_ycbcr_prob);
 scsize = size(s_cone_all_ycbcr_prob);
 hncsize = size(h_cone_all_ycbcr_prob_nc);
 sncsize = size(s_cone_all_ycbcr_prob_nc);
 count_hsc = zeros(256,256);
 count_hsnc = zeros(256,256);
for i=1:hcsize(1,2)
    if h_cone_all_ycbcr_prob(1,i)~=0 && s_cone_all_ycbcr_prob(1,i)~=0
       count_hsc(h_cone_all_ycbcr_prob(1,i),s_cone_all_ycbcr_prob(1,i))=count_hsc(h_cone_all_ycbcr_prob(1,i),s_cone_all_ycbcr_prob(1,i))+1;  
    end
end
 
for i=1:hncsize(1,2)
    if h_cone_all_ycbcr_prob_nc(1,i)~=0 && s_cone_all_ycbcr_prob_nc(1,i)~=0
       count_hsnc(h_cone_all_ycbcr_prob_nc(1,i),s_cone_all_ycbcr_prob_nc(1,i))=count_hsnc(h_cone_all_ycbcr_prob_nc(1,i),s_cone_all_ycbcr_prob_nc(1,i))+1;  
    end
end
 
X = zeros;
Y=zeros;
Z=zeros;
l=1;
for i=1:256
    for j=1:256
        if count_hsc(i,j)~=0;
            X(l,1)=i;
            Y(l,1)= j;
            Z(l,1)=count_hsc(i,j);
            l=l+1;
        end
     end
end
l=1;
for i=1:256
    for j=1:256
        if count_hsnc(i,j)~=0;
            Xnc(l,1)=i;
            Ync(l,1)= j;
            Znc(l,1)=count_hsnc(i,j);
            l=l+1;
        end
     end
end
 
 figure, m = scatter3(Ync,Xnc,Znc,'MarkerFaceColor',[0 1 0]);hold on;
 scatter3(Y,X,Z,'MarkerFaceColor',[255/255 140/255 0/255]);
 xlim([0 250]),ylim([0 250]);
 axis('square');%view(-30,20);
 view(2);
 xlabel('C_{b}','FontSize',15);ylabel('C_{r}','FontSize',15);zlabel('Count','FontSize',15);
 
