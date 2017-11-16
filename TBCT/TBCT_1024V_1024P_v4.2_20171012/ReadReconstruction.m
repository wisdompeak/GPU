%%
clc;

iter='21';
TV='0.00';
Lip='16.00';
view='360';
VoxelSize='1500';
initialAngle='0';
endAngle='359';

directory='Reconstructed_images_TBCT/';

Object='Phantom';

% Read SART results
fp = fopen([directory,Object,'_256_256_128_',VoxelSize,'um_new_view_',view,'_(',initialAngle,',',endAngle,')_TV_',TV,'_L1_0.00_Lip_',Lip,'.recon'],'rb');

%Iterative results (preferred)
% fp = fopen([directory,Object,'_256_256_128_',VoxelSize,'um_iterative_',iter,'_view_',view,'_(',initialAngle,',',endAngle,')_TV_',TV,'_L1_0.00_Lip_',Lip,'.recon'],'rb');

f_TV_L1= fread(fp, 256*256*128,'float');
f_TV_L1=reshape(f_TV_L1, [256 256 128]);
fclose(fp);

%%

GIFname = 'Test.gif';
% sz=get(0,'screensize');
% figure('outerposition',sz);
figure;

% figure1 = figure('Color',[1 1 1]);

for slice=128%100:5:240
    
f_show_TV_L1 = squeeze(f_TV_L1(slice,:,:));     % Extract Vertical Slice: look from 180 direction
% f_show_TV_L1 = squeeze(f_TV_L1(:,slice,:));   % Extract Vertical Slice: look from 90 direction
% f_show_TV_L1 = f_TV_L1(:,:,slice);    % Extract Horizontal Slice

% A=rot90((f_show_TV_L1'),2);
A=f_show_TV_L1;

% Flip the slice so that it looks normal orientation

% A= max(max(A))-A;
% A tricky way to make the slice look better

A=A';

% imagesc(A(7:303,380:740));
imagesc(A);

colormap(gray);
colorbar;
% axis equal
% axis off;
% caxis([-20 800]);

title(num2str(slice));

pause(0.5);

%     drawnow;
% 
%     frame = getframe(1);
%     im = frame2im(frame);
%     [imind,cm] = rgb2ind(im,256);
%     if slice == 310
%         imwrite(imind,cm,GIFname,'gif', 'Loopcount',inf);
%     elseif slice>310
%         imwrite(imind,cm,GIFname,'gif','WriteMode','append');
%     end

end

%%
clc; format long;
filename=['object_func_',Object,'_view_',view,'_(',initialAngle,',',endAngle,')_TV_',TV,'_Lip_',Lip,'.bin'];
fp = fopen([directory,filename]);
obj_fun=fread(fp,'double');
fclose(fp);
x=length(obj_fun);
obj_fun=reshape(obj_fun,3,x/3);
obj_fun'

figure;
subplot(2,2,1); plot(obj_fun(2,:));
subplot(2,2,2); plot(obj_fun(1,:));
subplot(2,2,3); plot(obj_fun(3,:));
