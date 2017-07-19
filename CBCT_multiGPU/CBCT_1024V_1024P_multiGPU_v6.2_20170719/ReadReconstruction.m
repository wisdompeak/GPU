%%
clc;

view='200';
VoxelSize='1000';
initialAngle='0';
endAngle='199';
TV='1.00';
Lip='64.00';
iter='10';

directory='Recon_Phantom_256/';
Object='Phantom1';

% Read latest results
fp = fopen([directory,Object,'_256_256_256_',VoxelSize,'um_new_view_',view,'_(',initialAngle,',',endAngle,')_TV_',TV,'_L1_0.00_Lip_',Lip,'.recon'],'rb');

% Read Saved results
% fp = fopen([directory,Object,'_256_256_256_',VoxelSize,'um_iterative_',iter,'_view_',view,'_(',initialAngle,',',endAngle,')_TV_',TV,'_L1_0.00_Lip_',Lip,'.recon'],'rb');

f_volume = fread(fp, 256*256*256,'float');
f_volume = reshape(f_volume, [256 256 256]);
fclose(fp);

%%

GIFname = 'Test.gif';
% sz=get(0,'screensize');
% figure('outerposition',sz);
figure;

% figure1 = figure('Color',[1 1 1]);

for slice = 128
    
% f_slice = squeeze(f_volume(slice,:,:));     % Extract Vertical Slice: look from 180 direction
f_slice = squeeze(f_volume(:,slice,:));   % Extract Vertical Slice: look from 90 direction
% f_slice = f_volume(:,:,slice);    % Extract Horizontal Slice

A=f_slice;

A=A';

imagesc(A);
colormap(gray);
colorbar;
caxis([0 0.7]);

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
subplot(2,2,1); plot(obj_fun(1,:));
subplot(2,2,2); plot(obj_fun(2,:));
subplot(2,2,3); plot(obj_fun(3,:));