%%
clc;

view='220';
VoxelSize='100';
initialAngle='0';
endAngle='219';
TV='0.00';
Lip='32.00';
iter='10';

directory='Recon_Phantom_512/';
Object='SLPhantom2';

% Read latest results
fp = fopen([directory,Object,'_512_512_512_',VoxelSize,'um_new_view_',view,'_(',initialAngle,',',endAngle,')_TV_',TV,'_L1_0.00_Lip_',Lip,'.recon'],'rb');

% Read Saved results
% fp = fopen([directory,Object,'_512_512_512_',VoxelSize,'um_iterative_',iter,'_view_',view,'_(',initialAngle,',',endAngle,')_TV_',TV,'_L1_0.00_Lip_',Lip,'.recon'],'rb');

f_volume = fread(fp, 512*512*512,'float');
f_volume = reshape(f_volume, [512 512 512]);
fclose(fp);

%%

GIFname = 'Test.gif';
% sz=get(0,'screensize');
% figure('outerposition',sz);
figure;

% figure1 = figure('Color',[1 1 1]);

for slice = 256
    
% f_slice = squeeze(f_volume(slice,:,:));     % Extract Vertical Slice: look from 180 direction
% f_slice = squeeze(f_volume(:,slice,:));   % Extract Vertical Slice: look from 90 direction
f_slice = f_volume(:,:,slice);    % Extract Horizontal Slice

A=f_slice;

A=A';

imagesc(A);
colormap(gray);
colorbar;
% caxis([0 0.7]);6

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

%%

fp=fopen('projCheckv21.data','rb');
A=fread(fp,1024*1024,'float');
A = reshape(A, [1024 1024])/2;
fclose(fp);

fp=fopen('projCheckv30.data','rb');
B=fread(fp,1024*1024,'float');
B = reshape(B, [1024 1024]);
fclose(fp);

C=B-A;
figure;imagesc(B);
caxis([0,0.06]);