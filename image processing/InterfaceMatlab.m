function varargout = InterfaceMatlab(varargin)
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @InterfaceMatlab_OpeningFcn, ...
                   'gui_OutputFcn',  @InterfaceMatlab_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before InterfaceMatlab is made visible.
function InterfaceMatlab_OpeningFcn(hObject, ~, handles, varargin)
% Choose default command line output for InterfaceMatlab
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = InterfaceMatlab_OutputFcn(~, ~, handles) 

varargout{1} = handles.output;

%#####################################################################
%################      GESTION DES FICHIERS         ##################
%#####################################################################
% --------------------------------------------------------------------
function Ouvrir_Callback(hObject, ~, handles)

[file,path] = uigetfile('*.*');
handles.ima = imread(sprintf('%s',path,file));
axes(handles.imgO)
handles.courant_data = handles.ima;
imshow(handles.courant_data);
title('Image Originale');

handles.ima_traite = 0;
axes(handles.imgT)
handles.output = hObject;
guidata(hObject, handles);

% --------------------------------------------------------------------
function Enregistrer_Callback(~, ~, handles)

image = handles.ima_traite;
[file,path] = uiputfile('*.png','Enregistrer Votre Image ...');
imwrite(image, sprintf('%s',path,file),'png');

% --------------------------------------------------------------------
function Quitter_Callback(~, ~, handles)

delete(handles.figure1)


%#####################################################################
%################      LES FILTRES PASSE BAS        ##################
%#####################################################################

% =======================     Median     =============================
function Median_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end
image_filtree = image;
[n,m]=size(image);
for i=2:n-1
    for j=2:m-1
       fenetre=image(i-1:i+1,j-1:j+1);
       image_filtree(i,j)=median(fenetre(:));
    end
end

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Median');

% ====================     Moyenneur 3x3     =========================     
function Moyenneur3x3_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

H=(1/9)*[1 1 1 ; 1 1 1 ; 1 1 1 ];

image_filtree = conv2(double(image), H, 'same');

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Moyenneur 3x3');

% ====================     Moyenneur 5x5     =========================     
function Moyenneur5x5_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

H=(1/25)*[1 1 1 1 1 ; 1 1 1 1 1 ; 1 1 1 1 1 ; 1 1 1 1 1 ; 1 1 1 1 1];

image_filtree = conv2(double(image), H, 'same');

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Moyenneur 5x5');

%======================     Gaussien 3x3     =========================     
function Gaussien3x3_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

H=(1/16)*[1 2 1 ;2 4 2 ; 1 2 1];

image_filtree = conv2(double(image), H, 'same');

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Gaussien 3x3');

%======================     Gaussien 5x5     =========================     
function Gaussien5x5_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

H=(1/256)*[1 4 6 4 1 ; 4 16 24 16 4 ; 6 24 36 24 6 ; 4 16 24 16 4 ; 1 4 6 4 1];

image_filtree = conv2(double(image), H, 'same');

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Gaussien 5x5');

%========================     Conique     ===========================
function Conique_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

H=(1/25)*[0 0 1 0 0 ; 0 2 2 2 0 ; 1 2 5 2 1 ; 0 2 2 2 0 ; 0 0 1 0 0];

image_filtree = conv2(double(image), H, 'same');

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Conique');

%======================     Piramidal     ==========================     
function Piramidal_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

H=(1/81)*[1 2 3 2 1 ; 2 4 6 4 2 ; 3 6 9 6 3 ; 2 4 6 4 2 ; 1 2 3 2 1];

image_filtree = conv2(double(image), H, 'same');

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Piramidal');

%#####################################################################
%################      LES FILTRES PASSE HAUT        #################
%#####################################################################

%=======================     Laplacien     ===========================
function Laplacien_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

%H=[0 1 0;1 -4 1;0 1 0];
H=[-1 -1 -1;-1 8 -1;-1 -1 -1];

image_filtree = conv2(double(image), H, 'same');

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Laplacien (8 conn)');

%=======================     Gradient     ===========================
function Gradient_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

Gx=[0,0,0;1,0,-1;0,0,0];
Gy=[0,1,0;0,0,0;0,-1,0];

gradient_x = conv2(double(image), Gx, 'same');
gradient_y = conv2(double(image), Gy, 'same');

image_filtree = sqrt(gradient_x.^2 + gradient_y.^2);

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Gradient');

%=======================     Sobel     ===========================
function Sobel_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

Gx=[-1,0,1;-2,0,2;-1,0,1];
Gy=[-1,-2,-1;0,0,0;1,2,1];

gradient_x = conv2(double(image), Gx, 'same');
gradient_y = conv2(double(image), Gy, 'same');

image_filtree = sqrt(gradient_x.^2 + gradient_y.^2);

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Sobel');

%=======================     Prewitt     ===========================
function Prewitt_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

Gx=[-1,0,1;-1,0,1;-1,0,1];
Gy=[-1,-1,-1;0,0,0;1,1,1];

gradient_x = conv2(double(image), Gx, 'same');
gradient_y = conv2(double(image), Gy, 'same');

image_filtree = sqrt(gradient_x.^2 + gradient_y.^2);

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Prewitt');

%=======================     Roberts     ===========================
function Roberts_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

Gx=[1 0; 0 -1];
Gy=[0 1; -1 0];

gradient_x = conv2(double(image), Gx, 'same');
gradient_y = conv2(double(image), Gy, 'same');

image_filtree = sqrt(gradient_x.^2 + gradient_y.^2);

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Roberts');

%=====================     Marr-Hildreth     =========================
function MarrHildreth_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

Gx=[0,0,0;1,0,-1;0,0,0];
Gy=[0,1,0;0,0,0;0,-1,0];
H=[-1 -1 -1;-1 8 -1;-1 -1 -1];

gradient_x = conv2(double(image), Gx, 'same');
gradient_y = conv2(double(image), Gy, 'same');

image_gradient = sqrt(gradient_x.^2 + gradient_y.^2);
image_filtree = conv2(double(image_gradient), H, 'same');

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Marr-Hildreth');

%#####################################################################
%###################      TRANSFORMATIONS        #####################
%#####################################################################

%=======================     Negative      ===========================
function Negatif_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

image_filtree = 255 - image;

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Negative');

%=======================     Contrast     ===========================
function Contraste_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

image_filtree = min(max(double(image-128) * 5 + 128, 0), 255);

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Contraste');

%=======================     Luminosité     ===========================
function Luminosite_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

image_filtree = min(max(double(image) + 50, 0), 255);
  

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Luminosité');

%======================     Binarisation     ========================
function Binarisation_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

image_filtree = image > 128;

axes(handles.imgT);
handles.ima_traite = image_filtree;
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Binarisation');

%======================     Niveau de gris     =======================
function NivGris_Callback(hObject, ~, handles)

image=handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

axes(handles.imgT);
handles.ima_traite = uint8(image);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Niveau de gris');

%=======================     Histogramme     =========================
function Histogramme_Callback(~, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

axes(handles.imgT);
bar(imhist(image));
title('Histogramme de l''Image');
xlabel('Niveau de gris');
ylabel('Fréquence');


%#####################################################################
%#########      LES FILTRES DANS LE DOMAIN FREQUENTIEL        ########
%#####################################################################

%====================     Passe-Bas Idéal     ========================
function pass_bas_ideal_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

F=fftshift(fft2(image));
[N,M]=size(F);
 
D0 = 10; % Rayon de la zone de fréquences basses
H0=zeros(N,M); 

Cx=round(N/2);  %centre horizontal 
Cy=round(M/2);  %centre vertical

H0(Cx-D0:Cx+D0,Cy-D0:Cy+D0)=1; 

F_filtre = F .* H0;

image_filtree = ifft2(ifftshift(F_filtre));
axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Passe-Bas Idéal');

%=================     Passe-Bas Butterworth     =====================
function pass_bas_butterworth_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

F=fftshift(fft2(image));
[N,M]=size(F);
 
 D0 = 10; % Rayon de la zone de fréquences basses
 H0=zeros(N,M); 

for u = 1:N
    for v = 1:M
        % Calculer la distance euclidienne par rapport au centre de la transformée de Fourier
        D_uv = sqrt((u - N/2)^2 + (v - M/2)^2);
        H0(u, v) = 1/(1 + (D_uv/D0).^2); % Filtre passe-bas Butterworth
    end
end 
F_filtre = F .* H0;

image_filtree = ifft2(ifftshift(F_filtre));

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Passe-Bas Butterworth');

%====================     Passe-Haut Idéal     ========================
function pass_haut_ideal_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

F=fftshift(fft2(image));
[N,M]=size(F);
 
D0 = 10; % Rayon de la zone de fréquences basses
H0=ones(N,M); 

Cx=round(N/2);  %centre horizontal 
Cy=round(M/2);  %centre vertical

H0(Cx-D0:Cx+D0,Cy-D0:Cy+D0)=0; 

F_filtre = F .* H0;

image_filtree = ifft2(ifftshift(F_filtre));
axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Passe-Haut Idéal');

%=================     Passe-Haut Butterworth     ====================
function pass_haut_butterworth_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

F=fftshift(fft2(image));
[N,M]=size(F);
 
 D0 = 10; % Rayon de la zone de fréquences basses
 H0=zeros(N,M); 

for u = 1:N
    for v = 1:M
        % Calculer la distance euclidienne par rapport au centre de la transformée de Fourier
        D_uv = sqrt((u - N/2)^2 + (v - M/2)^2);
        H0(u, v) = 1/(1 + (D0/D_uv).^2); % Filtre passe-Haut Butterworth
    end
end 
F_filtre = F .* H0;

image_filtree = ifft2(ifftshift(F_filtre));
axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Filtre Passe-Haut Butterworth');


%#####################################################################
%#######################      BRUIT        ###########################
%#####################################################################

%=======================     Gaussien     ============================
function Gaussien_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

image_filtree = imnoise(image, 'gaussian', 0, 0.01);

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Bruit Gaussien');

%=====================     Poivre Et Sel     =========================
function poivreetsel_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

image_filtree = imnoise(image, 'Salt & Pepper', 0.01);

axes(handles.imgT);
handles.ima_traite = uint8(image_filtree);
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Bruit Poivre Et Sel');

%#####################################################################
%################      LES POINTS D'INTERET        ###################
%#####################################################################

%=========================     SUSAN     =============================
function SUSAN_Callback(hObject, ~, handles)
image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end
image = double(image);
[n,m]=size(image);
% =============================données=================================
rayon=3;
alpha=80;
r=60;
alpha=alpha/100;
% ========================génerateur de mask===========================
mask = [0 0 1 1 1 0 0
        0 1 1 1 1 1 0
        1 1 1 1 1 1 1
        1 1 1 1 1 1 1
        1 1 1 1 1 1 1
        0 1 1 1 1 1 0
        0 0 1 1 1 0 0];
% =====================balayage de toute l'image=======================
f=zeros(n,m);
for i=(rayon+1):n-rayon
    for j=(rayon+1):m-rayon
        fenetre=image(i-rayon:i+rayon,j-rayon:j+rayon);
        fenetre=fenetre.*mask;
        Ipn= fenetre(rayon+1,rayon+1);
        CP=exp(-1*(((fenetre-Ipn).^6)/rayon));
        somme=sum(sum(CP));
% si le centre du mask est un 0 il faut soustraire les zeros des filtres
       if (Ipn==0)
        somme=somme-length((find(mask==0)));
       end
        f(i,j)=somme;
    end
end

% =============selection et seuillage des points d'interét=============
ff=f(rayon+1:n-(rayon+1),rayon+1:m-(rayon+1));
minf=min(min(ff));
maxf=max(max(ff));
fff=f;
d=2*r+1;
temp1 = round(n/d);
temp2 = round(m/d);
temp1 = temp1 - (temp1-n/d < 0.5 & temp1-n/d > 0);
temp2 = temp2 - (temp2-m/d < 0.5 & temp2-m/d > 0);

fff(n:temp1*d+d,m:temp2*d+d)=0;

for i=(r+1):d:temp1*d+d
   for j=(r+1):d:temp2*d+d
        window=fff(i-r:i+r,j-r:j+r);
        window0=window;
        [xx,yy]=find(window0==0);
        for k=1:length(xx)
            window0(xx(k),yy(k))=max(max(window0));
        end
        minwindow=min(min(window0));
        [y,x]=find(minwindow~=window & window<=minf+alpha*(maxf-minf) & window>0);
        [u,v]=find(minwindow==window);
    if length(u)>1
        for l=2:length(u)
            fff(i-r-1+u(l),j-r-1+v(l))=0 ;
        end
    end
   if ~isempty(x)
    for l=1:length(y)
        fff(i-r-1+y(l),j-r-1+x(l))=0 ;
    end
   end
   end
end
seuil=minf+alpha*(maxf-minf);
[u,v]=find(minf<=fff & fff<=seuil );

% ==============affichage des resultats================================
axes(handles.imgT);
handles.ima_traite = uint8(image);
imshow(handles.ima_traite);
hold on;
plot(u,v,'.g','MarkerSize',20);
hold off;
handles.output = hObject;
guidata(hObject, handles);
title('SUSAN');

%=========================     HARRIS     =============================
function Harris_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end
image = double(image);

dx=[-1 0 1; -2 0 2 ; -1 0 1]; % derivée horizontale : filtre de Sobel
dy=dx';

% Compute gradients using the Sobel operator
Ix=conv2(image,dx,'same');
Iy=conv2(image,dy,'same');

% Compute products of gradients
Ix2 = Ix.^2;
Iy2 = Iy.^2;
IxIy = Ix .* Iy;

% Define the size of the local window for computing the Harris response
w = 5; sigma = 1;

% Compute the sums of products in the local window using a Gaussian filter
G = fspecial('gaussian', w, sigma);
Sx2=conv2(Ix2, G, 'same');
Sy2=conv2(Iy2, G, 'same');
Sxy=conv2(IxIy, G,'same');

% Harris corner response function
k = 0.04; % Empirical constant
R = (Sx2 .* Sy2 - Sxy.^2) - k * (Sx2 + Sy2).^2;

% --------------- seuil ------------------
seuil = 0.01 * max(R(:));

% Find the corners
corners = R > seuil;

[x, y] = find(corners);

axes(handles.imgT);
handles.ima_traite = uint8(image);
imshow(handles.ima_traite);
hold on;
plot(y,x,'.r','MarkerSize',20);
hold off;
handles.output = hObject;
guidata(hObject, handles);
title('Harris');

%#####################################################################
%###############      MORPHOLOGIE MATHEMATIQUE       #################
%#####################################################################

%=========================     Erosion     =============================
function erosion_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

element_structurant = strel('square', 5);

image_filtree = imerode(image, element_structurant);

axes(handles.imgT);
handles.ima_traite = image_filtree;
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Erosion');

%========================     Dilatation   ============================
function dilatation_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

element_structurant = strel('square', 5);

image_filtree = imdilate(image, element_structurant);

axes(handles.imgT);
handles.ima_traite = image_filtree;
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Dilatation');

%========================     Ouverture     ===========================
function ouverture_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

element_structurant = strel('square', 5);

image_erode = imerode(image, element_structurant);

image_filtree = imdilate(image_erode, element_structurant);

axes(handles.imgT);
handles.ima_traite = image_filtree;
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Ouverture');

%========================     Fermeture     ==========================
function fermeture_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

element_structurant = strel('square', 5);

image_dilate = imdilate(image, element_structurant);

image_filtree = imerode(image_dilate, element_structurant);

axes(handles.imgT);
handles.ima_traite = image_filtree;
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Fermeture');


%#####################################################################
%####################      LES CONTOURS        #######################
%#####################################################################

%=====================     Contours internes    ======================
function C_interne_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

element_structurant = strel('square', 5);

image_erode = imerode(image, element_structurant);

image_filtree = image - image_erode;

axes(handles.imgT);
handles.ima_traite = image_filtree;
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Contour Interne');


%=====================     Contours Externes    ======================
function C_externe_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

element_structurant = strel('square', 5);

image_dilate = imdilate(image, element_structurant);

image_filtree = image_dilate - image;

axes(handles.imgT);
handles.ima_traite = image_filtree;
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Contour Externe');


%===================     Contours Morphologiques    ===================
function C_morphologie_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

element_structurant = strel('square', 5);

image_erode = imerode(image, element_structurant);
image_dilate = imdilate(image, element_structurant);

image_filtree = image_dilate - image_erode;

axes(handles.imgT);
handles.ima_traite = image_filtree;
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Contour Morphologique');


%#####################################################################
%#################      Fourier Transform      #######################
%#####################################################################
function Fourier_Callback(hObject, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

image_filtree = fftshift(fft2(image));
image_filtree = mat2gray(log(1 + abs(image_filtree)));

axes(handles.imgT);
handles.ima_traite = image_filtree;
imshow(handles.ima_traite);
handles.output = hObject;
guidata(hObject, handles);
title('Fourier Transform');


% #####################################################################
% ###########################   HOUGH   ###############################
% #####################################################################

% --------------------------- Hough Droites --------------------------
% --------------------------------------------------------------------
function Hough_droites_Callback(~, ~, handles)

image = handles.courant_data;
size(image)
if size(image, 3) == 3
    image = rgb2gray(image);
end

edges = edge(image, 'Roberts');

[H, theta, rho] = hough(edges);

peaks = houghpeaks(H, 100); % Vous pouvez ajuster le nombre de pics en fonction de votre image

lines = houghlines(edges, theta, rho, peaks);

axes(handles.imgT);
imshow(image);
hold on;

for i = 1:length(lines)
    xy = [lines(i).point1; lines(i).point2];
    plot(xy(:, 1), xy(:, 2), 'LineWidth', 2, 'Color', 'r');
end

hold off;

title('Hough Droites');


% --------------------------- Hough Cercles --------------------------
% --------------------------------------------------------------------
function Hough_cercles_Callback(~, ~, handles)

image = handles.courant_data;
if size(image, 3) == 3
    image = rgb2gray(image);
end

% Détection des bords avec un détecteur de contours (par exemple, Canny)
edges = edge(image, 'Prewitt');

% Paramètres
min_radius = 10;
max_radius = 100;

% Initialiser l'accumulateur Hough
[rows, cols] = size(edges);
H = zeros(rows, cols, max_radius - min_radius + 1);

% Boucle sur les rayons
for r = min_radius:max_radius
    % Pour chaque pixel de bord, calculer les coordonnées des cercles possibles
    [y, x] = find(edges);
    
    for i = 1:length(x)
        for t = linspace(0, 2 * pi, 100)
            a = round(x(i) - r * cos(t));
            b = round(y(i) - r * sin(t));
            
            if a > 0 && a <= cols && b > 0 && b <= rows
                H(b, a, r - min_radius + 1) = H(b, a, r - min_radius + 1) + 1;
            end
        end
    end
end

% Seuillage des pics dans l'espace Hough
threshold = 0.6 * max(H(:));
[cy, cx, radii] = ind2sub(size(H), find(H > threshold));

% Afficher les cercles détectés
axes(handles.imgT);
imshow(image);

hold on;
for i = 1:length(cx)
    viscircles([cx(i), cy(i)], radii(i) + min_radius - 1, 'EdgeColor', 'r');
end
hold off;

% Afficher les centres et les rayons
disp('Centres des cercles détectés :');
disp([cx, cy]);
disp('Rayons des cercles détectés :');
disp(radii + min_radius - 1);

title('Hough Cercles')