function [fhandle, resultImg] = showMatrixColumns(U,blockCols,imHeight,imWidth)

% Written by Filippo Pompili
% (filippopompili.09@gmail.com)
%

pixelBorder = 1;

U(U<0) = 0;
Uinvmax = 1./max(U,[],1);
U = bsxfun(@times,U,Uinvmax); U(isinf(U)) = 0;
nImgs = size(U,2);
blockRows = ceil(nImgs/blockCols);
nVertLines = blockCols + 1;
nHorLines = blockRows + 2;
resultImg = zeros(blockRows*imHeight + pixelBorder*nHorLines,blockCols*imWidth + pixelBorder*nVertLines);
n = 1;
i = 1;
hasFinished = 0;
while i <= blockRows && ~hasFinished
    j = 1;
    yoffset = 1 + pixelBorder*i + imHeight*(i-1);
    while j <= blockCols && ~hasFinished                        
        blockImg = reshape(U(:,n),imHeight,imWidth);    
        xoffset = 1 + pixelBorder*j + imWidth*(j-1);        
        resultImg(yoffset:yoffset+imHeight-1,xoffset:xoffset+imWidth-1) = blockImg;
        resultImg(1:end,xoffset-pixelBorder:xoffset-1) = ones(size(resultImg,1),pixelBorder);
        n = n + 1;
        j = j + 1;
        if n > nImgs
            hasFinished = 1;
        end
    end
    resultImg(yoffset-pixelBorder:yoffset-1,1:end) = ones(pixelBorder,size(resultImg,2));
    i = i + 1;
end
resultImg(1:end,end-pixelBorder+1:end) = ones(size(resultImg,1),pixelBorder);
resultImg(end-pixelBorder+1:end,1:end) = ones(pixelBorder,size(resultImg,2));

resultImg = 1-resultImg;
fhandle = figure;
imshow(resultImg,'InitialMagnification','fit'); 
colormap gray;
