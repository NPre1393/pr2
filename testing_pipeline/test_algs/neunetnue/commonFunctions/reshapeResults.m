function reshapeResults(copyFolder,resultsFolder,autoPairwiseTarDriv,handPairwiseTarDriv,params)


methodNames      = fieldnames(params.methods);
numMethods       = length(methodNames);

cd(resultsFolder);
i = 1;
if strcmp('neunetnue',methodNames{i})
    allResults = dir([methodNames{i} '*.mat']);
    if (autoPairwiseTarDriv == 1 || handPairwiseTarDriv == 1)
        for t = 1:length(allResults)
            load(allResults(t).name);
            numTargets = size(outputToStore.params.infoSeries,1);
            A = reshape(outputToStore.transferEntropy,numTargets-1,numTargets);
            if find(isnan(A));
                [idRow,idCol] = find(isnan(A));
                for idR = idRow
                    for idC = idCol
                        A(idR,idC) = 0;
                    end
                end
            end
            reshapedMtx = zeros(numTargets);
            reshapedMtx(:,1) = [0;A(:,1)];
            for j = 2:numTargets-1
                reshapedMtx(:,j) = [A(1:j-1,j);0;A(j:end,j)];
            end
            reshapedMtx(:,numTargets) = [A(:,numTargets);0];
            outputToStore.reshapedMtx = reshapedMtx;
            save(allResults(t).name(1:end-4),'outputToStore');
        end
        meanResMtx = zeros(numTargets,numTargets);
        for t = 1:length(allResults)
            load(allResults(t).name);
            meanResMtx = meanResMtx + outputToStore.reshapedMtx;
        end
        meanRes = meanResMtx / length(allResults);
        save([methodNames{i} '_meanReshapeMtx'],'meanRes');
        cd(copyFolder);
        s = load([methodNames{i} '_significanceOnDriv']);
        significance = sum(s.significanceOnDrivers,1);
        B = reshape(significance,numTargets-1,numTargets);
        reshapedSigni = zeros(numTargets);
        reshapedSigni(:,1) = [0;B(:,1)];
        for j = 2:numTargets-1
            reshapedSigni(:,j) = [B(1:j-1,j);0;B(j:end,j)];
        end
        reshapedSigni(:,numTargets) = [B(:,numTargets);0];
        save([methodNames{i} '_reshapedSignificance'],'reshapedSigni');
        cd(resultsFolder);
        meanImg = imagesc(meanRes);
        alpha(reshapedSigni / length(allResults));
        colormap(jet);%colormap(1-gray);
        colorbar;
        set(gca,'FontName','Times New Roman','FontSize',34);
        title(['row influences column  |  ' methodNames{i}],'FontSize',16);
        titleFig = [methodNames{i} '_meanReshapedMtx'];
        %                 saveas(meanImg,titleFig,'png');
        
        cd(copyFolder)
        significance = load([methodNames{i} '_significanceOnDriv.mat']);
        te           = load([methodNames{i} '_transferEntropyMtx.mat']);
        realizations = size(significance.significanceOnDrivers,1);
        significance = sum(significance.significanceOnDrivers,1);
        te           = mean(te.matrixTransferEntropy(:,2:end),1);
        B            = reshape(te,numTargets-1,numTargets);
        reshapedTE   = zeros(numTargets);
        reshapedTE(:,1) = [0;B(:,1)];
        for k = 2:numTargets-1
            reshapedTE(:,k) = [B(1:k-1,k);0;B(k:end,k)];
        end
        reshapedTE(:,numTargets) = [B(:,numTargets);0];
        figure(99);
        ba=bar3(reshapedTE);
        for l=1:length(ba)
            set(ba(l),'cdata',...
                get(ba(l),'zdata'));
        end
        colormap(hot);
        
        
        
        C = reshape(significance,numTargets-1,numTargets);
        reshapedSignif = zeros(numTargets);
        reshapedSignif(:,1) = [0;C(:,1)];
        for k = 2:numTargets-1
            reshapedSignif(:,k) = [C(1:k-1,k);0;C(k:end,k)];
        end
        reshapedSignif(:,numTargets) = [C(:,numTargets);0];
        f = figure(1);
        bh=bar3(reshapedSignif);
        for l=1:length(bh)
            set(bh(l),'cdata',...
                get(ba(l),'zdata'));
        end
        colormap(hot);
        colorbar
        zlim([0 realizations]);
        title(methodNames{i},'FontName','Times New Roman','FontSize',44)
        clear bh
        set(gca,'FontName','Times New Roman','FontSize',42);
        set(f,'Position',[1 1 1680 1050]);
        cd(resultsFolder)
        %                 saveas(f,['bars_' methodNames{i}],'png');
    end
end


return;