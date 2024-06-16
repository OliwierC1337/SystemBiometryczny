clear all 
close all 
load ('iris.mat');
distance = adativeHammingDistance(binary_vec,binary_vec1);
charList = {};
bad_eye_recogn = [91,90,89,88,82,81,80,79,78,65,63,60,48,42,41,38,37,33,32,31,27,17,13]

charList{end+1} = 'Pierwszy ciąg znaków';
charList{end+1} = 'Drugi ciąg znaków';
folder_path = 'C:\Users\Oliwier\Desktop\263629\Obrazy';
j=0;
g=1;
f=[]
for i=1:100
    
    if(j==5)
    g=g+1
    j=0;
    end
    f(end+1)=g;
    j=j+1
end



f=f'
goodnbad=zeros(1,100);
goodnbad(bad_eye_recogn)=1;
load("binary_code.mat")
tablica = table(binary_codes',f,goodnbad', 'VariableNames', {'binary_codes', 'class', 'quality'})
same_class=[];
diff_class=[];
bad_eye_recogn=0;

CA=0
CR=0
FA=0
FR=0

Best_CA=0;
Best_FA=999;
thresholds = 0.3:0.01:0.5;
FRR = zeros(size(thresholds));
FAR = zeros(size(thresholds));

for g= 1:length(thresholds)
    thresh=thresholds(g)
for i = 1:100
    for j = 1:100
        if tablica.quality(i) == 0 && tablica.quality(j) == 0 && i ~= j  
            hamming_dist = adativeHammingDistance(tablica.binary_codes{i},tablica.binary_codes{j})

            if hamming_dist < 0.52
          
            if hamming_dist > thresh
                if tablica.class(i) == tablica.class(j)
                    FR=FR+1;
                else
                    CR=CR+1
                end
            else
                if tablica.class(i) == tablica.class(j)
                    CA=CA+1;
                else
                    FA=FA+1
                end

            end
            if(tablica.class(i) == tablica.class(j))
            same_class(end+1)=hamming_dist;
            else
                diff_class(end+1)=hamming_dist;
            end
            else
                bad_eye_recogn=bad_eye_recogn+1
            end
            
        end
    end
end
if (Best_CA < CA && Best_FA >= FA)
    Best_CA = CA;
    Best_FA = FA;
    Best_FR = FR;
    Best_CR = CR;

    best_thresh=thresholds(g)
end
FAR(end+1)=FA/(FA+CR)
FRR(end+1)=FR/(FR+CA)
FA=0
FR=0
CR=0
CA=0
end
figure;
    plot(FAR, FRR, '-o');
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', 'log');
    xlabel('FAR');
    ylabel('FRR');
    title('DET - Detection Error Trade-off')
good=mean(same_class)
bad = mean(diff_class)
good_var=var(same_class)
bad_var=var(diff_class)
x = linspace(0, 1, 1000);
good_gauss = normpdf(x, good, sqrt(good_var));
bad_gauss= normpdf(x, bad, sqrt(bad_var));
figure;
    hold on;
    plot(x, good_gauss, 'b', 'LineWidth', 2);
    plot(x, bad_gauss, 'black', 'LineWidth', 2);
    fill([x fliplr(x)], [good_gauss zeros(size(good_gauss))], 'r', 'FaceAlpha', 0.3);
    fill([x fliplr(x)], [bad_gauss zeros(size(bad_gauss))], 'g', 'FaceAlpha', 0.3);
    xlabel('Odległość');
    ylabel('Prawdopodobieństwo');
    legend('P_{u}(x) (autentics)', 'P_{o}(x) (impostors)');
    title('Gaussowskie funkcje gęstości prawdopodobieństwa');
    grid on;
    hold off;
disp(['CR=', num2str(CR)]);
disp(['CA=', num2str(CA)]);
disp(['FA=', num2str(FA)]);
disp(['FR=', num2str(FR)]);
disp(['Mean Distance Same Class=', num2str(good)]);
disp(['Mean Distance Diffrent Class=', num2str(bad)]);
disp(['Bad eye recognition=', num2str(bad_eye_recogn)]);
disp(['Best threshold=', num2str(best_thresh)]);
disp(['Best FA=', num2str(Best_FA)]);
disp(['Best CA=', num2str(Best_CA)]);
disp(['Best CR=', num2str(Best_CR)]);
disp(['Best FR=', num2str(Best_FR)]);


