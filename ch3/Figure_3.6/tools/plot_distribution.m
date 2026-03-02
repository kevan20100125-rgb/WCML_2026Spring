
idx = True > 1e-3;
True = True(idx);
MHGD = MHGD(idx);
DMALA = DMALA(idx);

data = [True.' MHGD.' DMALA.'];
lnd = {'True', 'MHGD [21]', 'DMALA'};

% hatchfill2
figure()
hp = bar(data, 1.0);
ax = gca; %ax.XAxis.FontSize = 9; 
ymax=ax.YLim(end);
hatchfill2(hp(1), 'single','HatchAngle',45,'hatchcolor','m', 'HatchDensity', 75);
% hatchfill2(hp(2), 'single','HatchAngle',-45,'hatchcolor',color_green, 'HatchDensity', 50);
hatchfill2(hp(2), 'single','HatchAngle',0,'hatchcolor','b', 'HatchDensity', 75);  % add 
hatchfill2(hp(3),'single','HatchAngle',-45,'hatchcolor','r','HatchDensity', 75);  % mul
for b = 1:numel(hp)
    hp(b).FaceColor = 'none';
end
ylim([0 ymax])

set(gcf,'position',[329 192.2 588 400])
xlabel('State index')
ylabel('Probability')

[legend_h,object_h,plot_h,text_str] = legendflex(hp, lnd, 'FontSize',11, 'anchor', {'n', 'n'});
% object_h(1) is the first bar's text
% object_h(2) is the second bar's text
% object_h(3) is the third bar's text
% object_h(4) is the first bar's patch
% ...
hatchfill2(object_h(4), 'single','HatchAngle',45,'hatchcolor','m','HatchDensity',100/10);
hatchfill2(object_h(5), 'single','HatchAngle',0,'hatchcolor','b','HatchDensity',100/10);
hatchfill2(object_h(6), 'single','HatchAngle',-45,'hatchcolor','r','HatchDensity',100/10);