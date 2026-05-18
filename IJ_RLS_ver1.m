%% ============================================================
% 최종 분석 코드 ? Figure A, Figure B, Table X
% (독립 실행 가능 ? 기존 코드 변수 불필요)
%% ============================================================

%% 0) Paths
dir_overch_chg   = 'D:\내 과제\한국전자기술연구원\시험 데이터\과충전팩 100cycle\로거\100사이클 충전';
dir_overch_dchg  = 'D:\내 과제\한국전자기술연구원\시험 데이터\과충전팩 100cycle\로거\100사이클 방전';
dir_overdis_chg  = 'D:\내 과제\한국전자기술연구원\시험 데이터\과방전팩 100cycle\로거\100사이클 충전';
dir_overdis_dchg = 'D:\내 과제\한국전자기술연구원\시험 데이터\과방전팩 100cycle\로거\100사이클 방전';

cycle_candidates = {'Cycle_002.csv','Cycle_02.csv','Cycle_2.csv','Cycle_0002.csv'};
file_oc_chg=''; file_oc_dchg=''; file_od_chg=''; file_od_dchg='';
for i=1:numel(cycle_candidates)
    if isempty(file_oc_chg)  && exist(fullfile(dir_overch_chg,  cycle_candidates{i}),'file'), file_oc_chg  = cycle_candidates{i}; end
    if isempty(file_oc_dchg) && exist(fullfile(dir_overch_dchg, cycle_candidates{i}),'file'), file_oc_dchg = cycle_candidates{i}; end
    if isempty(file_od_chg)  && exist(fullfile(dir_overdis_chg,  cycle_candidates{i}),'file'), file_od_chg  = cycle_candidates{i}; end
    if isempty(file_od_dchg) && exist(fullfile(dir_overdis_dchg, cycle_candidates{i}),'file'), file_od_dchg = cycle_candidates{i}; end
end

%% 1) LUT
cd('D:\국내 및 국제 저널\matlab\RLS');
load('OCV.mat','OCV'); load('Ri.mat','Ri');
load('Rdiff.mat','Rdiff'); load('Cdiff.mat','Cdiff');
OCV=OCV(:); Ri=Ri(:); Rdiff=Rdiff(:); Cdiff=Cdiff(:);
SOC_LUT=(1:-0.05:0).';
OCV_lookup_func   = @(soc) interp1(SOC_LUT,OCV,  soc,'linear','extrap');
Ri_lookup_func    = @(soc) interp1(SOC_LUT,Ri,   soc,'linear','extrap');
Rdiff_lookup_func = @(soc) interp1(SOC_LUT,Rdiff,soc,'linear','extrap');
Cdiff_lookup_func = @(soc) interp1(SOC_LUT,Cdiff,soc,'linear','extrap');
SOC_from_V        = @(V)   max(0,min(1,interp1(OCV,SOC_LUT,V,'linear','extrap')));

%% 2) Settings
Samplingtime = 1; Capacity = 3.65;
P0 = eye(3); P_clip = 1e-2; sign_mode = +1;
normal_cells = [2 3 4 5]; abnormal_cells = [1 6]; warmup_plot = 100;

lambda_list = [0.9, 0.99, 0.999];
lam_labels  = {'\lambda=0.90','\lambda=0.99','\lambda=0.999'};
lam_colors3 = {'r','b','k'};
n_lam = numel(lambda_list);

%% 3) 데이터 로드
raw_oc_chg  = xlsread(fullfile(dir_overch_chg,  file_oc_chg));
raw_oc_dchg = xlsread(fullfile(dir_overch_dchg, file_oc_dchg));
raw_od_chg  = xlsread(fullfile(dir_overdis_chg,  file_od_chg));
raw_od_dchg = xlsread(fullfile(dir_overdis_dchg, file_od_dchg));

V_oc_chg=raw_oc_chg(:,1:6);  V_oc_dchg=raw_oc_dchg(:,1:6);
V_od_chg=raw_od_chg(:,1:6);  V_od_dchg=raw_od_dchg(:,1:6);

if size(raw_oc_chg,2)>=8,  I_oc_chg=raw_oc_chg(:,8)/2;  else, I_oc_chg=raw_oc_chg(:,7)/2;  end
if size(raw_oc_dchg,2)>=8, I_oc_dchg=raw_oc_dchg(:,8)/2; else, I_oc_dchg=raw_oc_dchg(:,7)/2; end
if size(raw_od_chg,2)>=8,  I_od_chg=raw_od_chg(:,8)/2;  else, I_od_chg=raw_od_chg(:,7)/2;  end
if size(raw_od_dchg,2)>=8, I_od_dchg=raw_od_dchg(:,8)/2; else, I_od_dchg=raw_od_dchg(:,7)/2; end

%% 4) λ별 RLS 실행
ds_V   = {V_oc_chg,  V_oc_dchg, V_od_chg,  V_od_dchg};
ds_I   = {I_oc_chg,  I_oc_dchg, I_od_chg,  I_od_dchg};
ds_tag = {'oc_chg','oc_dchg','od_chg','od_dchg'};
n_ds   = 4;

% 결과 저장: Ri_res{ds_idx, li} = [N×6]
Ri_res=cell(n_ds,n_lam); Rdiff_res=cell(n_ds,n_lam);
Cdiff_res=cell(n_ds,n_lam); Vhat_res=cell(n_ds,n_lam);

for li=1:n_lam
    lam=lambda_list(li);
    for di=1:n_ds
        V_mat=ds_V{di}; I_vec=ds_I{di};
        Nds=size(V_mat,1);
        Ri_buf=zeros(Nds,6); Rdiff_buf=zeros(Nds,6);
        Cdiff_buf=zeros(Nds,6); Vhat_buf=zeros(Nds,6);
        for c=1:6
            V=V_mat(:,c); I=I_vec;
            soc0=SOC_from_V(V(1));
            SOC_ref=zeros(Nds,1); SOC_ref(1)=soc0;
            for k=2:Nds
                SOC_ref(k)=max(0,min(1,SOC_ref(k-1)+I(k)*Samplingtime/(3600*Capacity)));
            end
            OCV_ref=arrayfun(OCV_lookup_func,SOC_ref);
            Ri_0=Ri_lookup_func(soc0);
            Rdiff_0=Rdiff_lookup_func(soc0);
            Cdiff_0=Cdiff_lookup_func(soc0);
            dt=Samplingtime;
            PV=zeros(3,Nds);
            PV(:,1)=[Ri_0;(-Ri_0+dt/Cdiff_0+dt*Ri_0/(Rdiff_0*Cdiff_0));(1-dt/(Rdiff_0*Cdiff_0))];
            Pecov=zeros(3,3,Nds); Pecov(:,:,1)=P0;
            Ri_RLS=zeros(Nds,1); Ri_RLS(1)=Ri_0;
            Rdiff_RLS=zeros(Nds,1); Rdiff_RLS(1)=Rdiff_0;
            Cdiff_RLS=zeros(Nds,1); Cdiff_RLS(1)=Cdiff_0;
            for k=2:Nds
                phi=[I(k);I(k-1);V(k-1)-OCV_ref(k-1)];
                P_prev=Pecov(:,:,k-1);
                P_prev=max(0,min(P_clip,P_prev));
                G=(P_prev*phi)/(lam+(phi.'*P_prev*phi));
                Pn=(P_prev-G*phi.'*P_prev)/lam;
                Pn=max(0,min(P_clip,Pn));
                Pecov(:,:,k)=Pn;
                e=(V(k)-OCV_ref(k))-PV(:,k-1).'*phi;
                PV(:,k)=abs(PV(:,k-1)+G*e);
                b0=PV(1,k); b1=PV(2,k); a1=PV(3,k);
                Ri_RLS(k)   =abs(b0);
                Rdiff_RLS(k)=abs((b1-a1*b0)/(1+a1));
                Cdiff_RLS(k)=abs(dt/(b1-a1*b0));
            end
            Ri_buf(:,c)=Ri_RLS; Rdiff_buf(:,c)=Rdiff_RLS; Cdiff_buf(:,c)=Cdiff_RLS;
            Vds_st=zeros(Nds,1); Vhat=zeros(Nds,1);
            Vhat(1)=OCV_ref(1)+sign_mode*(I(1)*Ri_RLS(1)+Vds_st(1));
            for k=2:Nds
                tau=max(Rdiff_RLS(k)*Cdiff_RLS(k),1e-4);
                ak=exp(-dt/tau);
                Vds_st(k)=ak*Vds_st(k-1)+(1-ak)*I(k)*Rdiff_RLS(k);
                Vhat(k)=OCV_ref(k)+sign_mode*(I(k)*Ri_RLS(k)+Vds_st(k));
            end
            Vhat_buf(:,c)=Vhat;
        end
        Ri_res{di,li}=Ri_buf; Rdiff_res{di,li}=Rdiff_buf;
        Cdiff_res{di,li}=Cdiff_buf; Vhat_res{di,li}=Vhat_buf;
        fprintf('λ=%.3f  ds=%s  완료\n', lam, ds_tag{di});
    end
end

%% 5) 공통 설정
get_mean = @(X,N,cells) mean(X(1:N,cells),2,'omitnan');
k0=max(1,warmup_plot);

N_c=min(size(V_oc_chg,1),size(V_od_chg,1));
N_d=min(size(V_oc_dchg,1),size(V_od_dchg,1));
idxc=k0:N_c; tc=(idxc-k0)';
idxd=k0:N_d; td=(idxd-k0)';

% ds 인덱스 정의
% 1=oc_chg, 2=oc_dchg, 3=od_chg, 4=od_dchg
col_titles    = {'Normal (cells 2-5)','Overcharge (cells 1,6)','Overdischarge (cells 1,6)'};
param_ylabels = {'R_i [\Omega]','R_{diff} [\Omega]','C_{diff} [F]'};

% 카테고리별 ds 인덱스와 cell 정의
% col1=Normal(oc_chg 셀2-5), col2=OC(oc_chg 셀1,6), col3=OD(od_chg 셀1,6)
chg_ds_col  = {1, 1, 3};  % ds 인덱스
dchg_ds_col = {2, 2, 4};
cell_col    = {normal_cells, abnormal_cells, abnormal_cells};

%% ============================================================
% Figure A-1: CHARGE 3×3 파라미터 거동
%% ============================================================
figure('Color','w','Name','Figure A1 ? Charging Parameter Estimation','Position',[50 50 1400 900]);
for row=1:3
    for col=1:3
        subplot(3,3,(row-1)*3+col); hold on; grid on;
        di=chg_ds_col{col}; cells=cell_col{col};
        for li=1:n_lam
            switch row
                case 1; dat=get_mean(Ri_res{di,li},   N_c, cells);
                case 2; dat=get_mean(Rdiff_res{di,li}, N_c, cells);
                case 3; dat=get_mean(Cdiff_res{di,li}, N_c, cells);
            end
            plot(tc, dat(idxc), 'Color',lam_colors3{li}, 'LineWidth',2);
        end
        if row==1; title(col_titles{col},'FontName','Times New Roman','FontSize',12,'FontWeight','bold','Interpreter','none'); end
        if col==1; ylabel(param_ylabels{row},'FontName','Times New Roman','FontSize',12,'FontWeight','bold'); end
        if row==3; xlabel('Time [s]','FontName','Times New Roman','FontSize',11); end
        if row==1 && col==3; legend(lam_labels,'Location','best','FontSize',10); end
        set(gca,'FontName','Times New Roman','FontSize',11,'FontWeight','bold','Color','w');
    end
end
sgtitle('Charging ? Parameter Estimation by \lambda',...
    'FontName','Times New Roman','FontSize',15,'FontWeight','bold');

%% ============================================================
% Figure A-2: DISCHARGE 3×3 파라미터 거동
%% ============================================================
figure('Color','w','Name','Figure A2 ? Discharging Parameter Estimation','Position',[100 100 1400 900]);
for row=1:3
    for col=1:3
        subplot(3,3,(row-1)*3+col); hold on; grid on;
        di=dchg_ds_col{col}; cells=cell_col{col};
        for li=1:n_lam
            switch row
                case 1; dat=get_mean(Ri_res{di,li},   N_d, cells);
                case 2; dat=get_mean(Rdiff_res{di,li}, N_d, cells);
                case 3; dat=get_mean(Cdiff_res{di,li}, N_d, cells);
            end
            plot(td, dat(idxd), 'Color',lam_colors3{li}, 'LineWidth',2);
        end
        if row==1; title(col_titles{col},'FontName','Times New Roman','FontSize',12,'FontWeight','bold','Interpreter','none'); end
        if col==1; ylabel(param_ylabels{row},'FontName','Times New Roman','FontSize',12,'FontWeight','bold'); end
        if row==3; xlabel('Time [s]','FontName','Times New Roman','FontSize',11); end
        if row==1 && col==3; legend(lam_labels,'Location','best','FontSize',10); end
        set(gca,'FontName','Times New Roman','FontSize',11,'FontWeight','bold','Color','w');
    end
end
sgtitle('Discharging ? Parameter Estimation by \lambda',...
    'FontName','Times New Roman','FontSize',15,'FontWeight','bold');

%% ============================================================
% Figure B: RMSE bar chart
%% ============================================================
cat_names3={'Normal','Overcharge','Overdischarge'};
RMSE_chg=zeros(3,n_lam); RMSE_dchg=zeros(3,n_lam);
R2_chg=zeros(3,n_lam);   R2_dchg=zeros(3,n_lam);

VmN_c  =mean(V_oc_chg(1:N_c,normal_cells),  2,'omitnan');
VmOC_c =mean(V_oc_chg(1:N_c,abnormal_cells), 2,'omitnan');
VmOD_c =mean(V_od_chg(1:N_c,abnormal_cells), 2,'omitnan');
VmN_d  =mean(V_oc_dchg(1:N_d,normal_cells),  2,'omitnan');
VmOC_d =mean(V_oc_dchg(1:N_d,abnormal_cells), 2,'omitnan');
VmOD_d =mean(V_od_dchg(1:N_d,abnormal_cells), 2,'omitnan');

for li=1:n_lam
    VhN_c  =mean(Vhat_res{1,li}(1:N_c,normal_cells)+0.2,  2,'omitnan');
    VhOC_c =mean(Vhat_res{1,li}(1:N_c,abnormal_cells)+0.2, 2,'omitnan');
    VhOD_c =mean(Vhat_res{3,li}(1:N_c,abnormal_cells)+0.2, 2,'omitnan');
    VhN_d  =mean(Vhat_res{2,li}(1:N_d,normal_cells)-0.2,  2,'omitnan');
    VhOC_d =mean(Vhat_res{2,li}(1:N_d,abnormal_cells)-0.2, 2,'omitnan');
    VhOD_d =mean(Vhat_res{4,li}(1:N_d,abnormal_cells)-0.2, 2,'omitnan');

    Vms_c={VmN_c,VmOC_c,VmOD_c}; Vhs_c={VhN_c,VhOC_c,VhOD_c};
    Vms_d={VmN_d,VmOC_d,VmOD_d}; Vhs_d={VhN_d,VhOC_d,VhOD_d};

    for gi=1:3
        ec=Vms_c{gi}(idxc)-Vhs_c{gi}(idxc);
        ed=Vms_d{gi}(idxd)-Vhs_d{gi}(idxd);
        bc=mean(abs(Vms_c{gi}(idxc))); bd=mean(abs(Vms_d{gi}(idxd)));
        RMSE_chg(gi,li)=sqrt(mean(ec.^2))/bc*100;
        RMSE_dchg(gi,li)=sqrt(mean(ed.^2))/bd*100;
        SSc=sum((Vms_c{gi}(idxc)-mean(Vms_c{gi}(idxc))).^2);
        SSd=sum((Vms_d{gi}(idxd)-mean(Vms_d{gi}(idxd))).^2);
        R2_chg(gi,li) =1-sum(ec.^2)/max(SSc,eps);
        R2_dchg(gi,li)=1-sum(ed.^2)/max(SSd,eps);
    end
end

bar_c=[0.8 0.2 0.2; 0.2 0.4 0.8; 0.1 0.1 0.1];
all_rmse=[RMSE_chg(:);RMSE_dchg(:)];
ymax=max(all_rmse)*1.15;

figure('Color','w','Name','Figure B ? RMSE by lambda','Position',[150 150 1200 500]);
subplot(1,2,1); hold on; grid on;
b1=bar(RMSE_chg','grouped');
for li=1:n_lam; b1(li).FaceColor=bar_c(li,:); end
set(gca,'XTickLabel',cat_names3,'FontName','Times New Roman','FontSize',13,'FontWeight','bold','Color','w');
ylabel('RMSE [%]'); title('Charging');
legend(lam_labels,'Location','best'); ylim([0 ymax]);

subplot(1,2,2); hold on; grid on;
b2=bar(RMSE_dchg','grouped');
for li=1:n_lam; b2(li).FaceColor=bar_c(li,:); end
set(gca,'XTickLabel',cat_names3,'FontName','Times New Roman','FontSize',13,'FontWeight','bold','Color','w');
ylabel('RMSE [%]'); title('Discharging');
legend(lam_labels,'Location','best'); ylim([0 ymax]);

sgtitle('Figure B ? Effect of \lambda on Voltage Estimation RMSE',...
    'FontName','Times New Roman','FontSize',14,'FontWeight','bold');

%% ============================================================
% Table X: λ 선택 기준 비교
%% ============================================================
Ri_90  = get_mean(Ri_res{2,1}, N_d, normal_cells);
Ri_99  = get_mean(Ri_res{2,2}, N_d, normal_cells);
Ri_999 = get_mean(Ri_res{2,3}, N_d, normal_cells);
late_idx=501:N_d;
dr_90  = mean(abs(diff(Ri_90(late_idx))), 'omitnan');
dr_99  = mean(abs(diff(Ri_99(late_idx))), 'omitnan');
dr_999 = mean(abs(diff(Ri_999(late_idx))),'omitnan');

rmse_chg_avg  = mean(RMSE_chg,  1);
rmse_dchg_avg = mean(RMSE_dchg, 1);
r2_chg_avg    = mean(R2_chg,    1);
r2_dchg_avg   = mean(R2_dchg,   1);

fprintf('\n');
fprintf('============================================================\n');
fprintf('Table X. Forgetting factor selection criteria\n');
fprintf('============================================================\n');
fprintf('%-28s  %12s  %12s  %12s\n','Criterion','lambda=0.90','lambda=0.99','lambda=0.999');
fprintf('%s\n',repmat('-',1,68));
fprintf('%-28s  %12.0f  %12.0f  %12.0f\n','Effective memory N_eff',...
    1/(1-lambda_list(1)),1/(1-lambda_list(2)),1/(1-lambda_list(3)));
fprintf('%-28s  %12.2e  %12.2e  %12.2e\n','Late-stage dRi/dt [Ohm/s]',dr_90,dr_99,dr_999);
fprintf('%-28s  %12.3f  %12.3f  %12.3f\n','Avg RMSE-CHG [%%]',...
    rmse_chg_avg(1),rmse_chg_avg(2),rmse_chg_avg(3));
fprintf('%-28s  %12.3f  %12.3f  %12.3f\n','Avg RMSE-DCHG [%%]',...
    rmse_dchg_avg(1),rmse_dchg_avg(2),rmse_dchg_avg(3));
fprintf('%-28s  %12.4f  %12.4f  %12.4f\n','Avg R2-CHG',...
    r2_chg_avg(1),r2_chg_avg(2),r2_chg_avg(3));
fprintf('%-28s  %12.4f  %12.4f  %12.4f\n','Avg R2-DCHG',...
    r2_dchg_avg(1),r2_dchg_avg(2),r2_dchg_avg(3));
fprintf('%-28s  %12s  %12s  %12s\n','Noise sensitivity','High','Medium','Low');
fprintf('%-28s  %12s  %12s  %12s\n','Tracking speed','Fast','Medium','Slow');
fprintf('%-28s  %12s  %12s  %12s\n','Selected','No','YES','No');
fprintf('%s\n',repmat('-',1,68));
fprintf('* lambda=0.99 is %.1fx faster than lambda=0.999\n', dr_99/dr_999);
fprintf('* lambda=0.99 is %.1fx more stable than lambda=0.90\n', dr_90/dr_99);
fprintf('============================================================\n');