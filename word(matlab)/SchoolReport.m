% SchoolReport.m
% n=学生人数
% 为使得程序可读性加强，我们需要格式化成绩单
% 1. 第一行格式：姓名，各个学科，总分
% 2. 只有学生成绩，没有其他很杂的信息。
% 3. 部分成绩若空缺，请不要用数字填充，可以空着或者文字(缺考)填充。

%% 数据导入
filename='成绩.xlsx';
savename='成绩单';
[score,textdata]=xlsread(filename);
m=3; %学科数
n=size(score,1);% 学生人数
name=textdata(2:n+1,1);
mean_score=mean(score,'omitnan');
max_score=max(score,[],'omitnan');
grade=Gradegen(score); %Gradegen是自定义函数，生成各个同学的等级

%模板，实考对应变量score，等级对应变量grade
template=cell(5,m+1);
template(1,1:m+1)={'姓名','语文','数学','总分'};
template(2:5,1)={'实考成绩';'平均成绩';'最高成绩';'等级'};

%% 相关接口
import mlreportgen.dom.*;
d = Document(savename,'docx',fullfile(pwd,'mytemplate'));
open(d);

%% 开始生成word成绩单列表

% 添加分割线
hr = HorizontalRule();
hr.Style={Border('dotdash','blue','2px')};
append(d,hr);
% n是学生人数，m是学科数
for i=1:n
    % 填充表格
    template{1,1}=name{i};
    for jj=2:m+1
        s=score(i,jj-1);
        if isnan(s)
            s='缺考';
        else
            s=num2str(s);
        end
        template{2,jj}=s;% 具体分数
        template{3,jj}=sprintf('%4.2f',mean_score(jj-1));%学科平均分数
        template{4,jj}=num2str(max_score(jj-1));%学科最高分数
        template{5,jj}=grade{i,jj-1};% 等级
    end

    q=Table(template,'mytable2'); 
    q.TableEntriesStyle={Width('100'),Height('50')};
    append(d,q);
    append(d,clone(hr));
end
close(d);
rptview(d.OutputPath);