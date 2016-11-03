% 利用模板中的样式来生成报告
% 如果是中文版Word，自带样式名称请参见目录下的pdf文档

import mlreportgen.dom.*
%Document.createTemplate('mytemplate.dotx','docx');
d=Document('demo3','docx',fullfile(pwd,'mytemplate'));
open(d);


%% 标题
p=Paragraph('成绩报告单','Heading 1');
append(d,p);

%% 文字段落(doc mlreportgen.dom.Paragraph)
append(d,Heading(2,'一、段落模板'));
s='这里是段落';
s=repmat(s,[1,30]);
p = Paragraph(s,'mypara1');%自定义段落样式
append(d,p);

%% 添加空的段落行
for i=1:8
    append(d,' ');
end

%% 插入表格
append(d,Heading(2,'二、表格模板'));
t={'姓名','语文','数学','英语'; ...
    '成绩','70','94','82'; ...
    '等级','C','A','B'}; 
p=Table(t,'mytable1');% 自定义表格样式
append(d,p);

close(d);
rptview(d.OutputPath);