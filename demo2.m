% demo2.m
import mlreportgen.dom.*;
d=Document('demo2','docx');
open(d);

%% 页面设置
s = d.CurrentDOCXSection;
s.PageSize.Orientation  ='landscape'; % portrait(default)
s.PageSize.Height = '8.5in';
s.PageSize.Width = '11in';
s.PageMargins.Left = '3.0cm';
s.PageMargins.Right = '3.0cm';
s.PageMargins.Top = '2.5cm';
s.PageMargins.Bottom = '2.5cm';
% 中文字体样式设置
heiti=FontFamily;
heiti.FamilyName='Arial';
heiti.EastAsiaFamilyName='黑体';
songti=FontFamily;
songti.FamilyName='Arial';
songti.EastAsiaFamilyName='宋体';

%% 标题
p=Heading(1,'Matlab 自动化报告模板');% 一级标题
%p.Color='red';
%p.HAlign='center';
p.Style={heiti,Color('red'),HAlign('center')};
append(d,p);

%% 段落
append(d,Heading(2,'一、段落模板'));
s='这里是段落。';
s=repmat(s,[1,12]);
p = Paragraph(s);
% 中文字体样式自定义
p.Style={songti,Color('blue'),...
    LineSpacing(1.5),...
    OuterMargin('10pt','0pt','0pt','10pt')};
p1=Text('下划线。');%同段落差不多.
p1.Underline='single';
p1.Color='red';
append(p,p1);
append(p,s);
p2=ExternalLink('http://github.com/gasongjian/', '超链接');
append(p,p2);
p.FontSize='14pt';
p.FirstLineIndent='28pt';%这里差不多就是2个字符大小
append(d,p);

%% 简易表格
append(d,Heading(2,'二、简单表格'));
t={'志明','语文','数学','英语'; ...
    '成绩','70','98','89'; ...
    '等级','B','A','A'};
p=Table(t);

% 格式化单元格中的段落
for ii=1:p.NRows
    for jj=1:p.NCols
        t=entry(p,ii,jj);
         t.Children(1).Style={songti,...
            Color('green'),...
            FontSize('12pt'),...
            LineSpacing(1.0),...
            OuterMargin('0pt','0pt','0pt','0pt')};
    end
end

p.Style = {Border('single','blue','3px'), ...
               ColSep('single','blue','1px'), ...
               RowSep('single','blue','1px')};

p.Width = '50%';
p.HAlign='center';% 居中对齐
p.TableEntriesHAlign='center';
p.TableEntriesVAlign='middle';
append(d,p);

%% 复杂表格
append(d,Heading(2,'三、复杂表格'));
q = Table(5);
q.Border = 'single';
q.ColSep = 'single';
q.RowSep = 'single';

row = TableRow;
te = TableEntry('算法名称');
te.RowSpan = 2;
append(row, te);

te = TableEntry('第一类');
te.ColSpan = 2;
%te.Border = 'single';
append(row, te);
te = TableEntry('第二类');
te.ColSpan = 2;
%te.Border = 'single';
append(row, te);
append(q,row);

% 第二行
row=TableRow;
append(row,TableEntry('T1'));
append(row,TableEntry('T2'));
append(row,TableEntry('T3'));
append(row,TableEntry('T4'));
append(q,row);

% 其他行
t=TableRow;
append(t,TableEntry('条目'));
for i=1:4
    append(t,TableEntry(''));
end
append(q,t);
append(q,clone(t));
append(q,clone(t));
append(q,clone(t));
q.TableEntriesStyle={Width('80'),Height('40')};
q.Style = {Border('single','green','3px'), ...
    ColSep('single','green','1px'), ...
    RowSep('single','green','1px')};

q.HAlign='center';% 居中对齐
q.TableEntriesHAlign='center';
q.TableEntriesVAlign='middle';
append(d,q);

%% 插入图片
append(d,Heading(2,'四、图片模板'));
p1 = Image('myPlot_img.png');
% ScaleToFit是为了使图片大小适应页面，也可以换成下方的自定义大小设置
%p1.Style={HAlign('center'),ScaleToFit(1)};
p1.Style={HAlign('center'),Width('600px'),Height('400px')};
append(d,p1);

close(d);
rptview(d.OutputPath);

